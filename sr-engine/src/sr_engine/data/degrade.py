"""HR -> LR degradation pipeline (blur, noise, JPEG, downsample)."""

import concurrent.futures
from functools import partial
import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

from sr_engine.utils.progress import ProgressReporter
import cv2
import numpy as np
from numpy import dtype, ndarray

# Setup basic logging to handle silent frame drops cleanly
logger = logging.getLogger(__name__)


def _init_worker() -> None:
    """Initializer to ensure isolated RNG seeds and prevent CPU core oversubscription."""
    # 1. Prevent OpenCV from creating internal thread pools on top of multiprocessing
    cv2.setNumThreads(1)

    # 2. Ensure unique RNG states across distinct system processes
    seed = os.getpid() + int.from_bytes(os.urandom(4), "little")
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


def _add_gaussian_noise(
        image: np.ndarray,
        sigma_range: list[float]
) -> np.ndarray:
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _add_poisson_noise(
        image: np.ndarray,
        scale_range: list[float]
) -> np.ndarray:
    scale = random.uniform(scale_range[0], scale_range[1])
    img_float = image.astype(np.float32) / 255.0

    # Using a fixed dynamic range approximation to decouple noise strength from image content
    vals = 255.0
    noisy = np.random.poisson(img_float * vals * scale) / (vals * scale)
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)


def _apply_jpeg_compression(
        image: np.ndarray,
        quality_range: list[int]
) -> ndarray[tuple[Any, ...], dtype[Any]] | None | Any:
    quality = random.randint(int(quality_range[0]), int(quality_range[1]))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, fencing = cv2.imencode('.jpg', image, encode_param)
    if not success:
        return image  # Fallback gracefully if encoding fails
    return cv2.imdecode(fencing, 1)


def _degrade_image(
    hr_image: np.ndarray,
    scale: int,
    blur_kwargs: dict | None = None,
    noise_kwargs: dict | None = None,
    jpeg_kwargs: dict | None = None,
    resize_method: str = "bicubic",
) -> ndarray[tuple[Any, ...], dtype[Any]] | None | Any:
    """Apply a synthetic degradation pipeline to a high-resolution image.

    Steps: Crop to scale -> blur -> downsample -> noise -> JPEG compression.
    Returns the low-resolution image as a numpy array.
    """
    img = hr_image.copy()

    # 1. Enforce HR dimensions to be perfectly divisible by scale factor
    height, width = img.shape[:2]
    height -= height % scale
    width -= width % scale
    img = img[:height, :width]

    # 2. Blur (Applied to HR to simulate lens blur/anti-aliasing filters)
    if blur_kwargs and random.random() < blur_kwargs.get("prob", 1.0):
        k_size = blur_kwargs.get("kernel_size", 21)
        if k_size % 2 == 0:
            k_size += 1
        sigma_range = blur_kwargs.get("sigma", [0.1, 3.0])
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        img = cv2.GaussianBlur(img, (k_size, k_size), sigmaX=sigma, sigmaY=sigma)

    # 3. Downsampling (Happens BEFORE noise/JPEG so artifacts remain intact at LR resolution)
    interp_map = {
        "bilinear": cv2.INTER_LINEAR,
        "lanczos": cv2.INTER_LANCZOS4,
        "bicubic": cv2.INTER_CUBIC
    }
    interpolation = interp_map.get(resize_method.lower(), cv2.INTER_CUBIC)

    lr_width = width // scale
    lr_height = height // scale
    img = cv2.resize(img, (lr_width, lr_height), interpolation=interpolation)

    # 4. Synthesize Sensor Noise at LR scale
    if noise_kwargs:
        gauss_cfg = noise_kwargs.get("gaussian", {})
        poiss_cfg = noise_kwargs.get("poisson", {})
        use_gauss = random.random() < gauss_cfg.get("prob", 0.5)
        use_poiss = random.random() < poiss_cfg.get("prob", 0.5)

        if use_gauss and use_poiss:
            if random.random() < 0.5:
                img = _add_gaussian_noise(img, gauss_cfg.get("sigma_range", [1, 30]))
            else:
                img = _add_poisson_noise(img, poiss_cfg.get("scale_range", [0.05, 3.0]))
        elif use_gauss:
            img = _add_gaussian_noise(img, gauss_cfg.get("sigma_range", [1, 30]))
        elif use_poiss:
            img = _add_poisson_noise(img, poiss_cfg.get("scale_range", [0.05, 3.0]))

    # 5. Synthesize Transmission / Compression Artifacts at LR scale
    if jpeg_kwargs and random.random() < jpeg_kwargs.get("prob", 1.0):
        img = _apply_jpeg_compression(img, jpeg_kwargs.get("quality_range", [30, 95]))

    return img

def _process_single_frame(
        hr_path: Path,
        lr_dir: Path,
        scale: int,
        kwargs: dict
) -> tuple[Path, Path | None]:
    """Helper worker task to read, degrade, and write a single frame.

    Always returns the *hr_path* alongside the result (or None on failure)
    so callers can match results back to their source frame by identity
    rather than by relying on list position/order.
    """
    hr_img = cv2.imread(str(hr_path))
    if hr_img is None:
        logger.warning(f"[degrade] Skipping unreadable frame: {hr_path}")
        return hr_path, None

    lr_img = _degrade_image(hr_img, scale, **kwargs)

    lr_path = lr_dir / hr_path.name
    cv2.imwrite(str(lr_path), lr_img)
    return hr_path, lr_path


def batch_degrade(
    hr_paths: list[Path],
    lr_dir: Path,
    scale: int,
    config: dict,
    reporter: Optional[ProgressReporter] = None,
) -> list[tuple[Path, Path]]:
    """Generate LR images for all HR images in *hr_paths* and write to *lr_dir*.

    Returns a sorted list of ``(hr_path, lr_path)`` pairs for every frame that
    was successfully degraded. Frames that failed to read/decode (see
    ``_process_single_frame``) are simply omitted from the result — callers
    must NOT assume the returned list lines up positionally with *hr_paths*,
    since a dropped frame would otherwise silently shift every pair after it.
    """
    lr_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[Path, Path]] = []

    if not hr_paths:
        return pairs

    # Extract kwargs mapping exactly to the degrade_image signature
    deg_cfg = config.get("degradation", {})
    degrade_kwargs = {
        "blur_kwargs": deg_cfg.get("blur"),
        "noise_kwargs": deg_cfg.get("noise"),
        "jpeg_kwargs": deg_cfg.get("jpeg"),
        "resize_method": deg_cfg.get("resize", {}).get("method", "bicubic")
    }

    # Parallel processing via ProcessPoolExecutor using initialized workers
    worker = partial(_process_single_frame, lr_dir=lr_dir, scale=scale, kwargs=degrade_kwargs)

    reporter = reporter or ProgressReporter()
    reporter.start(total=len(hr_paths), desc="Degrading Dataset Frames")

    with concurrent.futures.ProcessPoolExecutor(initializer=_init_worker) as executor:
        results = executor.map(worker, hr_paths)

        for hr_path, lr_path in results:
            if lr_path is not None:
                pairs.append((hr_path, lr_path))
            reporter.update(1)

    reporter.finish()

    pairs.sort(key=lambda pair: pair[0])
    return pairs