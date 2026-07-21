"""HR -> LR degradation pipeline (blur, noise, JPEG, JPEG2000, downsample)."""

import concurrent.futures
from concurrent.futures.process import BrokenProcessPool
from functools import partial
import logging
import multiprocessing
import os
import random
from pathlib import Path
from typing import Any, Optional

from sr_engine.utils.progress import ProgressReporter
import cv2
import numpy as np
from numpy import dtype, ndarray

logger = logging.getLogger(__name__)


def _init_worker() -> None:
    """Initializer to ensure isolated RNG seeds and prevent CPU core oversubscription."""
    cv2.setNumThreads(1)

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

    vals = 255.0
    noisy = np.random.poisson(img_float * vals * scale) / (vals * scale)
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)


def _add_salt_pepper_noise(
        image: np.ndarray,
        amount: float,
        salt_vs_pepper: float = 0.5,
) -> np.ndarray:
    noisy = image.copy()
    num_salt = int(np.ceil(amount * image.size * 0.33 * salt_vs_pepper))
    num_pepper = int(np.ceil(amount * image.size * 0.33 * (1.0 - salt_vs_pepper)))

    if num_salt > 0:
        coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
        noisy[coords[0], coords[1], :] = 255

    if num_pepper > 0:
        coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
        noisy[coords[0], coords[1], :] = 0

    return noisy


def _apply_gaussian_blur(
        image: np.ndarray,
        kernel_size: int = 21,
        sigma_range: list[float] | None = None,
) -> np.ndarray:
    if sigma_range is None:
        sigma_range = [0.1, 3.0]
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def _apply_motion_blur(
        image: np.ndarray,
        max_kernel_size: int = 31,
) -> np.ndarray:
    kernel_size = random.randint(3, max_kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    angle = random.uniform(0, 180)

    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = np.ones(kernel_size)

    M = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel = kernel / np.sum(kernel)

    return cv2.filter2D(image, -1, kernel)


def _apply_jpeg_compression(
        image: np.ndarray,
        quality_range: list[int]
) -> ndarray[tuple[Any, ...], dtype[Any]] | None | Any:
    quality = random.randint(int(quality_range[0]), int(quality_range[1]))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, fencing = cv2.imencode('.jpg', image, encode_param)
    if not success:
        return image
    return cv2.imdecode(fencing, 1)


def _apply_jpeg2000_compression(
        image: np.ndarray,
        quality_range: list[int]
) -> ndarray[tuple[Any, ...], dtype[Any]] | None | Any:
    quality = random.randint(int(quality_range[0]), int(quality_range[1]))
    encode_param = [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), quality]
    success, fencing = cv2.imencode('.jp2', image, encode_param)
    if not success:
        return image
    return cv2.imdecode(fencing, 1)


def _apply_color_jitter(
        image: np.ndarray,
        hue_range: list[float],
        saturation_range: list[float],
        value_range: list[float],
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    h_delta = random.uniform(hue_range[0], hue_range[1]) * 180
    s_delta = random.uniform(saturation_range[0], saturation_range[1]) * 255
    v_delta = random.uniform(value_range[0], value_range[1]) * 255

    hsv[:, :, 0] = (hsv[:, :, 0] + h_delta) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + s_delta, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + v_delta, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _degrade_image(
    hr_image: np.ndarray,
    scale: int,
    color_jitter_kwargs: dict | None = None,
    blur_kwargs: dict | None = None,
    noise_kwargs: dict | None = None,
    jpeg_kwargs: dict | None = None,
    jpeg2000_kwargs: dict | None = None,
    resize_method: str = "area",
    resize_antialias: bool = True,
) -> ndarray[tuple[Any, ...], dtype[Any]] | None | Any:
    img = hr_image.copy()

    height, width = img.shape[:2]
    if height < scale or width < scale:
        logger.warning(
            "[degrade] Image dimensions (%dx%d) are smaller than scale factor %d",
            width, height, scale,
        )
        return None
    height -= height % scale
    width -= width % scale
    img = img[:height, :width]

    # Color jitter — applied before blur as a global color transform
    if color_jitter_kwargs and color_jitter_kwargs.get("enabled", True):
        if random.random() < color_jitter_kwargs.get("prob", 1.0):
            img = _apply_color_jitter(
                img,
                color_jitter_kwargs.get("hue_range", [-0.05, 0.05]),
                color_jitter_kwargs.get("saturation_range", [-0.3, 0.3]),
                color_jitter_kwargs.get("value_range", [-0.3, 0.3]),
            )

    # Blur stage — gaussian and/or motion blur (mutually exclusive)
    if blur_kwargs and blur_kwargs.get("enabled", True):
        gauss_cfg = blur_kwargs.get("gaussian")
        motion_cfg = blur_kwargs.get("motion", {})

        # Support old flat config format: {"kernel_size": ..., "sigma": ..., "prob": ...}
        if gauss_cfg is None and "kernel_size" in blur_kwargs:
            gauss_cfg = blur_kwargs

        use_gauss = gauss_cfg is not None and random.random() < gauss_cfg.get("prob", 1.0)
        use_motion = bool(motion_cfg) and random.random() < motion_cfg.get("prob", 0.5)

        if use_gauss and use_motion:
            if random.random() < 0.5:
                img = _apply_gaussian_blur(
                    img,
                    gauss_cfg.get("kernel_size", 21),
                    gauss_cfg.get("sigma", [0.1, 3.0]),
                )
            else:
                img = _apply_motion_blur(img, motion_cfg.get("max_kernel_size", 31))
        elif use_gauss:
            img = _apply_gaussian_blur(
                img,
                gauss_cfg.get("kernel_size", 21),
                gauss_cfg.get("sigma", [0.1, 3.0]),
            )
        elif use_motion:
            img = _apply_motion_blur(img, motion_cfg.get("max_kernel_size", 31))

    # Antialias pre-filter before downsampling
    if resize_antialias and resize_method != "area":
        sigma = 0.5
        k_size = max(3, int(2 * int(3 * sigma)) + 1)
        img = cv2.GaussianBlur(img, (k_size, k_size), sigmaX=sigma, sigmaY=sigma)

    interp_map = {
        "bilinear": cv2.INTER_LINEAR,
        "lanczos": cv2.INTER_LANCZOS4,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
    }
    interpolation = interp_map.get(resize_method.lower(), cv2.INTER_AREA)

    lr_width = width // scale
    lr_height = height // scale
    img = cv2.resize(img, (lr_width, lr_height), interpolation=interpolation)

    # Noise stage — gaussian, poisson, salt & pepper (mutually exclusive)
    if noise_kwargs and noise_kwargs.get("enabled", True):
        gauss_cfg = noise_kwargs.get("gaussian", {})
        poiss_cfg = noise_kwargs.get("poisson", {})
        sp_cfg = noise_kwargs.get("salt_pepper", {})

        use_gauss = random.random() < gauss_cfg.get("prob", 0.5)
        use_poiss = random.random() < poiss_cfg.get("prob", 0.5)
        use_sp = random.random() < sp_cfg.get("prob", 0.3)

        chosen = []
        if use_gauss:
            chosen.append("gauss")
        if use_poiss:
            chosen.append("poiss")
        if use_sp:
            chosen.append("sp")

        if chosen:
            pick = random.choice(chosen)
            if pick == "gauss":
                img = _add_gaussian_noise(img, gauss_cfg.get("sigma_range", [1, 30]))
            elif pick == "poiss":
                img = _add_poisson_noise(img, poiss_cfg.get("scale_range", [0.05, 3.0]))
            elif pick == "sp":
                img = _add_salt_pepper_noise(
                    img,
                    sp_cfg.get("amount", 0.01),
                    sp_cfg.get("salt_vs_pepper", 0.5),
                )

    # JPEG compression
    if jpeg_kwargs and jpeg_kwargs.get("enabled", True):
        if random.random() < jpeg_kwargs.get("prob", 1.0):
            img = _apply_jpeg_compression(img, jpeg_kwargs.get("quality_range", [30, 95]))

    # JPEG2000 compression
    if jpeg2000_kwargs and jpeg2000_kwargs.get("enabled", True):
        if random.random() < jpeg2000_kwargs.get("prob", 1.0):
            img = _apply_jpeg2000_compression(img, jpeg2000_kwargs.get("quality_range", [30, 95]))

    return img


def _process_single_frame(
        hr_path: Path,
        lr_dir: Path,
        scale: int,
        kwargs: dict
) -> tuple[Path, Path | None]:
    hr_img = cv2.imread(str(hr_path))
    if hr_img is None:
        logger.warning("[degrade] Skipping unreadable frame: %s", hr_path)
        return hr_path, None

    lr_img = _degrade_image(hr_img, scale, **kwargs)

    if lr_img is None or lr_img.size == 0:
        logger.warning("[degrade] Degradation produced empty image, skipping: %s", hr_path)
        return hr_path, None

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
    lr_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[Path, Path]] = []

    if not hr_paths:
        return pairs

    deg_cfg = config.get("degradation", {})
    resize_cfg = deg_cfg.get("resize", {})
    degrade_kwargs = {
        "color_jitter_kwargs": deg_cfg.get("color_jitter"),
        "blur_kwargs": deg_cfg.get("blur"),
        "noise_kwargs": deg_cfg.get("noise"),
        "jpeg_kwargs": deg_cfg.get("jpeg"),
        "jpeg2000_kwargs": deg_cfg.get("jpeg2000"),
        "resize_method": resize_cfg.get("method", "area"),
        "resize_antialias": resize_cfg.get("antialias", True),
    }

    worker = partial(_process_single_frame, lr_dir=lr_dir, scale=scale, kwargs=degrade_kwargs)

    reporter = reporter or ProgressReporter()
    reporter.start(total=len(hr_paths), desc="Degrading Dataset Frames")

    with concurrent.futures.ProcessPoolExecutor(
        initializer=_init_worker,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        results = executor.map(worker, hr_paths)

        try:
            for hr_path, lr_path in results:
                if lr_path is not None:
                    pairs.append((hr_path, lr_path))
                reporter.update(1)
        except (BrokenPipeError, BrokenProcessPool):
            logger.warning(
                "Degradation worker crashed after %d/%d frames. "
                "Returning partial results.",
                len(pairs), len(hr_paths),
            )

    reporter.finish()

    pairs.sort(key=lambda pair: pair[0])
    return pairs
