"""Inference engine — run a model on images or videos."""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from sr_engine.utils.progress import ProgressReporter

from sr_engine.engine.tiling import tile_image, stitch_tiles
from sr_engine.models.checkpoint import load_checkpoint
from sr_engine.models.registry import build_model


def load_model(model_checkpoint: Path, device: str) -> tuple[torch.nn.Module, int]:
    """Load a checkpoint, rebuild the model architecture, and load its weights.

    Returns the model (in eval mode, moved to *device*) and its configured
    scale factor.
    """
    checkpoint = load_checkpoint(model_checkpoint, map_location="cpu")
    config = checkpoint.get("config")
    if not config or "name" not in config:
        raise ValueError(
            f"Checkpoint at '{model_checkpoint}' has no usable 'config' (with a "
            f"'name' key) — cannot reconstruct the model architecture."
        )

    model = build_model(config["name"], config)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()

    scale = int(config.get("scale", 4))
    return model, scale

# Backward-compatible alias
_load_model = load_model


def _read_image_tensor(path: Path) -> torch.Tensor:
    """Load an image as a float32 CHW tensor in [0, 1], RGB order."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
    return tensor.permute(2, 0, 1).contiguous()


def _frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """Convert a raw BGR video frame (as read by cv2) to a float32 CHW tensor in [0, 1]."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
    return tensor.permute(2, 0, 1).contiguous()


def _tensor_to_bgr_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW float tensor in [0, 1] back to a uint8 BGR numpy image."""
    tensor = tensor.clamp(0.0, 1.0).detach().cpu()
    img = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _super_resolve_tensor(
    model: torch.nn.Module,
    lr_tensor: torch.Tensor,
    scale: int,
    tile_size: int,
    tile_overlap: int,
    device: str,
) -> torch.Tensor:
    """Run *model* on a single LR image tensor, tiling if needed, and return the HR tensor."""
    _, h, w = lr_tensor.shape

    if tile_size <= 0 or (h <= tile_size and w <= tile_size):
        with torch.no_grad():
            hr_tensor = model(lr_tensor.unsqueeze(0).to(device))[0]
        return hr_tensor.cpu()

    lr_tiles = tile_image(lr_tensor, tile_size, tile_overlap)

    hr_tiles: list[tuple[torch.Tensor, tuple[int, int]]] = []
    with torch.no_grad():
        for tile, (row, col) in lr_tiles:
            output = model(tile.unsqueeze(0).to(device))[0].cpu()
            # Tile positions are in LR pixel space — scale them up to match
            # the HR output resolution before stitching.
            hr_tiles.append((output, (row * scale, col * scale)))

    output_size = (h * scale, w * scale)
    return stitch_tiles(hr_tiles, output_size, tile_overlap * scale)


def infer_image(
    model_checkpoint: Path | None = None,
    input_path: Path | None = None,
    output_path: Path | None = None,
    tile_size: int = 512,
    tile_overlap: int = 64,
    device: str = "cuda",
    model: torch.nn.Module | None = None,
    scale: int | None = None,
) -> Path:
    """Run super-resolution inference on a single image.

    Args:
        model_checkpoint: Path to the model checkpoint (alternative to model).
        input_path: Input image path.
        output_path: Output image path.
        tile_size: Tile size for VRAM-safe tiled inference.
        tile_overlap: Overlap between tiles in pixels.
        device: Torch device string.
        model: Pre-loaded model (alternative to model_checkpoint).
        scale: Model scale factor (required if model is pre-loaded).

    Returns:
        Path to the output image.
    """
    if model is None:
        model, scale = load_model(model_checkpoint, device)
    else:
        assert scale is not None, "scale is required when passing a pre-loaded model"

    lr_tensor = _read_image_tensor(input_path)
    hr_tensor = _super_resolve_tensor(model, lr_tensor, scale, tile_size, tile_overlap, device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), _tensor_to_bgr_image(hr_tensor))

    return output_path


def infer_video(
    model_checkpoint: Path | None = None,
    input_path: Path | None = None,
    output_path: Path | None = None,
    tile_size: int = 512,
    tile_overlap: int = 64,
    device: str = "cuda",
    reporter: Optional[ProgressReporter] = None,
    model: torch.nn.Module | None = None,
    scale: int | None = None,
) -> Path:
    """Run super-resolution inference on a video file frame-by-frame.

    Args:
        model_checkpoint: Path to the model checkpoint (alternative to model).
        input_path: Input video path.
        output_path: Output video path.
        tile_size: Tile size for VRAM-safe tiled inference.
        tile_overlap: Overlap between tiles in pixels.
        device: Torch device string.
        reporter: Optional progress reporter.
        model: Pre-loaded model (alternative to model_checkpoint).
        scale: Model scale factor (required if model is pre-loaded).

    Returns:
        Path to the output video.
    """
    if model is None:
        model, scale = load_model(model_checkpoint, device)
    else:
        assert scale is not None, "scale is required when passing a pre-loaded model"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.mp4")

    reporter = reporter or ProgressReporter()
    reporter.start(total=frame_count if frame_count > 0 else None,
                   desc="Super-resolving frames")

    writer = None
    try:
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break

            lr_tensor = _frame_to_tensor(frame_bgr)
            hr_tensor = _super_resolve_tensor(
                model, lr_tensor, scale, tile_size, tile_overlap, device
            )
            output_bgr = _tensor_to_bgr_image(hr_tensor)

            if writer is None:
                out_h, out_w = output_bgr.shape[:2]
                writer = None
                for codec in ("avc1", "mp4v"):
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (out_w, out_h))
                    if writer.isOpened():
                        break
                    writer = None
                if writer is None:
                    raise RuntimeError(
                        f"Could not open video writer for: {output_path} "
                        "(tried 'avc1' and 'mp4v' codecs)"
                    )

            writer.write(output_bgr)
            reporter.update(1)
    finally:
        reporter.finish()
        cap.release()
        if writer is not None:
            writer.release()
        if tmp_path.exists():
            os.replace(str(tmp_path), str(output_path))

    return output_path