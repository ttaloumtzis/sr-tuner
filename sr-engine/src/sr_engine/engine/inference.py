"""Inference engine — run a model on images or videos."""

from pathlib import Path


def infer_image(
    model_checkpoint: Path,
    input_path: Path,
    output_path: Path,
    tile_size: int = 512,
    tile_overlap: int = 64,
    device: str = "cuda",
) -> Path:
    """Run super-resolution inference on a single image.

    Args:
        model_checkpoint: Path to the model checkpoint.
        input_path: Input image path.
        output_path: Output image path.
        tile_size: Tile size for VRAM-safe tiled inference.
        tile_overlap: Overlap between tiles in pixels.
        device: Torch device string.

    Returns:
        Path to the output image.
    """
    raise NotImplementedError("TODO: implement single-image inference")


def infer_video(
    model_checkpoint: Path,
    input_path: Path,
    output_path: Path,
    tile_size: int = 512,
    tile_overlap: int = 64,
    device: str = "cuda",
) -> Path:
    """Run super-resolution inference on a video file frame-by-frame.

    Args:
        model_checkpoint: Path to the model checkpoint.
        input_path: Input video path.
        output_path: Output video path.
        tile_size: Tile size for VRAM-safe tiled inference.
        tile_overlap: Overlap between tiles in pixels.
        device: Torch device string.

    Returns:
        Path to the output video.
    """
    raise NotImplementedError("TODO: implement video inference")
