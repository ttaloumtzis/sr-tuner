"""CLI commands for inference."""

from pathlib import Path

import click

from sr_engine.engine.inference import infer_image, infer_video


@click.group()
def infer() -> None:
    """Inference commands (image or video)."""


@infer.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True, path_type=Path),
              help="Model checkpoint path.")
@click.option("--input", "-i", required=True, type=click.Path(exists=True, path_type=Path),
              help="Input image or video file.")
@click.option("--output", "-o", required=True, type=click.Path(path_type=Path),
              help="Output image or video path.")
@click.option("--tile", type=int, default=512, show_default=True,
              help="Tile size for VRAM-safe tiled inference (0 = no tiling).")
@click.option("--overlap", type=int, default=64, show_default=True,
              help="Overlap between tiles in pixels.")
@click.option("--device", default="cuda", show_default=True,
              help="Device to run inference on.")
def run(
    model: Path,
    input: Path,
    output: Path,
    tile: int,
    overlap: int,
    device: str,
) -> None:
    """Run super-resolution inference on an image or video."""
    suffix = input.suffix.lower()
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".ts"}

    if suffix in video_extensions:
        result = infer_video(model, input, output, tile, overlap, device)
    else:
        result = infer_image(model, input, output, tile, overlap, device)

    click.echo(f"Output written to: {result}")
