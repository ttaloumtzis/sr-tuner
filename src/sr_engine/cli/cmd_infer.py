"""CLI commands for inference."""

from pathlib import Path

import click

from sr_engine.engine.inference import infer_image, infer_video
from .helpers import resolve_reporter, require_workspace


@click.group()
def infer() -> None:
    """Inference commands (image or video)."""


@infer.command()
@click.option("--model", "-m", type=click.Path(exists=True, path_type=Path),
              help="Model checkpoint path. Required without --instance.")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True, path_type=Path),
              help="Input image or video file.")
@click.option("--output", "-o", required=True, type=click.Path(path_type=Path),
              help="Output image or video path.")
@click.option("--tile", type=int, default=512, show_default=True,
              help="Tile size for VRAM-safe tiled inference (0 = no tiling).")
@click.option("--overlap", type=int, default=64, show_default=True,
              help="Overlap between tiles in pixels (must be less than --tile).")
@click.option("--device", default="cuda", show_default=True,
              type=click.Choice(["cuda", "cpu", "auto"]),
              help="Device to run inference on.")
@click.option("--instance", "inst", type=str, default=None,
              help="Model instance name. Auto-resolves latest checkpoint.")
@click.pass_context
def run(
    ctx,
    model: Path | None,
    input_path: Path,
    output: Path,
    tile: int,
    overlap: int,
    device: str,
    inst: str | None,
) -> None:
    """Run super-resolution inference on an image or video.

    Provide --model <path> to use a specific checkpoint, or
    --instance to auto-resolve the latest checkpoint.
    """
    if inst:
        ws = require_workspace(ctx)
        model_inst = ws.get_model_instance(inst)
        ckpts = sorted(model_inst.path.glob("checkpoints/*.pt"))
        if not ckpts:
            raise click.ClickException(f"No checkpoints in instance '{inst}'")
        model = ckpts[-1]
    elif not model:
        raise click.ClickException(
            "Provide --model <path> or --instance"
        )

    if tile > 0 and overlap >= tile:
        raise click.BadParameter(
            f"overlap ({overlap}) must be less than tile ({tile})"
        )
    suffix = input_path.suffix.lower()
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".ts"}

    if suffix in video_extensions:
        result = infer_video(model, input_path, output, tile, overlap, device,
                             reporter=resolve_reporter(unit="fr"))
    else:
        result = infer_image(model, input_path, output, tile, overlap, device)

    click.echo(f"Output written to: {result}")
