"""CLI commands for inference."""

from pathlib import Path

import click
import torch
import yaml

from sr_engine.engine.inference import infer_image, infer_video
from sr_engine.models.registry import build_model
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
              help="Model instance name. Resolves latest version.")
@click.option("--version", type=str, default=None,
              help="Version tag to use (e.g. 'v2'). Defaults to latest.")
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
    version: str | None,
) -> None:
    """Run super-resolution inference on an image or video.

    Provide --model <path> to use a specific checkpoint, or
    --instance to auto-resolve the latest model version.
    """
    loaded_model = None
    model_scale = None

    if inst:
        ws = require_workspace(ctx)
        model_inst = ws.get_model_instance(inst)
        inst_cfg = yaml.safe_load(
            (model_inst.path / "config.yaml").read_text(encoding="utf-8")
        )

        v_path = ws.resolve_version(inst, version)
        if not v_path:
            raise click.ClickException(
                f"No versions found for instance '{inst}'. "
                "Train it first or use --model <path>."
            )

        state_dict = torch.load(v_path, weights_only=True, map_location="cpu")
        loaded_model = build_model(inst_cfg["name"], inst_cfg)
        loaded_model.load_state_dict(state_dict)
        loaded_model = loaded_model.to(device).eval()
        model_scale = int(inst_cfg.get("scale", 4))
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
        result = infer_video(
            model_checkpoint=model, input_path=input_path,
            output_path=output, tile_size=tile,
            tile_overlap=overlap, device=device,
            reporter=resolve_reporter(unit="fr"),
            model=loaded_model, scale=model_scale,
        )
    else:
        result = infer_image(
            model_checkpoint=model, input_path=input_path,
            output_path=output, tile_size=tile,
            tile_overlap=overlap, device=device,
            model=loaded_model, scale=model_scale,
        )

    click.echo(f"Output written to: {result}")
