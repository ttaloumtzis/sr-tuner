"""CLI commands for model utilities (export, info)."""

from pathlib import Path

import click

from sr_engine.models.checkpoint import (
    load_checkpoint,
    export_to_safetensors,
    export_to_onnx,
    export_to_torchscript,
)


@click.group()
def model() -> None:
    """Model utility commands (export, inspect)."""


@model.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True, path_type=Path),
              help="Model checkpoint path.")
@click.option("--format", "-f", "fmt", required=True,
              type=click.Choice(["onnx", "safetensors", "torchscript"]),
              help="Export format.")
@click.option("--out", "-o", required=True, type=click.Path(path_type=Path),
              help="Output path.")
def export_cmd(model: Path, fmt: str, out: Path) -> None:
    """Export a model checkpoint to the specified format."""
    if fmt == "safetensors":
        export_to_safetensors(model, out)
    elif fmt == "onnx":
        export_to_onnx(model, out)
    elif fmt == "torchscript":
        export_to_torchscript(model, out)
    click.echo(f"Model exported to: {out}")


@model.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True, path_type=Path),
              help="Model checkpoint path.")
def info(model: Path) -> None:
    """Display information about a model checkpoint."""
    ckpt = load_checkpoint(model)
    click.echo(f"Checkpoint: {model}")
    click.echo(f"  Step:      {ckpt.get('step', 'unknown')}")
    click.echo(f"  Config:    {ckpt.get('config', 'not saved')}")
