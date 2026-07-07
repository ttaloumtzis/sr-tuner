"""CLI commands for model utilities (export, info)."""

from pathlib import Path

import click

from sr_engine.models.checkpoint import (
    load_checkpoint,
    export_to_safetensors,
    export_to_onnx,
    export_to_torchscript,
)
from sr_engine.utils.config import DefaultConfigs, load_config


@click.group()
def model() -> None:
    """Model utility commands (export, inspect)."""


@model.command()
@click.option("--model-name", "-n", required=True, help="Model name (e.g., 'swinir').")
@click.option("--ckpt", "-c", required=True, type=click.Path(exists=True, path_type=Path),
              help="Path to the model checkpoint.")
@click.option("--format", "-f", "fmt", required=True,
              type=click.Choice(["onnx", "safetensors", "torchscript"]),
              help="Export format.")
@click.option("--out", "-o", required=True, type=click.Path(path_type=Path),
              help="Output path.")
def export_cmd(model_name: str, ckpt: Path, fmt: str, out: Path) -> None:
    """Export a model checkpoint using its architecture configuration."""

    # 1. Access your architecture config via the loader
    cfg_loader = DefaultConfigs()
    model_arch_cfg = cfg_loader.models.get(model_name)

    if not model_arch_cfg:
        raise click.ClickException(f"Unknown model: {model_name}")

    # 2. Logic to handle export
    # You could potentially pass model_arch_cfg to these functions if they need
    # to know about input shapes/types for ONNX/TorchScript exports
    export_map = {
        "safetensors": export_to_safetensors,
        "onnx": export_to_onnx,
        "torchscript": export_to_torchscript
    }

    export_map[fmt](ckpt, out)
    click.echo(f"Model '{model_name}' exported to {out} as {fmt}")


@model.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True, path_type=Path),
              help="Model checkpoint path.")
def info(model: Path) -> None:
    """Display information about a model checkpoint."""
    ckpt = load_checkpoint(model)
    click.echo(f"Checkpoint: {model}")
    click.echo(f"  Step:      {ckpt.get('step', 'unknown')}")
    click.echo(f"  Config:    {ckpt.get('config', 'not saved')}")
