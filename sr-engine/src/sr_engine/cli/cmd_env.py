"""CLI commands for environment diagnostics."""

from pathlib import Path

import click

from sr_engine.device.backend import get_device, is_rocm, autocast_dtype, supports_flash_attn
from sr_engine.utils.config import load_config


@click.group()
def env() -> None:
    """Environment diagnostic commands."""


@env.command()
def check() -> None:
    """Check the current environment and print a report."""
    import torch

    device = get_device()
    click.echo(f"PyTorch version:  {torch.__version__}")
    click.echo(f"Detected device:  {device}")
    click.echo(f"CUDA available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"Device name:      {torch.cuda.get_device_name(device)}")
    click.echo(f"ROCm backend:     {is_rocm()}")
    click.echo(f"Autocast dtype:   {autocast_dtype()}")
    click.echo(f"Flash attention:  {supports_flash_attn()}")
    if torch.cuda.is_available():
        click.echo(f"BF16 support:     {torch.cuda.is_bf16_supported()}")


@env.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True, path_type=Path),
              help="Model config YAML to benchmark.")
def bench(model: Path) -> None:
    """Run a micro-benchmark (forward+backward pass) and report throughput."""
    import torch
    from sr_engine.models.registry import build_model

    model_cfg = load_config(model)
    net = build_model(model_cfg["name"], model_cfg)
    device = get_device()
    net = net.to(device)

    dummy = torch.randn(1, 3, 128, 128, device=device)
    out = net(dummy)
    loss = out.sum()
    loss.backward()

    click.echo(f"Micro-benchmark complete: forward+backward pass on {device}")
    click.echo(f"  Output shape: {out.shape}")
