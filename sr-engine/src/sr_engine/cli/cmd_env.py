"""CLI commands for environment diagnostics."""

from pathlib import Path
import click
import torch

from sr_engine.device.backend import get_device, is_rocm, autocast_dtype, supports_flash_attn
from sr_engine.utils.config import load_config

@click.group()
def env() -> None:
    """Environment diagnostic commands."""

@env.command()
def check() -> None:
    """Check the current environment and print a report."""
    device = get_device()
    click.echo(f"PyTorch version:  {torch.__version__}")
    click.echo(f"Detected device:  {device}")

    is_cuda = torch.cuda.is_available()
    click.echo(f"CUDA/ROCm avail:  {is_cuda}")

    if is_cuda:
        dev_idx = torch.cuda.current_device()
        click.echo(f"Device name:      {torch.cuda.get_device_name(dev_idx)}")
        click.echo(f"VRAM total:       {torch.cuda.get_device_properties(dev_idx).total_memory // 1024**2} MB")
        click.echo(f"BF16 support:     {torch.cuda.is_bf16_supported()}")

    click.echo(f"ROCm backend:     {is_rocm()}")
    click.echo(f"Autocast dtype:   {autocast_dtype()}")
    click.echo(f"Flash attention:  {supports_flash_attn()}")

@env.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True, path_type=Path),
              help="Model config YAML to benchmark.")
def bench(model: Path) -> None:
    """Run a micro-benchmark (forward+backward pass) and report throughput."""
    from sr_engine.models.registry import build_model
    import time

    try:
        model_cfg = load_config(model)
        net = build_model(model_cfg["name"], model_cfg)
        device = get_device()
        net = net.to(device).train() # Ensure in training mode

        # Warm-up pass
        dummy = torch.randn(1, 3, 128, 128, device=device)
        _ = net(dummy)

        # Benchmark pass
        start_time = time.perf_counter()
        out = net(dummy)
        loss = out.sum()
        loss.backward()
        end_time = time.perf_counter()

        click.echo(f"Benchmark complete on {device}")
        click.echo(f"Execution time:   {(end_time - start_time) * 1000:.2f} ms")
        click.echo(f"Output shape:     {list(out.shape)}")

    except Exception as e:
        click.secho(f"Benchmark failed: {e}", fg="red")