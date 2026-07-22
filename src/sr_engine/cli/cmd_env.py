"""CLI commands for environment diagnostics."""

import click
import torch

from sr_engine.device.backend import get_device, is_rocm, autocast_dtype, supports_flash_attn
from .helpers import make_workspace_config_loader, resolve_model_config, no_workspace_config_option

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
@click.option("--model", "-m", default="rrdb_esrgan", help="Model name (e.g., 'swinir').")
@click.option("--iterations", "-n", type=int, default=10, show_default=True,
              help="Number of timed iterations for the benchmark.")
@no_workspace_config_option
@click.pass_context
def bench(ctx, model: str, iterations: int, no_workspace_config: bool) -> None:
    """Run a micro-benchmark (forward+backward) and report throughput statistics."""
    from sr_engine.models.registry import build_model
    import statistics
    import time

    try:
        _, cfg_loader = make_workspace_config_loader(ctx, no_workspace_config)
        model_cfg = resolve_model_config(cfg_loader, model)

        net = build_model(model, model_cfg)
        device = get_device()
        net = net.to(device).train()

        dummy = torch.randn(1, 3, 128, 128, device=device)

        # Warm-up iterations (forward only)
        WARMUP = 3
        for _ in range(WARMUP):
            _ = net(dummy)

        # Timed iterations (forward + backward)
        times = []
        for _ in range(iterations):
            net.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            out = net(dummy)
            loss = out.sum()
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        times_ms = [t * 1000 for t in times]
        mean = statistics.mean(times_ms)
        median = statistics.median(times_ms)
        stddev = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        min_t = min(times_ms)
        max_t = max(times_ms)

        click.echo(f"Benchmark complete on {device}")
        click.echo(f"Model:                     {model}")
        click.echo(f"Iterations:                {iterations} (+ {WARMUP} warm-up)")
        click.echo(f"Mean execution time:       {mean:.2f} ms")
        click.echo(f"Median execution time:     {median:.2f} ms")
        click.echo(f"Std deviation:             {stddev:.2f} ms")
        click.echo(f"Min / Max:                 {min_t:.2f} / {max_t:.2f} ms")
        click.echo(f"Output shape:              {list(out.shape)}")

    except (ValueError, KeyError, RuntimeError) as e:
        click.secho(f"Benchmark failed: {e}", fg="red")
        raise click.Abort()