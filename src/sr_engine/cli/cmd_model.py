"""CLI commands for model utilities (export, info, instances)."""

import sys
from pathlib import Path

import click
import yaml

from sr_engine.models.checkpoint import (
    load_checkpoint,
    export_to_safetensors,
    export_to_onnx,
    export_to_torchscript,
)
from .helpers import (
    make_workspace_config_loader,
    resolve_model_config,
    no_workspace_config_option,
    require_workspace,
)


@click.group()
def model() -> None:
    """Model utility commands (export, inspect, manage instances)."""


# ── Instance management ─────────────────────────────────────────────


@model.command()
@click.argument("name")
@click.option("--model", "-m", "arch", required=True, help="Model architecture name (e.g., 'swinir').")
@click.pass_context
def create_instance(ctx, name: str, arch: str) -> None:
    """Create a named model instance in the workspace.

    NAME is the identifier for this model instance (checkpoints,
    runs, and configs are stored under this name).

    The instance stores a frozen architecture config, its own checkpoint
    history, and per-training-run metadata.
    """
    ws = require_workspace(ctx)
    _, cfg_loader = make_workspace_config_loader(ctx, ws=ws)
    arch_config = resolve_model_config(cfg_loader, arch)

    ws.create_model_instance(name, arch_config)
    click.echo(f"Created model instance '{name}' (arch: {arch})")


@model.command()
@click.pass_context
def list_instances(ctx) -> None:
    """List model instances in the workspace."""
    ws = require_workspace(ctx)
    instances = ws.list_model_instances()
    if not instances:
        click.echo("No model instances in workspace.")
        return
    click.echo("Model instances:")
    for inst in instances:
        ckpts = len(list(inst.path.glob("checkpoints/*.pt")))
        runs = len(list(inst.path.glob("runs/run_*")))
        click.echo(f"  {inst.name}  ({ckpts} checkpoints, {runs} runs)")


@model.command()
@click.option("--instance", "-i", required=True, help="Model instance name.")
@click.pass_context
def list_runs(ctx, instance: str) -> None:
    """List training runs for a model instance."""
    ws = require_workspace(ctx)
    run_dirs = ws.list_runs(instance)
    if not run_dirs:
        click.echo(f"No runs for instance '{instance}'.")
        return
    click.echo(f"Runs for '{instance}':")
    for d in run_dirs:
        tc = d / "train_config.yaml"
        has_metrics = (d / "metrics.jsonl").exists()
        summary = ""
        if tc.exists():
            tc_data = yaml.safe_load(tc.read_text(encoding="utf-8"))
            summary = f"  arch={tc_data.get('model', '?')}  max_epochs={tc_data.get('max_epochs', '?')}"
        click.echo(f"  {d.name}  {'[metrics]' if has_metrics else ''}{summary}")


# ── Export ──────────────────────────────────────────────────────────


@model.command()
@click.option("--model-name", "-m", help="Model name (e.g., 'swinir'). Required without --instance.")
@click.option("--ckpt", "-c", type=click.Path(exists=True, path_type=Path),
              help="Path to the model checkpoint. Required without --instance.")
@click.option("--format", "-f", "fmt", required=True,
              type=click.Choice(["onnx", "safetensors", "torchscript"]),
              help="Export format.")
@click.option("--out", "-o", required=True, type=click.Path(path_type=Path),
              help="Output path.")
@click.option("--instance", "-i", type=str, default=None,
              help="Model instance name. Resolves checkpoint and arch config automatically.")
@no_workspace_config_option
@click.pass_context
def export_cmd(ctx, model_name: str | None, ckpt: Path | None, fmt: str, out: Path,
               instance: str | None, no_workspace_config: bool) -> None:
    """Export a model checkpoint.

    When --instance is given, the checkpoint and model name
    are resolved from the instance automatically.
    """
    if instance:
        ws = require_workspace(ctx)
        model_inst = ws.get_model_instance(instance)
        inst_cfg = yaml.safe_load(
            (model_inst.path / "config.yaml").read_text(encoding="utf-8")
        )
        model_name = inst_cfg.get("name") or model_name
        ckpts = sorted(model_inst.path.glob("checkpoints/*.pt"))
        if not ckpts:
            raise click.ClickException(f"No checkpoints in instance '{instance}'")
        ckpt = ckpts[-1]
    elif not model_name or not ckpt:
        raise click.ClickException(
            "--model-name and --ckpt are required without --instance"
        )

    _, cfg_loader = make_workspace_config_loader(ctx, no_workspace_config)
    resolve_model_config(cfg_loader, model_name)

    export_map = {
        "safetensors": export_to_safetensors,
        "onnx": export_to_onnx,
        "torchscript": export_to_torchscript,
    }

    export_map[fmt](ckpt, out)
    click.echo(f"Model '{model_name}' exported to {out} as {fmt}")


# ── Info ────────────────────────────────────────────────────────────


@model.command()
@click.option("--model", "-m", type=click.Path(exists=True, path_type=Path),
              help="Model checkpoint path. Alternative to --instance.")
@click.option("--instance", "-i", type=str, default=None,
              help="Model instance name. Shows arch config, checkpoints, runs.")
@click.pass_context
def info(ctx, model: Path | None, instance: str | None) -> None:
    """Display information about a model checkpoint or instance.

    Provide --model <path> for checkpoint-level info, or
    --instance for instance-level info (arch config, checkpoints, runs).
    """
    if instance:
        ws = require_workspace(ctx)
        model_inst = ws.get_model_instance(instance)

        click.echo(f"Instance:   {instance}")
        click.echo(f"Path:       {model_inst.path}")

        cfg = yaml.safe_load(
            (model_inst.path / "config.yaml").read_text(encoding="utf-8")
        )
        click.echo(f"Arch config:")
        for k, v in cfg.items():
            click.echo(f"  {k}: {v}")

        ckpts = sorted(model_inst.path.glob("checkpoints/*.pt"))
        click.echo(f"\nCheckpoints ({len(ckpts)}):")
        for c in ckpts:
            stat = c.stat()
            click.echo(f"  {c.name}  ({stat.st_size / 1024:.0f} KB)")

        runs = sorted(
            d for d in (model_inst.path / "runs").iterdir()
            if d.is_dir() and d.name.startswith("run_")
        )
        click.echo(f"\nRuns ({len(runs)}):")
        for d in runs:
            tc = d / "train_config.yaml"
            has_metrics = (d / "metrics.jsonl").exists()
            summary = ""
            if tc.exists():
                tc_data = yaml.safe_load(tc.read_text(encoding="utf-8"))
                summary = f"  max_epochs={tc_data.get('max_epochs', '?')}"
            click.echo(f"  {d.name}  {'[metrics]' if has_metrics else ''}{summary}")
    elif model:
        ckpt_data = load_checkpoint(model)
        click.echo(f"Checkpoint: {model}")
        click.echo(f"  Step:      {ckpt_data.get('step', 'unknown')}")
        click.echo(f"  Config:    {ckpt_data.get('config', 'not saved')}")
    else:
        raise click.ClickException(
            "Provide --model <path> or --instance"
        )
