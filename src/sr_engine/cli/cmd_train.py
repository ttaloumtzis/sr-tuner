"""CLI command for model training."""

import sys
import time
from pathlib import Path
import click
import yaml

from sr_engine.engine.trainer import Trainer
from sr_engine.engine.metrics_stream import MetricsStream
from sr_engine.utils.config import load_config, merge_overrides, validate_config
from .helpers import (make_workspace_config_loader, resolve_model_config,
                      no_workspace_config_option, resolve_reporter,
                      resolve_callbacks, resolve_cancel_check)


@click.group()
def train() -> None:
    """Model training commands."""


@train.command()
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path), help="Training config YAML.")
@click.option("--model", "-m", default="rrdb_esrgan", help="Model name (e.g., 'swinir', 'rrdb_esrgan').")
@click.option("--dataset", "-d", required=True, type=click.Path(path_type=Path), help="Dataset directory path.")
@click.option("--resume", "-r", type=str, default=None,
              help="Resume from a checkpoint. A path, a filename (with --instance), or 'latest'.")
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu", "auto"]),
              help="Training device.")
@click.option("--batch-size", type=int, default=None, help="Batch size for training.")
@click.option("--learning-rate", type=float, default=None, help="Learning rate.")
@click.option("--seed", type=int, default=None, help="Random seed.")
@click.option("--weight-decay", type=float, default=None, help="Adam weight decay.")
@click.option("--betas", type=float, nargs=2, default=None, help="Adam betas (two floats, e.g. --betas 0.9 0.999).")
@click.option("--max-epochs", type=int, default=None, help="Total number of epochs to train.")
@click.option("--num-workers", type=int, default=None, help="Dataloader worker count.")
@click.option("--patch-size", type=int, default=None, help="Training patch size.")
@click.option("--save-per-epoch", type=int, default=None, help="Save checkpoint every N epochs.")
@click.option("--validation-enabled/--no-validation-enabled", default=None, help="Enable/disable validation split.")
@click.option("--validation-split", type=click.FloatRange(0.0, 1.0), default=None, help="Fraction of data for validation.")
@click.option("--machine", is_flag=True, default=False,
              help="Emit metrics as JSON Lines (one JSON object per event) for programmatic consumption.")
@click.option("--experiment-id", type=str, default=None, help="Experiment identifier (auto-generated if omitted).")
@click.option("--metrics-frequency", type=int, default=1, help="Log metrics every N batches.")
@click.option("--bf16/--no-bf16", default=None, help="Enable bfloat16 mixed precision training.")
@click.option("--dump-config", is_flag=True, default=False, help="Print final merged config and exit.")
@no_workspace_config_option
@click.option("--instance", "-i", type=str, default=None,
              help="Model instance name. Overrides checkpoint_dir and creates a run directory.")
@click.pass_context
def run(ctx, config, model, dataset, resume, device, batch_size, learning_rate,
        seed, weight_decay, betas, max_epochs,
        num_workers, patch_size, save_per_epoch,
        validation_enabled, validation_split, machine, experiment_id, metrics_frequency,
        bf16, dump_config, instance, no_workspace_config):
    """Train a super-resolution model."""

    ws, cfg_loader = make_workspace_config_loader(ctx, no_workspace_config)

    if instance:
        if not ws:
            raise click.ClickException(
                "--instance requires a workspace. Initialize one with 'workspace init' "
                "or set --workspace explicitly."
            )
        try:
            model_inst = ws.get_model_instance(instance)
        except FileNotFoundError:
            raise click.ClickException(
                f"Model instance '{instance}' not found in workspace. "
                f"Create it with: sre model create-instance {instance} --model <arch>"
            )

        inst_ckpt_dir = model_inst.path / "checkpoints"
        run_dir = ws.get_run_path(instance)

        if resume:
            resume_path = Path(resume)
            if not resume_path.is_absolute() and '/' not in str(resume_path):
                if resume == "latest":
                    ckpts = sorted(inst_ckpt_dir.glob("*.pt"))
                    if ckpts:
                        resume = str(ckpts[-1])
                    else:
                        resume = None
                else:
                    candidate = inst_ckpt_dir / resume
                    if candidate.suffix != ".pt":
                        candidate = inst_ckpt_dir / f"{resume}.pt"
                    if candidate.exists():
                        resume = str(candidate)

    if ws:
        dataset = ws.resolve_dataset(dataset)

    model_cfg = resolve_model_config(cfg_loader, model)

    train_cfg = load_config(config) if config else cfg_loader.get_train_config()

    overrides = {
        k: v for k, v in {
            "device": device,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "weight_decay": weight_decay,
            "max_epochs": max_epochs,
            "num_workers": num_workers,
            "patch_size": patch_size,
            "save_per_epoch": save_per_epoch,
        }.items() if v is not None
    }
    if betas is not None:
        overrides["betas"] = list(betas)
    if bf16 is not None:
        overrides["dtype"] = "bf16" if bf16 else "float32"
    if validation_enabled is not None:
        overrides.setdefault("validation", {})["enabled"] = validation_enabled
    if validation_split is not None:
        overrides.setdefault("validation", {})["split"] = validation_split

    if instance:
        overrides["checkpoint_dir"] = str(inst_ckpt_dir)
        overrides["model"] = model

    if overrides:
        train_cfg = merge_overrides(train_cfg, overrides)

    validate_config(train_cfg, required_keys=["max_epochs", "batch_size"])

    if dump_config:
        yaml.safe_dump(train_cfg, sys.stdout, default_flow_style=False, sort_keys=False)
        return

    val_cfg = train_cfg.get("validation", {})
    val_enabled = bool(val_cfg.get("enabled", True))
    val_split = float(val_cfg.get("split", 0.1))

    if instance:
        (run_dir / "train_config.yaml").write_text(
            yaml.safe_dump(train_cfg, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

    metrics_stream = None
    if machine:
        if experiment_id is None:
            experiment_id = f"exp_{int(time.time())}"
        if ws and instance:
            metrics_dir = run_dir
        elif ws:
            metrics_dir = ws.path / "experiments"
            metrics_dir.mkdir(parents=True, exist_ok=True)
        else:
            metrics_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints")) / "metrics"
        metrics_path = metrics_dir / f"{experiment_id}.jsonl"
        metrics_stream = MetricsStream(metrics_path, metadata={
            "experiment_id": experiment_id,
            "model": model,
            "dataset": str(dataset),
        })

    progress_reporter = resolve_reporter(unit="batch")

    trainer = Trainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        dataset_dir=dataset,
        resume_from=resume,
        device=train_cfg.get("device", "cuda"),
        validation_enabled=val_enabled,
        validation_split=val_split,
        metrics_stream=metrics_stream,
        metrics_frequency=metrics_frequency,
        progress_reporter=progress_reporter,
        callbacks=resolve_callbacks(),
        cancel_check=resolve_cancel_check(),
    )
    trainer.train()
