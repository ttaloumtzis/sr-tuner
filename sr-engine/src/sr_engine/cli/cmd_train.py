import sys
import time
from pathlib import Path
import click
import yaml

from sr_engine.engine.trainer import Trainer
from sr_engine.engine.metrics_stream import MetricsStream
from sr_engine.utils.config import load_config, merge_overrides, DefaultConfigs, validate_config
from sr_engine.workspace import Workspace


@click.group()
def train() -> None:
    """Model training commands."""


@train.command()
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path), help="Training config YAML.")
@click.option("--model", "-m", default="rrdb_esrgan", help="Model name.")
@click.option("--dataset", "-d", required=True, type=click.Path(path_type=Path))
@click.option("--resume", "-r", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--device", default="cuda", help="Training device.")
@click.option("--batch-size", type=int, default=None)
@click.option("--learning-rate", type=float, default=None)
@click.option("--max-epochs", type=int, default=None, help="Total number of epochs to train.")
@click.option("--num-workers", type=int, default=None, help="Dataloader workers.")
@click.option("--patch-size", type=int, default=None, help="Training patch size.")
@click.option("--save-per-epoch", type=int, default=None, help="Save checkpoint every N epochs.")
@click.option("--validation-enabled/--no-validation-enabled", default=None, help="Enable/disable validation split.")
@click.option("--validation-split", type=click.FloatRange(0.0, 1.0), default=None, help="Fraction of data for validation.")
@click.option("--machine", is_flag=True, default=False, help="Enable machine-readable metrics output.")
@click.option("--experiment-id", type=str, default=None, help="Experiment identifier (auto-generated if omitted).")
@click.option("--metrics-frequency", type=int, default=1, help="Log metrics every N batches.")
@click.option("--dump-config", is_flag=True, default=False, help="Print final merged config and exit.")
@click.option("--project", type=str, default=None, help="Project name (requires workspace).")
@click.pass_context
def run(ctx, config, model, dataset, resume, device, batch_size, learning_rate, max_epochs,
        num_workers, patch_size, save_per_epoch,
        validation_enabled, validation_split, machine, experiment_id, metrics_frequency,
        dump_config, project):
    """Train a super-resolution model."""

    ws: Workspace | None = ctx.obj.get("workspace") if ctx.obj else Workspace.discover()

    if project and not ws:
        raise click.ClickException(
            "--project requires a workspace. Initialize one with 'workspace init' "
            "or set --workspace explicitly."
        )

    if project:
        ws.get_project(project)

    if ws and project:
        dataset = ws.resolve_dataset(dataset)

    cfg_loader = DefaultConfigs()

    model_cfg = cfg_loader.models.get(model)
    if not model_cfg:
        raise click.ClickException(f"Model '{model}' not found. Available: {list(cfg_loader.models.keys())}")

    train_cfg = load_config(config) if config else cfg_loader.train

    overrides = {
        k: v for k, v in {
            "device": device,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "num_workers": num_workers,
            "patch_size": patch_size,
            "save_per_epoch": save_per_epoch,
        }.items() if v is not None
    }
    if validation_enabled is not None:
        overrides.setdefault("validation", {})["enabled"] = validation_enabled
    if validation_split is not None:
        overrides.setdefault("validation", {})["split"] = validation_split

    if overrides:
        train_cfg = merge_overrides(train_cfg, overrides)

    validate_config(train_cfg, required_keys=["max_epochs", "batch_size"])

    if dump_config:
        yaml.safe_dump(train_cfg, sys.stdout, default_flow_style=False, sort_keys=False)
        return

    val_cfg = train_cfg.get("validation", {})
    val_enabled = bool(val_cfg.get("enabled", True))
    val_split = float(val_cfg.get("split", 0.1))

    metrics_stream = None
    if machine:
        if experiment_id is None:
            experiment_id = f"exp_{int(time.time())}"
        if ws and project:
            metrics_dir = ws.path / "projects" / project / "metrics"
        else:
            metrics_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints")) / "metrics"
        metrics_path = metrics_dir / f"{experiment_id}.jsonl"
        metrics_stream = MetricsStream(metrics_path, metadata={
            "experiment_id": experiment_id,
            "model": model,
            "dataset": str(dataset),
        })

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
    )
    trainer.train()
