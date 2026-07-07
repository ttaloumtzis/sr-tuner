"""CLI commands for training."""

from pathlib import Path
import click

from sr_engine.engine.trainer import Trainer
from sr_engine.utils.config import load_config, merge_overrides, DefaultConfigs

@click.group()
def train() -> None:
    """Model training commands."""

@train.command()
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path), help="Training config YAML.")
@click.option("--model", "-m", default="rrdb_esrgan", help="Model name.")
@click.option("--dataset", "-d", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--resume", "-r", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--device", default="cuda", help="Training device.")
@click.option("--batch-size", type=int, default=None)
@click.option("--learning-rate", type=float, default=None)
@click.option("--max-epochs", type=int, default=None, help="Total number of epochs to train.")
def run(config, model, dataset, resume, device, batch_size, learning_rate, max_epochs):
    """Train a super-resolution model."""

    # 1. Initialize the config loader
    cfg_loader = DefaultConfigs()

    # 2. Get model configuration
    model_cfg = cfg_loader.models.get(model)
    if not model_cfg:
        raise click.ClickException(f"Model '{model}' not found. Available: {list(cfg_loader.models.keys())}")

    # 3. Load training configuration
    train_cfg = load_config(config) if config else cfg_loader.train

    # 4. Handle CLI overrides (Swapped max_steps for max_epochs)
    overrides = {
        k: v for k, v in {
            "device": device,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs
        }.items() if v is not None
    }

    if overrides:
        train_cfg = merge_overrides(train_cfg, overrides)

    # 5. Initialize and run Trainer
    trainer = Trainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        dataset_dir=dataset,
        resume_from=resume,
        device=train_cfg.get("device", "cuda"),
    )
    trainer.train()