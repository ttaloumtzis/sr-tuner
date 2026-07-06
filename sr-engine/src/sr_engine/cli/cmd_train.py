"""CLI commands for training."""

from pathlib import Path

import click

from sr_engine.engine.trainer import Trainer
from sr_engine.utils.config import load_config, merge_overrides


@click.group()
def train() -> None:
    """Model training commands."""


@train.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True, path_type=Path),
              help="Training config YAML.")
@click.option("--model", "-m", required=True, type=click.Path(exists=True, path_type=Path),
              help="Model architecture config YAML.")
@click.option("--dataset", "-d", required=True, type=click.Path(exists=True, path_type=Path),
              help="Dataset directory (HR/ + LR/).")
@click.option("--resume", "-r", type=click.Path(exists=True, path_type=Path), default=None,
              help="Checkpoint path to resume from.")
@click.option("--device", default=None, help="Override device (cuda, rocm, cpu).")
@click.option("--batch-size", type=int, default=None, help="Override batch size.")
@click.option("--learning-rate", type=float, default=None, help="Override learning rate.")
@click.option("--max-steps", type=int, default=None, help="Override max training steps.")
def run(
    config: Path,
    model: Path,
    dataset: Path,
    resume: Path | None,
    device: str | None,
    batch_size: int | None,
    learning_rate: float | None,
    max_steps: int | None,
) -> None:
    """Train a super-resolution model.

    Uses the training YAML for hyperparameters and the model YAML for
    architecture config. CLI flags override corresponding config keys.
    """
    train_cfg = load_config(config)
    model_cfg = load_config(model)

    overrides = {}
    if device is not None:
        overrides["device"] = device
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    if learning_rate is not None:
        overrides["learning_rate"] = learning_rate
    if max_steps is not None:
        overrides["max_steps"] = max_steps

    if overrides:
        train_cfg = merge_overrides(train_cfg, overrides)

    trainer = Trainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        dataset_dir=dataset,
        resume_from=resume,
        device=train_cfg.get("device", "cuda"),
    )
    trainer.train()
