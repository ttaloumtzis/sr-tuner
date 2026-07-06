"""CLI commands for dataset operations."""

from pathlib import Path

import click

from sr_engine.data.dataset_builder import build_from_video, build_from_preprocessed
from sr_engine.data.dataset_validator import validate
from sr_engine.utils.config import load_config


@click.group()
def dataset() -> None:
    """Dataset creation and validation commands."""


@dataset.command()
@click.option("--input", "-i", required=True, type=click.Path(exists=True, path_type=Path),
              help="Input video file or preprocessed dataset directory.")
@click.option("--config", "-c", required=True, type=click.Path(exists=True, path_type=Path),
              help="Dataset config YAML (degradation pipeline parameters).")
@click.option("--out", "-o", required=True, type=click.Path(path_type=Path),
              help="Output dataset directory (HR/ + LR/ + manifest.json).")
def build(input: Path, config: Path, out: Path) -> None:
    """Build a dataset from a video file or validate a preprocessed directory.

    If --input is a video file, frames are extracted and LR pairs are
    generated via the degradation pipeline in --config.

    If --input is an existing dataset directory (containing HR/ and LR/),
    it is validated and used directly.
    """
    cfg = load_config(config)
    if input.is_dir():
        result = build_from_preprocessed(input, cfg)
    else:
        result = build_from_video(input, out, cfg)
    click.echo(f"Dataset ready at: {result}")


@dataset.command()
@click.option("--path", "-p", required=True, type=click.Path(exists=True, path_type=Path),
              help="Dataset directory to validate (must contain HR/ and LR/).")
def validate_cmd(path: Path) -> None:
    """Validate that an existing dataset directory is well-formed."""
    report = validate(path)
    if report.ok:
        click.echo(f"Valid dataset: {report.num_pairs} paired images, no issues.")
    else:
        click.echo(f"Validation failed: {report.num_pairs} pairs found, {len(report.problems)} problems:")
        for p in report.problems:
            click.echo(f"  - {p}")
        raise click.Abort()
