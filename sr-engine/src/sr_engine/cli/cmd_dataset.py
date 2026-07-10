"""CLI commands for dataset operations."""

from pathlib import Path
import click
import yaml

from sr_engine.data.dataset_builder import build_from_video, build_from_preprocessed
from sr_engine.data.dataset_validator import validate
from sr_engine.utils.config import load_config, merge_overrides
from sr_engine.data.dataset_health import check_dataset_health, prune_black_frames
from sr_engine.utils.progress import TqdmReporter
from .helpers import make_workspace_config_loader, no_workspace_config_option

import sys


@click.group()
def dataset() -> None:
    """Dataset creation and validation commands."""


@dataset.command()
@click.option("--input", "-i", required=True, type=click.Path(exists=True, path_type=Path),
              help="Input video file or preprocessed dataset directory.")
@click.option("--config", "-c", required=False, default=None, type=click.Path(exists=True, path_type=Path),
              help="Dataset config YAML. Defaults to internal project config.")
@click.option("--out", "-o", required=False, type=click.Path(path_type=Path),
              help="Output dataset directory. Required if input is a video file.")
@no_workspace_config_option
@click.option("--dump-config", is_flag=True, default=False, help="Print final merged config and exit.")
@click.pass_context
def build(ctx, input: Path, config: Path | None, out: Path | None,
          no_workspace_config: bool, dump_config: bool) -> None:
    """Build a dataset from a video file or validate a preprocessed directory."""
    ws, cfg_loader = make_workspace_config_loader(ctx, no_workspace_config)

    if config is not None:
        custom_cfg = load_config(config)
        cfg = merge_overrides(cfg_loader.get_dataset_config(), custom_cfg)
    else:
        cfg = cfg_loader.get_dataset_config()

    if dump_config:
        yaml.safe_dump(cfg, sys.stdout, default_flow_style=False, sort_keys=False)
        return

    if ws and out is None and not input.is_dir():
        if input.is_file():
            out = ws.path / "datasets" / input.stem

    if input.is_dir():
        click.echo(f"Processing and validating preprocessed directory: {input}")
        result = build_from_preprocessed(input, cfg)
    else:
        if out is None:
            raise click.BadParameter("The '--out / -o' option is required for video files.")
        out.parent.mkdir(parents=True, exist_ok=True)
        click.echo(f"Extracting and degrading video: {input} -> {out}")
        result = build_from_video(input, out, cfg)

    click.secho(f"Dataset ready at: {result}", fg="green", bold=True)


@dataset.command(name="validate")
@click.option(
    "--path", "-p", required=True, type=click.Path(exists=True, path_type=Path),
    help="Dataset directory to validate (must contain HR/ and LR/)."
)
def validate_cmd(path: Path) -> None:
    """Validate that an existing dataset directory is well-formed."""
    click.echo(f"Running deep validation scan on: {path}...")

    report = validate(path, reporter=TqdmReporter(unit="pair"))

    if report.ok:
        click.secho(
            f"Valid dataset: {report.num_pairs} paired images, no issues.",
            fg="green",
            bold=True
        )
    else:
        click.secho(
            f"Validation failed: {report.num_pairs} valid pairs found, "
            f"{len(report.problems)} problems identified:",
            fg="red",
            bold=True
        )
        for p in report.problems:
            click.echo(f"  - {p}")
        raise click.Abort()


@dataset.command(name="health")
@click.option(
    "--path", "-p", required=True, type=click.Path(exists=True, path_type=Path),
    help="Dataset root directory containing HR/ folder to analyze."
)
@click.option(
    "--yes", "-y", is_flag=True, default=False,
    help="Automatically proceed and delete identified black frame pairs without asking."
)
def health_cmd(path: Path, yes: bool) -> None:
    """Profile statistical balance and attributes of an existing dataset."""
    click.echo(f"Analyzing dataset health and distribution at: {path}...")
    report = check_dataset_health(path, reporter=TqdmReporter(unit="img"))

    if "error" in report:
        click.secho(f"Error: {report['error']}", fg="red", bold=True)
        raise click.Abort()

    click.echo(f"\n{'='*40}\nDATASET HEALTH PROFILE\n{'='*40}")
    click.echo(f"Total Logged Frames: {report['total_images']}")

    click.echo("\nResolution Breakdown:")
    for res, count in report['resolutions'].items():
        click.echo(f"  - {res}: {count} frames")

    click.echo("\nAspect Ratios:")
    for ratio, count in report['aspect_ratios'].items():
        click.echo(f"  - {ratio}: {count} frames")

    click.echo("\nColor Spaces / Channels:")
    for ch, count in report['channels'].items():
        click.echo(f"  - {ch}: {count} frames")

    black_count = len(report["black_frames"])
    if black_count > 0:
        click.echo(f"\n{'!'*40}")
        click.secho(f"Warning: Found {black_count} completely black frame pairs!", fg="yellow", bold=True)
        click.echo('!'*40)

        proceed = False
        if yes:
            proceed = True
        else:
            proceed = click.confirm(
                "Do you want to permanently delete these black frames and update manifest.json?",
                default=False
            )

        if proceed:
            click.echo(f"Pruning {black_count} pairs from dataset paths and updating layout manifest...")
            prune_black_frames(path, report["black_frames"])
            click.secho("Dataset successfully cleaned!", fg="green", bold=True)
        else:
            click.echo("Skipping cleanup.")
    else:
        click.echo("\nQuality Check: No dead or black frame anomalies discovered.")
