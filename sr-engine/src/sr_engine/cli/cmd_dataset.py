"""CLI commands for dataset operations."""

from pathlib import Path
import click
from sr_engine.data.dataset_builder import build_from_video, build_from_preprocessed
from sr_engine.data.dataset_validator import validate
from sr_engine.utils.config import load_config
from pathlib import Path
from sr_engine.data.dataset_health import check_dataset_health, prune_black_frames

@click.group()
def dataset() -> None:
    """Dataset creation and validation commands."""


@dataset.command()
@click.option("--input", "-i", required=True, type=click.Path(exists=True, path_type=Path),
              help="Input video file or preprocessed dataset directory.")
# CHANGED: required=False, and added an informative help string
@click.option("--config", "-c", required=False, default=None, type=click.Path(exists=True, path_type=Path),
              help="Dataset config YAML. If omitted, defaults to sr_engine/configs/datasets/video_pairs.yml")
@click.option("--out", "-o", required=False, type=click.Path(path_type=Path),
              help="Output dataset directory. Required if input is a video file.")
def build(input: Path, config: Path | None, out: Path | None) -> None:
    """Build a dataset from a video file or validate a preprocessed directory.

    If --config is not specified, the system automatically falls back to the
    default configuration file located inside the package installation directory.
    """
    # 1. Fallback to package default config if none is provided
    if config is None:
        # __file__ is /.../src/sr_engine/cli/cmd_dataset.py
        cli_dir = Path(__file__).resolve().parent  # /.../src/sr_engine/cli
        package_root = cli_dir.parent  # /.../src/sr_engine

        # FIXED: Corrected extension from .yml to .yaml to match your file tree exactly
        config = package_root / "configs" / "datasets" / "video_pairs.yaml"

        if not config.is_file():
            raise FileNotFoundError(
                f"Expected internal default configuration template at '{config}', "
                f"but it could not be located. Please check the asset layout."
            )

        click.echo(f"No configuration provided. Using package defaults: {config.name}")

    # 2. Load the resolved config path
    cfg = load_config(config)

    if input.is_dir():
        click.echo(f"Processing and validating preprocessed directory: {input}")
        result = build_from_preprocessed(input, cfg)
    else:
        if out is None:
            raise click.BadParameter("The '--out / -o' option is required when the input is a video file.")
        click.echo(f"Extracting and degrading video: {input} -> {out}")
        result = build_from_video(input, out, cfg)

    click.secho(f"Dataset ready at: {result}", fg="green", bold=True)


# FIXED: Added name="validate" explicitly here so it isn't called "validate-cmd"
@dataset.command(name="validate")
@click.option(
    "--path", "-p", required=True, type=click.Path(exists=True, path_type=Path),
    help="Dataset directory to validate (must contain HR/ and LR/)."
)
def validate_cmd(path: Path) -> None:
    """Validate that an existing dataset directory is well-formed."""
    click.echo(f"Running deep validation scan on: {path}...")

    report = validate(path)

    if report.ok:
        click.secho(
            f"✔ Valid dataset: {report.num_pairs} paired images, no issues.",
            fg="green",
            bold=True
        )
    else:
        click.secho(
            f"❌ Validation failed: {report.num_pairs} valid pairs found, "
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
# NEW FLAG: Automatically answers 'yes' to deletion prompt if flag is provided
@click.option(
    "--yes", "-y", is_flag=True, default=False,
    help="Automatically proceed and delete identified black frame pairs without asking."
)
def health_cmd(path: Path, yes: bool) -> None:
    """Profile statistical balance and attributes of an existing dataset (Data Quality check)."""
    click.echo(f"Analyzing dataset health and distribution at: {path}...")

    report = check_dataset_health(path)

    if "error" in report:
        click.secho(f"❌ {report['error']}", fg="red", bold=True)
        raise click.Abort()

    # Print out standard statistics
    click.echo("\n" + "=" * 40 + "\n📊 DATASET HEALTH PROFILE\n" + "=" * 40)
    click.echo(f"Total Logged Frames: {report['total_images']}")

    click.echo("\n🔹 Resolution Breakdown:")
    for res, count in report['resolutions'].items():
        click.echo(f"  - {res}: {count} frames")

    click.echo("\n🔹 Aspect Ratios:")
    for ratio, count in report['aspect_ratios'].items():
        click.echo(f"  - {ratio}: {count} frames")

    click.echo("\n🔹 Color Spaces / Channels:")
    for ch, count in report['channels'].items():
        click.echo(f"  - {ch}: {count} frames")

    #3. Handle Black Frame Identification and Actions
    black_count = len(report["black_frames"])
    if black_count > 0:
        click.echo("\n" + "!" * 40)
        click.secho(f"⚠️ Warning: Found {black_count} completely black frame pairs!", fg="yellow", bold=True)
        click.echo("!" * 40)

        # Check if we should execute deletion right away or ask via prompt
        proceed_to_delete = False
        if yes:
            proceed_to_delete = True
        else:
            proceed_to_delete = click.confirm(
                click.style("Do you want to permanently delete these black frames and update manifest.json?",
                            fg="cyan", bold=True),
                default=False
            )

        if proceed_to_delete:
            click.echo(f"Pruning {black_count} pairs from dataset paths and updating layout manifest...")
            prune_black_frames(path, report["black_frames"])
            click.secho("✔ Dataset successfully cleaned!", fg="green", bold=True)
        else:
            click.echo("Skipping cleanup. Black frames left untouched in dataset structure.")
    else:
        click.echo("\n✔ Quality Check: No dead or black frame anomalies discovered.")