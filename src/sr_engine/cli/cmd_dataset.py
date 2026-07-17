"""CLI commands for dataset operations."""

from pathlib import Path
import click
import yaml

from sr_engine.data.dataset_builder import build_from_video, build_from_preprocessed
from sr_engine.data.dataset_validator import validate
from sr_engine.data.dataset_merge import merge_datasets
from sr_engine.utils.config import load_config, merge_overrides
from sr_engine.data.dataset_health import check_dataset_health, prune_black_frames
from .helpers import make_workspace_config_loader, no_workspace_config_option, resolve_reporter

import sys


_DEGRADATION_SECTIONS = {
    "blur": "blur",
    "noise": "noise",
    "jpeg": "jpeg",
    "jpeg2000": "jpeg2000",
    "color-jitter": "color_jitter",
}
"""Mapping from CLI hyphen names to config section names for ``--degradations``."""


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
@click.option("--degradations", "-d", default=None,
              help="Comma-separated enabled degradations: blur,noise,jpeg,jpeg2000,color-jitter. "
                   "Omit to use per-section 'enabled' fields from config.")
@click.option("--resize-method", default=None,
              type=click.Choice(["area", "bicubic", "bilinear", "lanczos", "nearest"]),
              help="Downsampling interpolation method (default: area). Overrides config.")
@no_workspace_config_option
@click.option("--dump-config", is_flag=True, default=False, help="Print final merged config and exit.")
@click.pass_context
def build(ctx, input: Path, config: Path | None, out: Path | None,
          degradations: str | None, resize_method: str | None,
          no_workspace_config: bool, dump_config: bool) -> None:
    """Build a dataset from a video file or validate a preprocessed directory."""
    ws, cfg_loader = make_workspace_config_loader(ctx, no_workspace_config)

    if config is not None:
        custom_cfg = load_config(config)
        cfg = merge_overrides(cfg_loader.get_dataset_config(), custom_cfg)
    else:
        cfg = cfg_loader.get_dataset_config()

    if degradations is not None:
        enabled = set(d.strip() for d in degradations.split(","))
        deg_cfg = cfg.setdefault("degradation", {})
        for cli_name, cfg_key in _DEGRADATION_SECTIONS.items():
            if cfg_key in deg_cfg:
                deg_cfg[cfg_key]["enabled"] = cli_name in enabled

    if resize_method is not None:
        deg_cfg = cfg.setdefault("degradation", {})
        resize_cfg = deg_cfg.setdefault("resize", {})
        resize_cfg["method"] = resize_method

    if dump_config:
        yaml.safe_dump(cfg, sys.stdout, default_flow_style=False, sort_keys=False)
        return

    if ws and out is None and not input.is_dir():
        if input.is_file():
            out = ws.path / "datasets" / input.stem

    if input.is_dir():
        click.echo(f"Processing and validating preprocessed directory: {input}")
        result = build_from_preprocessed(input, cfg, reporter=resolve_reporter(unit="pair"))
    else:
        if out is None:
            raise click.BadParameter("The '--out / -o' option is required for video files.")
        out.parent.mkdir(parents=True, exist_ok=True)
        click.echo(f"Extracting and degrading video: {input} -> {out}")
        result = build_from_video(input, out, cfg, reporter=resolve_reporter(unit="pair"))

    click.secho(f"Dataset ready at: {result}", fg="green", bold=True)


@dataset.command(name="validate")
@click.option(
    "--path", "-p", required=True, type=click.Path(exists=True, path_type=Path),
    help="Dataset directory to validate (must contain HR/ and LR/)."
)
def validate_cmd(path: Path) -> None:
    """Validate that an existing dataset directory is well-formed."""
    click.echo(f"Running deep validation scan on: {path}...")

    report = validate(path, reporter=resolve_reporter(unit="pair"))

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
    report = check_dataset_health(path, reporter=resolve_reporter(unit="img"))

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
            prune_black_frames(path, report["black_frames"], reporter=resolve_reporter(unit="pair"))
            click.secho("Dataset successfully cleaned!", fg="green", bold=True)
        else:
            click.echo("Skipping cleanup.")
    else:
        click.echo("\nQuality Check: No dead or black frame anomalies discovered.")


@dataset.command()
@click.option("--input", "-i", required=True, type=click.Path(exists=True, path_type=Path),
              help="Directory containing dataset subdirectories (each must have HR/, LR/, and manifest.json).")
@click.option("--out", "-o", required=False, type=click.Path(path_type=Path),
              help="Output directory. Defaults to <input>/merged.")
@click.option("--scale", type=int, default=None,
              help="Only merge datasets with this scale factor.")
@click.option("--name", type=str, default=None,
              help="Custom output subdirectory name (default: scale_{N}). Requires --scale or single scale group.")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Skip deletion confirmation.")
@click.option("--keep-sources", is_flag=True, default=False,
              help="Keep original datasets after merge (don't delete).")
@click.pass_context
def merge(ctx, input: Path, out: Path | None, scale: int | None,
          name: str | None, yes: bool, keep_sources: bool) -> None:
    """Merge all datasets under INPUT into combined datasets grouped by scale."""
    if out is None:
        out = input / "merged"

    click.echo(f"Scanning for datasets in: {input}")
    results = merge_datasets(
        datasets_root=input,
        out_dir=out,
        scale=scale,
        output_name=name,
        reporter=resolve_reporter(unit="dataset"),
    )

    for r in results:
        click.secho(f"✓ Merged dataset (scale {r.scale}) at: {r.output_path}", fg="green", bold=True)

    if not keep_sources and results:
        click.echo("")
        for r in results:
            click.echo(f"Scale {r.scale}: {len(r.source_datasets)} source dataset(s)")
            for src in r.source_datasets:
                click.echo(f"  - {src}")

        delete_ok = yes or click.confirm(
            "Delete these source datasets?",
            default=False,
        )
        if delete_ok:
            import shutil
            for r in results:
                for src in r.source_datasets:
                    click.echo(f"Removing: {src}")
                    shutil.rmtree(src)
            click.secho("Source datasets removed.", fg="green", bold=True)
        else:
            click.echo("Source datasets preserved.")
