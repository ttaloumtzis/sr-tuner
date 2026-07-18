"""Dataset merge — combines multiple datasets into one per scale group."""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sr_engine.utils.logging import get_logger
from sr_engine.utils.progress import ProgressReporter

log = get_logger(__name__)


@dataclass
class MergeResult:
    """Outcome of merging a group of datasets with the same scale."""
    scale: int
    output_path: Path
    source_datasets: list[Path] = field(default_factory=list)


def _is_dataset_dir(d: Path) -> bool:
    """Check whether a directory is a valid dataset directory.

    A valid dataset directory must contain ``HR/``, ``LR/``, and
    ``manifest.json``.

    Args:
        d: Directory to check.

    Returns:
        ``True`` if the directory has the expected structure.
    """
    return (d / "HR").is_dir() and (d / "LR").is_dir() and (d / "manifest.json").is_file()


def _discover_datasets(root: Path, exclude: Optional[Path] = None) -> list[Path]:
    """Find all valid dataset directories under a root directory.

    Scans immediate subdirectories of *root* for dataset structure.
    Directories matching the exclusion path are skipped.

    Args:
        root: Parent directory to scan.
        exclude: Optional path to exclude from results.

    Returns:
        List of valid dataset directory paths.
    """
    datasets: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if exclude is not None and child.resolve() == exclude.resolve():
            continue
        if _is_dataset_dir(child):
            datasets.append(child)
        else:
            log.warning("Skipping '%s': missing HR/, LR/, or manifest.json", child.name)
    return datasets


def _read_scale(dataset_dir: Path) -> int:
    """Read the super-resolution scale factor from a dataset's manifest.

    Args:
        dataset_dir: Path to a dataset directory containing ``manifest.json``.

    Returns:
        The scale factor (e.g. ``4`` for 4× super-resolution).
    """
    manifest_path = dataset_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return int(float(data.get("config", {}).get("scale", 4)))


def _safe_copy(src: Path, dst_dir: Path, prefix: str) -> Path:
    """Copy a file into a directory with a prefix to avoid name collisions.

    If a file with the same prefixed name already exists, appends a
    numeric counter (e.g. ``prefix_file_1.png``) until the path is unique.

    Args:
        src: Source file to copy.
        dst_dir: Destination directory.
        prefix: String prepended to the filename for disambiguation.

    Returns:
        The destination path of the copied file.
    """
    dst = dst_dir / f"{prefix}_{src.name}"
    counter = 1
    while dst.exists():
        stem = src.stem
        suffix = src.suffix
        dst = dst_dir / f"{prefix}_{stem}_{counter}{suffix}"
        counter += 1
    shutil.copy2(src, dst)
    return dst


def merge_datasets(
    datasets_root: Path,
    out_dir: Path,
    scale: Optional[int] = None,
    output_name: Optional[str] = None,
    reporter: Optional[ProgressReporter] = None,
    dataset_dirs: Optional[list[Path]] = None,
) -> list[MergeResult]:
    """Merge datasets under *datasets_root* into combined datasets grouped by scale.

    Each dataset subdirectory must contain ``HR/``, ``LR/``, and ``manifest.json``.
    Datasets are grouped by the ``scale`` value in their manifest. For each scale
    group a new merged dataset is created under ``out_dir/scale_{N}`` with:

    * A minimal ``manifest.json`` containing only ``config.scale`` and ``sources``.
    * All HR/LR image pairs copied with source-directory prefixes.

    Args:
        datasets_root: Parent directory whose immediate subdirectories are datasets.
        out_dir: Base output directory. Per-scale subdirectories are created inside.
        scale: If set, only merge datasets with this exact scale.
        output_name: Custom subdirectory name instead of ``scale_{N}``.
            Only allowed when merging a single scale group.
        reporter: Optional progress reporter.
        dataset_dirs: If set, only merge these specific subdirectories (relative to
            *datasets_root*). When omitted, all valid datasets are discovered.

    Returns:
        A list of ``MergeResult``, one per scale group processed.

    Raises:
        ValueError: If no valid datasets are found, or if *output_name* is given
            with multiple scale groups.
        FileExistsError: If the output directory for a scale group already exists.
    """
    reporter = reporter or ProgressReporter()

    if dataset_dirs is not None:
        datasets = [d for d in dataset_dirs if _is_dataset_dir(d)]
        if not datasets:
            raise ValueError(
                f"None of the specified directories are valid datasets. "
                "Each must contain HR/, LR/, and manifest.json."
            )
    else:
        datasets = _discover_datasets(datasets_root, exclude=out_dir)
    if not datasets:
        raise ValueError(
            f"No valid datasets found in '{datasets_root}'. "
            "Each dataset directory must contain HR/, LR/, and manifest.json."
        )

    groups: dict[int, list[Path]] = {}
    for d in datasets:
        s = _read_scale(d)
        groups.setdefault(s, []).append(d)

    if scale is not None:
        groups = {s: ds for s, ds in groups.items() if s == scale}

    if not groups:
        raise ValueError(
            f"No datasets found with scale={scale} in '{datasets_root}'."
        )

    if output_name is not None and len(groups) > 1:
        raise ValueError(
            f"Custom --name '{output_name}' requires a single scale group, "
            f"but found {len(groups)} groups: {sorted(groups)}. "
            "Use --scale to filter."
        )

    results: list[MergeResult] = []

    for s, source_dirs in sorted(groups.items()):
        dir_name = output_name if output_name is not None else f"scale_{s}"
        target = out_dir / dir_name

        if target.exists():
            raise FileExistsError(
                f"Output directory already exists: {target}. "
                "Remove it first or choose a different --out."
            )

        tmp_dir = target.with_name(f".tmp_{dir_name}")

        try:
            tmp_hr = tmp_dir / "HR"
            tmp_lr = tmp_dir / "LR"
            tmp_hr.mkdir(parents=True, exist_ok=False)
            tmp_lr.mkdir(parents=True, exist_ok=False)

            total_pairs = 0
            reporter.start(total=len(source_dirs), desc=f"Merging scale {s} datasets")

            for src_dir in source_dirs:
                prefix = str(src_dir.relative_to(datasets_root)).replace("/", "--").replace("\\", "--")
                manifest_path = src_dir / "manifest.json"
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)

                for pair in manifest_data.get("pairs", []):
                    hr_rel = pair.get("hr")
                    lr_rel = pair.get("lr")
                    if not hr_rel or not lr_rel:
                        continue

                    hr_src = src_dir / hr_rel
                    lr_src = src_dir / lr_rel

                    if not hr_src.is_file() or not lr_src.is_file():
                        log.warning("Skipping missing pair: %s / %s", hr_rel, lr_rel)
                        continue

                    _safe_copy(hr_src, tmp_hr, prefix)
                    _safe_copy(lr_src, tmp_lr, prefix)
                    total_pairs += 1

                reporter.update(1)

            reporter.finish()

            merged_manifest = {
                "config": {
                    "scale": s,
                    "sources": [str(d.relative_to(datasets_root)) for d in source_dirs],
                },
                "pairs": [],
            }
            (tmp_dir / "manifest.json").write_text(
                json.dumps(merged_manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Validate the merged dataset
            from sr_engine.data.dataset_validator import validate
            report = validate(tmp_dir, reporter=reporter)
            if not report.ok:
                problems = "\n- ".join(report.problems)
                raise RuntimeError(
                    f"Merged dataset (scale {s}) validation failed:\n- {problems}"
                )

            tmp_dir.rename(target)

            results.append(MergeResult(
                scale=s,
                output_path=target,
                source_datasets=source_dirs,
            ))

            log.info(
                "Merged %d dataset(s) (scale %d) → %s (%d pairs)",
                len(source_dirs), s, target, total_pairs,
            )

        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            raise

    return results
