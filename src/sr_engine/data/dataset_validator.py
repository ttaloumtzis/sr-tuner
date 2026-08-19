"""Dataset validator — checks an existing HR/LR folder matches the manifest.json."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from sr_engine.utils.progress import ProgressReporter
from sr_engine.data.image_files import scan_image_paths


@dataclass
class ValidationReport:
    """Result of validating a dataset directory."""
    ok: bool
    num_pairs: int = 0
    problems: list[str] = field(default_factory=list)


def _normalize_key(key: str) -> str:
    """Normalize a manifest/disk key so matching is platform independent.

    Converts to a forward-slash relative path and folds case so that Windows
    backslash keys (``HR\\foo.png``) and case-insensitive filesystems compare
    equal to the posix keys emitted by the dataset builders.
    """
    return os.path.normcase(Path(key).as_posix())


def _image_size(path: Path) -> Optional[tuple[int, int]]:
    """Return ``(width, height)`` from the image header without decoding pixels.

    ``PIL.Image.open`` only parses the file header; pixel data is never
    decoded. Returns ``None`` when the file is not header-parseable.
    """
    max_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None  # header-only read; ignore decompression bombs
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None
    finally:
        Image.MAX_IMAGE_PIXELS = max_pixels


def validate(dataset_dir: Path,
             reporter: Optional[ProgressReporter] = None,
             ) -> ValidationReport:
    """Validate that *dataset_dir* contains a well-formed HR/LR dataset tracking the manifest.

    Checks:
        - HR/ and LR/ subdirectories exist.
        - manifest.json exists and is parseable.
        - Every image pair cataloged in the manifest exists physically on disk.
        - All referenced files are header-parseable images.
        - HR image dimensions are exactly *scale* times LR dimensions.
        - No orphan files exist on disk that are missing from the manifest tracking log.

    Integrity is verified at the file-header level. Deep corruption detection
    (black frames, truncated pixel data) lives in ``dataset_health``.

    Returns a ValidationReport with the result.
    """
    problems: list[str] = []

    hr_dir = dataset_dir / "HR"
    lr_dir = dataset_dir / "LR"
    manifest_path = dataset_dir / "manifest.json"

    # 1. Structural Check: Verify subdirectories exist
    if not hr_dir.is_dir():
        problems.append("Missing 'HR/' subdirectory.")
    if not lr_dir.is_dir():
        problems.append("Missing 'LR/' subdirectory.")
    if not manifest_path.is_file():
        problems.append("Missing 'manifest.json' configuration benchmark.")

    # If structural foundation is missing, stop early to avoid downstream crashes
    if problems:
        return ValidationReport(ok=False, problems=problems)

    # 2. Parse Manifest File Data
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        scale = int(float(manifest_data.get("config", {}).get("scale", 4)))
        manifest_pairs = manifest_data.get("pairs", [])
    except json.JSONDecodeError as e:
        problems.append(f"Failed to parse manifest.json: {e}")
        return ValidationReport(ok=False, problems=problems)

    # 3. Single-pass directory scan — one listing per side, no per-file stat.
    hr_on_disk = {_normalize_key(str(p.relative_to(dataset_dir))) for p in scan_image_paths(hr_dir)}
    lr_on_disk = {_normalize_key(str(p.relative_to(dataset_dir))) for p in scan_image_paths(lr_dir)}

    # 2b. Minimal manifest (empty pairs) — validate via directory scan
    if not manifest_pairs:
        if not hr_on_disk:
            problems.append("HR/ directory contains no images.")
        if not lr_on_disk:
            problems.append("LR/ directory contains no images.")
        if hr_on_disk and lr_on_disk and len(hr_on_disk) != len(lr_on_disk):
            problems.append(
                f"HR/ has {len(hr_on_disk)} file(s) but LR/ has {len(lr_on_disk)}."
            )

        num_pairs = min(len(hr_on_disk), len(lr_on_disk)) if hr_on_disk and lr_on_disk else 0
        is_ok = len(problems) == 0 and num_pairs > 0
        return ValidationReport(ok=is_ok, num_pairs=num_pairs, problems=problems)

    # 4. Integrity and Dimensional Scale Checks via Manifest Records.
    #    Pre-pass: split out malformed entries and dedupe complete pairs so
    #    the progress total and num_pairs count each unique pair once.
    malformed_entries: list[dict] = []
    checkable: list[tuple[str, str, str, str]] = []  # (raw_hr, raw_lr, hr_key, lr_key)
    seen: set[tuple[str, str]] = set()
    for pair in manifest_pairs:
        raw_hr = pair.get("hr") or pair.get("HR")
        raw_lr = pair.get("lr") or pair.get("LR")
        if not raw_hr or not raw_lr:
            malformed_entries.append(pair)
            continue
        hr_key = _normalize_key(raw_hr)
        lr_key = _normalize_key(raw_lr)
        dedup_key = (hr_key, lr_key)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        checkable.append((raw_hr, raw_lr, hr_key, lr_key))

    for pair in malformed_entries:
        problems.append(f"Malformed manifest track entry: missing path mappings in entry: {pair}")

    num_pairs = 0

    reporter = reporter or ProgressReporter()
    reporter.start(total=len(checkable), desc="Checking Manifest Alignment & Integrity")

    for raw_hr, raw_lr, hr_key, lr_key in checkable:
        # A. Check for missing disk files registered in the manifest.
        #    Disk sets are keyed by normalized relative path; fall back to a
        #    direct probe to stay tolerant of absolute / dotted / nested keys
        #    in hand-edited manifests.
        if hr_key not in hr_on_disk:
            hr_abs = dataset_dir / hr_key
            if hr_abs.is_file():
                hr_on_disk.add(hr_key)
            else:
                problems.append(f"Manifest alignment failure: File '{raw_hr}' is logged in manifest.json but missing from disk.")
                continue
        if lr_key not in lr_on_disk:
            lr_abs = dataset_dir / lr_key
            if lr_abs.is_file():
                lr_on_disk.add(lr_key)
            else:
                problems.append(f"Manifest alignment failure: File '{raw_lr}' is logged in manifest.json but missing from disk.")
                continue

        # B. Read image headers to verify parseability and dimensions.
        hr_size = _image_size(dataset_dir / hr_key)
        lr_size = _image_size(dataset_dir / lr_key)

        if hr_size is None:
            problems.append(f"Corrupted Image: HR file '{raw_hr}' is unreadable or malformed.")
            continue
        if lr_size is None:
            problems.append(f"Corrupted Image: LR file '{raw_lr}' is unreadable or malformed.")
            continue

        hr_w, hr_h = hr_size
        lr_w, lr_h = lr_size

        # C. Check scale metrics strictly
        if hr_h != lr_h * scale or hr_w != lr_w * scale:
            problems.append(
                f"Dimension mismatch on '{Path(hr_key).name}': HR dimensions ({hr_w}x{hr_h}) "
                f"are not exactly {scale}x scale multiplier of LR dimensions ({lr_w}x{lr_h})."
            )
            continue

        num_pairs += 1
        reporter.update(1)

    reporter.finish()

    # 5. Check for Orphaned Files (Files on disk that aren't in the manifest).
    #    Basename keys mirror the original tracker semantics and stay correct
    #    for datasets whose manifest entries use absolute or nested paths.
    def _basenames(rel_keys: set[str]) -> set[str]:
        return {os.path.normcase(Path(k).name) for k in rel_keys}

    disk_hr_names = _basenames(hr_on_disk)
    disk_lr_names = _basenames(lr_on_disk)
    manifest_hr_names = {os.path.normcase(Path(p.get("hr") or p.get("HR") or "").name) for p in manifest_pairs}
    manifest_lr_names = {os.path.normcase(Path(p.get("lr") or p.get("LR") or "").name) for p in manifest_pairs}

    for filename in sorted(disk_hr_names - manifest_hr_names):
        problems.append(f"Orphaned asset: '{filename}' exists in HR/ directory but is missing from manifest.json.")
    for filename in sorted(disk_lr_names - manifest_lr_names):
        problems.append(f"Orphaned asset: '{filename}' exists in LR/ directory but is missing from manifest.json.")

    # 6. Final decision matrix evaluation
    is_ok = len(problems) == 0 and num_pairs > 0
    if num_pairs == 0 and len(problems) == 0:
        problems.append("Dataset manifest is completely empty (0 registered frame tracking structures).")
        is_ok = False

    return ValidationReport(ok=is_ok, num_pairs=num_pairs, problems=sorted(problems))
