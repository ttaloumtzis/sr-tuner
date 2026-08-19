"""Dataset builder — orchestrates video -> HR/LR folder output."""
import json
import threading
from pathlib import Path
import shutil
from typing import Optional

import cv2

from .dataset_validator import validate
from .degrade import batch_degrade
from .image_files import list_images, lr_scale_from_suffix, pair_hr_lr
from .video_extract import extract_frames

from sr_engine.utils.logging import get_logger
from sr_engine.utils.progress import ProgressReporter

log = get_logger(__name__)

SCALE_TOLERANCE = 0.01
"""Relative tolerance when deciding whether a detected scale is an integer."""


def inspect_dataset(dataset_dir: Path) -> dict:
    """Inspect a pre-extracted HR/LR folder without modifying it.

    Reads image dimensions from the first few matched pairs only, so it is
    fast even on large datasets.

    Args:
        dataset_dir: Path to the dataset root containing ``HR/`` and ``LR/``.

    Returns:
        Dict with keys: ``hr_count``, ``lr_count``, ``pair_count``,
        ``hr_size``, ``lr_size``, ``scale_ratio``, ``scale_exact``,
        ``scale_w``, ``scale_h``, ``has_manifest``, ``warnings``.
    """
    hr_dir = dataset_dir / "HR"
    lr_dir = dataset_dir / "LR"

    hr_count = len(list_images(hr_dir))
    lr_count = len(list_images(lr_dir))
    pairs = pair_hr_lr(hr_dir, lr_dir)

    warnings: list[str] = []
    if not hr_dir.is_dir():
        warnings.append("Missing 'HR/' subdirectory.")
    if not lr_dir.is_dir():
        warnings.append("Missing 'LR/' subdirectory.")
    if hr_count > lr_count:
        warnings.append(
            f"HR has {hr_count} image(s) but LR only has {lr_count} — "
            f"{hr_count - lr_count} HR file(s) have no LR match."
        )
    elif lr_count > hr_count:
        warnings.append(
            f"LR has {lr_count} image(s) but HR only has {hr_count} — "
            f"{lr_count - hr_count} LR file(s) have no HR match."
        )
    if pairs and len(pairs) != hr_count:
        warnings.append(
            f"{hr_count - len(pairs)} HR file(s) could not be matched to an LR file."
        )

    hr_size = None
    lr_size = None
    scale_w = None
    scale_h = None
    scale_ratio = None
    scale_exact = False

    for hr_path, lr_path in pairs[:5]:
        hr_img = cv2.imread(str(hr_path))
        lr_img = cv2.imread(str(lr_path))
        if hr_img is None or lr_img is None:
            continue
        hr_h, hr_w = hr_img.shape[:2]
        lr_h, lr_w = lr_img.shape[:2]
        if hr_h == 0 or hr_w == 0 or lr_h == 0 or lr_w == 0:
            continue

        hr_size = {"width": hr_w, "height": hr_h}
        lr_size = {"width": lr_w, "height": lr_h}
        if lr_w > 0 and lr_h > 0:
            scale_w = hr_w / lr_w
            scale_h = hr_h / lr_h
        if scale_w is not None and scale_h is not None:
            scale_ratio = round((scale_w + scale_h) / 2, 4)
            rounded = round(scale_ratio)
            scale_exact = rounded > 0 and abs(scale_ratio - rounded) <= SCALE_TOLERANCE * rounded
        break

    if not pairs:
        if hr_dir.is_dir() and lr_dir.is_dir():
            warnings.append("No matching HR/LR image pairs found.")
    elif hr_size is None:
        # Images could not be read — fall back to the scale encoded in
        # LR filename suffixes (e.g. 000001x2.png -> 2), if consistent.
        suffix_scales = {lr_scale_from_suffix(lr) for _, lr in pairs}
        if len(suffix_scales) == 1 and None not in suffix_scales:
            scale_ratio = float(suffix_scales.pop())
            rounded = round(scale_ratio)
            scale_exact = rounded > 0 and abs(scale_ratio - rounded) <= SCALE_TOLERANCE * rounded
        else:
            warnings.append(
                "Could not read image dimensions from the first pairs — "
                "images may be corrupt or in an unsupported format."
            )
    elif scale_w is not None and scale_h is not None and abs(scale_w - scale_h) > max(0.05, 0.01 * scale_w):
        warnings.append(
            f"Non-uniform scale detected: width ratio ×{scale_w:.2f} vs "
            f"height ratio ×{scale_h:.2f}."
        )

    return {
        "hr_count": hr_count,
        "lr_count": lr_count,
        "pair_count": len(pairs),
        "hr_size": hr_size,
        "lr_size": lr_size,
        "scale_ratio": scale_ratio,
        "scale_exact": scale_exact,
        "scale_w": scale_w,
        "scale_h": scale_h,
        "has_manifest": (dataset_dir / "manifest.json").is_file(),
        "warnings": warnings,
    }


def _validate_scale(scale: float) -> int:
    """Coerce *scale* to a near-integer and return it, or raise ValueError."""
    rounded = round(scale)
    if rounded <= 0:
        raise ValueError(f"Scale must be a positive integer, got {scale}")
    if abs(scale - rounded) > SCALE_TOLERANCE * rounded:
        raise ValueError(
            f"Scale {scale} is not a near-integer. Refusing to write a "
            f"manifest that would fail validation."
        )
    return rounded


def _write_manifest(dataset_dir: Path, pairs: list[dict], scale: int) -> Path:
    """Write ``manifest.json`` in the canonical dataset_builder format."""
    config_block = {
        "scale": scale,
        "frame_rate": None,
        "video_source": "preprocessed_folder",
    }
    manifest_data = {
        "config": config_block,
        "pairs": pairs,
    }
    manifest_path = dataset_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4, ensure_ascii=False)
    return manifest_path


def build_manifest(dataset_dir: Path, scale: float) -> dict:
    """Build ``manifest.json`` for a pre-extracted HR/LR dataset.

    Pairs are matched by filename (see ``image_files.pair_hr_lr``). The
    manifest is written in the exact same shape as
    ``build_from_preprocessed`` so the validator, loader and merge all agree.

    Args:
        dataset_dir: Dataset root containing ``HR/`` and ``LR/``.
        scale: Super-resolution scale; must be a near-integer.

    Returns:
        Dict with keys ``path``, ``scale``, ``num_pairs``.

    Raises:
        FileNotFoundError: If ``HR/`` or ``LR/`` is missing.
        ValueError: If no pairs are found or *scale* is not near-integer.
    """
    hr_dir = dataset_dir / "HR"
    lr_dir = dataset_dir / "LR"

    if not hr_dir.is_dir() or not lr_dir.is_dir():
        raise FileNotFoundError(
            f"Preprocessed source structure missing in '{dataset_dir}'. "
            f"Ensure both 'HR/' and 'LR/' subdirectories exist explicitly."
        )

    scale_int = _validate_scale(scale)
    hr_lr_pairs = pair_hr_lr(hr_dir, lr_dir)

    if not hr_lr_pairs:
        raise ValueError(f"No matching HR/LR image pairs found in: {dataset_dir}")

    pairs = [
        {
            "hr": str(hr.relative_to(dataset_dir)),
            "lr": str(lr.relative_to(dataset_dir)),
        }
        for hr, lr in hr_lr_pairs
    ]

    _write_manifest(dataset_dir, pairs, scale_int)
    return {"path": str(dataset_dir), "scale": scale_int, "num_pairs": len(pairs)}


def build_from_video(
        video_path: Path,
        out_dir: Path,
        config: dict,
        reporter: Optional[ProgressReporter] = None,
        cancel_event: Optional[threading.Event] = None,
) -> Path:
    """Build a dataset from a video file.

    Steps:
        1. Extract frames from the video to the HR subfolder.
        2. Apply the degradation pipeline to generate LR pairs.
        3. Write a manifest.json and verify the integrity of the dataset.

    Returns the path to *out_dir*.
    """
    out_dir_hr = out_dir / "HR"
    out_dir_lr = out_dir / "LR"

    try:
        # 1. Extract the video frames to HR subfolder
        hr_paths = extract_frames(
            video_path=video_path,
            out_dir=out_dir_hr,
            frame_rate=config.get("frame_rate"),
            start_time=config.get("start_time", 0.0),
            duration=config.get("duration"),
            reporter=reporter,
            cancel_event=cancel_event,
        )

        # Fast fallback if no frames were extracted
        if not hr_paths:
            raise ValueError(f"No frames were extracted from video: {video_path}")

        # 2. Degradation pipeline to generate the LR pairs.
        # NOTE: batch_degrade returns (hr, lr) pairs matched by identity, not two
        # separately-sorted lists — do NOT zip(hr_paths, lr_paths) here. If any
        # frame fails to decode, a positional zip would silently misalign every
        # pair after the dropped one.
        hr_lr_pairs = batch_degrade(
            hr_paths=hr_paths,
            lr_dir=out_dir_lr,
            scale=config.get("scale", 4),
            config=config,
            reporter=reporter,
        )

        if len(hr_lr_pairs) < len(hr_paths):
            skipped = len(hr_paths) - len(hr_lr_pairs)
            loss_ratio = skipped / len(hr_paths)
            raise RuntimeError(
                f"{skipped}/{len(hr_paths)} frames ({loss_ratio:.0%}) failed to decode during "
                f"degradation. The video likely uses a pixel format or codec that OpenCV "
                f"cannot decode. Try re-encoding: "
                f"ffmpeg -i {video_path} -c:v libx264 -pix_fmt yuv420p -crf 18 output.mp4"
            )

        # Build the temporary manifest metadata block first so the validator
        # can read the configured scale factor dynamically.
        manifest_data = {
            "config": {
                "scale": config.get("scale", 4),
                "frame_rate": config.get("frame_rate"),
                "video_source": str(video_path.name),
            },
            "pairs": [
                {
                    "hr": str(hr.relative_to(out_dir)),
                    "lr": str(lr.relative_to(out_dir)),
                }
                for hr, lr in hr_lr_pairs
            ],
        }

        # Write the manifest file BEFORE validating so the validator can audit it
        manifest_path = out_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=4, ensure_ascii=False)

        # 3. Comprehensive Deep Integrity Verification
        report = validate(out_dir, reporter=reporter)
        if not report.ok:
            # If verification fails, clean up the bad manifest to keep things unstable/invalidated
            if manifest_path.exists():
                manifest_path.unlink()

            # Structure problems list into a clean message
            error_msg = "\n- ".join(report.problems)
            raise RuntimeError(
                f"Dataset validation failed for '{out_dir}'! Found the following problems:\n- {error_msg}"
            )

    except Exception:
        # On any failure, clean up the partial output directory so retries
        # don't encounter stale files from the aborted build.
        if out_dir.exists():
            shutil.rmtree(out_dir)
        raise

    log.info("Successfully verified and built a stable dataset with %d pairs at: %s", report.num_pairs, out_dir)
    return out_dir



def build_from_preprocessed(
        dataset_dir: Path,
        config: dict,
        reporter: Optional[ProgressReporter] = None,
) -> Path:
    """Validate and finalize a dataset that is already in HR/LR folder format.

    Returns the path to *dataset_dir* after validation.
    """
    hr_dir = dataset_dir / "HR"
    lr_dir = dataset_dir / "LR"
    manifest_path = dataset_dir / "manifest.json"

    # 1. Base Structural Integrity Check
    if not hr_dir.is_dir() or not lr_dir.is_dir():
        raise FileNotFoundError(
            f"Preprocessed source structure missing in '{dataset_dir}'. "
            f"Ensure both 'HR/' and 'LR/' subdirectories exist explicitly."
        )

    # 2. Gather matching file paths to build the manifest map
    hr_paths = list_images(hr_dir)

    if not hr_paths:
        raise ValueError(f"No source images found inside the HR directory: {hr_dir}")

    # 3. Write a tentative manifest.json file so the validator has its audit target
    hr_lr_pairs = pair_hr_lr(hr_dir, lr_dir)
    scale = _validate_scale(float(config.get("scale", 4)))
    pairs = [
        {
            "hr": str(hr.relative_to(dataset_dir)),
            "lr": str(lr.relative_to(dataset_dir)),
        }
        for hr, lr in hr_lr_pairs
    ]
    manifest_data = {
        "config": {
            "scale": scale,
            "frame_rate": config.get("frame_rate"),
            "video_source": "preprocessed_folder",
        },
        "pairs": pairs,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4, ensure_ascii=False)

    # 4. Trigger Deep Validation Scan
    report = validate(dataset_dir, reporter=reporter)
    if not report.ok:
        # Self-cleaning: purge the generated manifest if structural rules are broken
        if manifest_path.exists():
            manifest_path.unlink()

        error_msg = "\n- ".join(report.problems)
        raise RuntimeError(
            f"Preprocessed folder validation failed for '{dataset_dir}'!\n- {error_msg}"
        )

    log.info("Successfully finalized preprocessed dataset with %d verified pairs at: %s", report.num_pairs, dataset_dir)
    return dataset_dir