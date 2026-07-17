"""Dataset health check — evaluates spatial distribution and prunes black pairs."""

import json
from collections import Counter
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from sr_engine.utils.progress import ProgressReporter

from sr_engine.utils.logging import get_logger

log = get_logger(__name__)


def _extract_color_data(img: np.ndarray, channels_summary: Counter) -> np.ndarray:
    """Helper to extract relevant color layers and track bit-depth channel counts."""
    if len(img.shape) == 2:
        channels_summary["Grayscale (1 channel)"] += 1
        return img

    num_channels = img.shape[2]
    if num_channels == 3:
        channels_summary["RGB (3 channels)"] += 1
        return img
    elif num_channels == 4:
        channels_summary["RGBA (4 channels)"] += 1
        return img[:, :, :3]
    else:
        channels_summary[f"Unknown ({num_channels} channels)"] += 1
        return img


# --- Adaptive threshold constants ---
DARK_PERCENTILE: float = 0.15
"""Percentile of darkest frames to analyse for gap detection.
0.15 = bottom 15% — empirically covers the noise floor without
pulling in legitimate low-light content."""

GAP_THRESHOLD: float = 1.5
"""Minimum brightness gap (on 0-255 scale) to classify a transition
as a "noise floor vs. real content" boundary. Values above 1.5
indicate a genuine intensity discontinuity rather than random noise."""

MAX_THRESHOLD: float = 25.0
"""Upper clamp for the computed adaptive threshold. Prevents the threshold
from exceeding 25.0 even if the detected gap is very large, avoiding
false positives on legitimately dark-but-valid content."""

FULL_RANGE_FALLBACK: float = 3.5
"""Fallback threshold when no clear gap is detected and the data
distribution suggests a full-range (0-255) encoding. 3.5 is tight
enough to catch only truly black/near-black frames."""

LIMITED_RANGE_FALLBACK: float = 18.5
"""Fallback threshold when no clear gap is detected and the data
distribution suggests a limited-range (16-235) encoding. 18.5 corresponds
to the BT.709 black level (~16) plus a small margin."""


def _compute_adaptive_threshold(image_means: list[float]) -> float:
    """Calculate a data-driven brightness threshold for black-frame detection.

    Analyses the distribution of mean pixel intensities in the darkest
    ``DARK_PERCENTILE`` of frames. If a sharp brightness gap is found,
    uses the midpoint as threshold; otherwise selects a conservative
    fallback based on the observed dynamic range.

    Args:
        image_means: List of mean pixel intensities per image.

    Returns:
        A float threshold value. Images with mean below this threshold
        are considered black frames.
    """
    if not image_means:
        return 3.0  # Safe minimum fallback floor

    sorted_means = np.sort(image_means)

    # Look for sharp brightness jumps within the lowest DARK_PERCENTILE of the dataset
    lower_bound_count = max(1, int(len(sorted_means) * DARK_PERCENTILE))
    dark_subset = sorted_means[:lower_bound_count]
    percentile_15_score = sorted_means[lower_bound_count - 1]

    log.info("Absolute darkest frame mean: %.2f", sorted_means[0])
    log.info("Dataset %dth percentile mean: %.2f", int(DARK_PERCENTILE * 100), percentile_15_score)

    if len(dark_subset) > 1:
        gaps = np.diff(dark_subset)
        max_gap_idx = np.argmax(gaps)

        # If a true jump between dark noise and real footage exists, split the difference
        if gaps[max_gap_idx] > GAP_THRESHOLD:
            computed_threshold = float(dark_subset[max_gap_idx] + (gaps[max_gap_idx] / 2.0))
            clamped_threshold = min(computed_threshold, MAX_THRESHOLD)

            log.info(
                "Detected brightness gap: %.2f (between %.2f and %.2f)",
                gaps[max_gap_idx], dark_subset[max_gap_idx], dark_subset[max_gap_idx + 1])
            log.info("Calculated optimal threshold: %.2f", clamped_threshold)
            return clamped_threshold

    # Smart Fallback logic based on dynamic ranges
    log.info("No significant brightness gap detected in dark frames.")
    if percentile_15_score < 10.0:
        fallback_threshold = FULL_RANGE_FALLBACK
        log.info("Data leans Full Range (0-255). Applying tight fallback: %.2f", fallback_threshold)
    else:
        fallback_threshold = LIMITED_RANGE_FALLBACK
        log.info("Data leans Limited Range (16-235). Applying standard fallback: %.2f", fallback_threshold)
    return fallback_threshold


def check_dataset_health(dataset_dir: Path,
                         reporter: Optional[ProgressReporter] = None,
                         ) -> dict:
    """Analyze dataset spatial properties, color channels, and detect black frames.

    Examines all images in the ``HR/`` subdirectory, collecting resolution
    and aspect-ratio distributions, channel counts, and mean pixel
    brightness. Uses an adaptive thresholding algorithm to identify
    completely black or near-black frames.

    Args:
        dataset_dir: Path to the dataset directory containing an ``HR/`` folder.
        reporter: Optional progress reporter.

    Returns:
        A dict with keys:
        ``total_images``, ``resolutions``, ``aspect_ratios``, ``channels``,
        ``computed_threshold``, ``black_frames``. On error, returns
        ``{"error": <message>}``.
    """
    hr_dir = dataset_dir / "HR"
    if not hr_dir.is_dir():
        return {"error": "HR directory not found. Run validation/build first."}

    hr_files = list(hr_dir.glob("*.png"))
    total_files = len(hr_files)

    if total_files == 0:
        return {"error": "No images found in HR directory to analyze."}

    resolutions = Counter()
    aspect_ratios = Counter()
    channels_summary = Counter()

    image_means = []
    file_metadata = []

    reporter = reporter or ProgressReporter()
    reporter.start(total=len(hr_files), desc="Analyzing Dataset Metrics")

    # 1. Gather file dimensions and color mean footprints
    for path in hr_files:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        h, w = img.shape[:2]
        resolutions[f"{w}x{h}"] += 1
        aspect_ratios[round(w / h, 2)] += 1

        color_data = _extract_color_data(img, channels_summary)
        img_mean = float(np.mean(color_data))

        image_means.append(img_mean)
        file_metadata.append((path.name, img_mean))
        reporter.update(1)

    reporter.finish()

    # 2. Compute the ideal dynamic threshold slice for this exact asset pool
    threshold = _compute_adaptive_threshold(image_means)

    # 3. Separate near-black dead frames using the custom threshold profile
    black_filenames = [
        filename for filename, img_mean in file_metadata
        if img_mean < threshold
    ]

    return {
        "total_images": total_files,
        "resolutions": dict(resolutions),
        "aspect_ratios": dict(aspect_ratios),
        "channels": dict(channels_summary),
        "computed_threshold": round(threshold, 2),
        "black_frames": black_filenames
    }


def prune_black_frames(dataset_dir: Path, black_filenames: list[str],
                       reporter: Optional[ProgressReporter] = None) -> None:
    """Delete black frame pairs from disk and update the dataset manifest.

    Removes the corresponding HR and LR image files, then filters the
    entries out of ``manifest.json`` so it remains consistent with the
    filesystem state.

    Args:
        dataset_dir: Path to the dataset directory.
        black_filenames: List of filenames (not full paths) to remove.
        reporter: Optional progress reporter.

    Raises:
        RuntimeError: If any files could not be deleted.
    """
    hr_dir = dataset_dir / "HR"
    lr_dir = dataset_dir / "LR"
    manifest_path = dataset_dir / "manifest.json"

    black_set = set(black_filenames)

    reporter = reporter or ProgressReporter()
    reporter.start(total=len(black_filenames), desc="Pruning Black Frames")

    # Delete physical disk assets
    failed = []
    for filename in black_filenames:
        hr_path = hr_dir / filename
        lr_path = lr_dir / filename

        try:
            if hr_path.is_file():
                hr_path.unlink()
        except OSError as e:
            failed.append((str(hr_path), e))

        try:
            if lr_path.is_file():
                lr_path.unlink()
        except OSError as e:
            failed.append((str(lr_path), e))

        reporter.update(1)

    reporter.finish()

    if failed:
        msg = "; ".join(f"{p}: {e}" for p, e in failed)
        raise RuntimeError(f"Failed to delete some black frame files: {msg}")

    # Sync and rewrite structural tracks inside manifest file
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            manifest_data["pairs"] = [
                p for p in manifest_data.get("pairs", [])
                if Path(p["hr"]).name not in black_set and Path(p["lr"]).name not in black_set
            ]

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=4)

        except (json.JSONDecodeError, OSError) as e:
            log.warning("Could not sync manifest.json adjustments: %s", e)