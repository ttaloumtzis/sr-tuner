"""Dataset health check — evaluates spatial distribution and prunes black pairs."""

import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


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


def _compute_adaptive_threshold(image_means: list[float]) -> float:
    """Helper that calculates a tailored black-cutoff point based on image distribution gaps."""
    if not image_means:
        return 3.0  # Safe minimum fallback floor

    sorted_means = np.sort(image_means)

    # Look for sharp brightness jumps within the lowest 15% of the dataset
    lower_bound_count = max(1, int(len(sorted_means) * 0.15))
    dark_subset = sorted_means[:lower_bound_count]
    percentile_15_score = sorted_means[lower_bound_count - 1]

    print("\n" + "-" * 40 + "\n⚙️ THRESHOLD DIAGNOSTICS\n" + "-" * 40)
    print(f"Absolute darkest frame mean: {sorted_means[0]:.2f}")
    print(f"Dataset 15th percentile mean: {percentile_15_score:.2f}")

    if len(dark_subset) > 1:
        gaps = np.diff(dark_subset)
        max_gap_idx = np.argmax(gaps)

        # If a true jump between dark noise and real footage exists, split the difference
        if gaps[max_gap_idx] > 1.5:
            computed_threshold = float(dark_subset[max_gap_idx] + (gaps[max_gap_idx] / 2.0))
            clamped_threshold = min(computed_threshold, 25.0)

            print(
                f"Detected brightness gap: {gaps[max_gap_idx]:.2f} (between {dark_subset[max_gap_idx]:.2f} and {dark_subset[max_gap_idx + 1]:.2f})")
            print(f"Calculated optimal threshold: {clamped_threshold:.2f}")
            print("-" * 40)
            return clamped_threshold

    # Smart Fallback logic based on dynamic ranges
    print("No significant brightness gap detected in dark frames.")
    if percentile_15_score < 10.0:
        fallback_threshold = 3.5
        print(f"Data leans Full Range (0-255). Applying tight fallback: {fallback_threshold:.2f}")
    else:
        fallback_threshold = 18.5
        print(f"Data leans Limited Range (16-235). Applying standard fallback: {fallback_threshold:.2f}")

    print("-" * 40)
    return fallback_threshold


def check_dataset_health(dataset_dir: Path) -> dict:
    """Analyze the dataset's spatial properties, color channels, and locate bad black frames."""
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

    # 1. Gather file dimensions and color mean footprints
    for path in tqdm(hr_files, desc="🩺 Analyzing Dataset Metrics", unit="img"):
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


def prune_black_frames(dataset_dir: Path, black_filenames: list[str]) -> None:
    """Physically remove black frame pairs from disk and filter out their manifest records."""
    hr_dir = dataset_dir / "HR"
    lr_dir = dataset_dir / "LR"
    manifest_path = dataset_dir / "manifest.json"

    black_set = set(black_filenames)

    # Delete physical disk assets
    for filename in black_filenames:
        hr_path = hr_dir / filename
        lr_path = lr_dir / filename

        if hr_path.is_file():
            hr_path.unlink()
        if lr_path.is_file():
            lr_path.unlink()

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

        except Exception as e:
            print(f"[⚠️ Warning] Could not sync manifest.json adjustments: {e}")