"""Dataset health check — evaluates spatial distribution and prunes black pairs."""

import json
from collections import Counter
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from sr_engine.data.image_files import list_images, pair_hr_lr
from sr_engine.utils.progress import ProgressReporter

from sr_engine.utils.logging import get_logger

log = get_logger(__name__)

HEALTH_REPORT_FILENAME = ".health_report.json"

MAX_UNREADABLE_FILES_IN_MESSAGE = 20


class DatasetIntegrityError(Exception):
    """Raised when a dataset contains unreadable or missing image files.

    Attributes:
        files: Relative paths (e.g. ``HR/Scene15.png``) of the affected files.
    """

    def __init__(self, files: list[str]) -> None:
        self.files = files
        shown = files[:MAX_UNREADABLE_FILES_IN_MESSAGE]
        suffix = f" (+{len(files) - len(shown)} more)" if len(files) > len(shown) else ""
        message = (
            f"{len(files)} unreadable image(s) in dataset: "
            + ", ".join(shown)
            + suffix
            + ". Fix or remove them, then retry."
        )
        super().__init__(message)


def find_unreadable_images(dataset_dir: Path,
                           reporter: Optional[ProgressReporter] = None,
                           ) -> list[str]:
    """Find images that cannot be read or are missing from disk.

    Scans every registered pair (from ``manifest.json`` when present,
    otherwise by matching files between ``HR/`` and ``LR/``) and decodes
    both sides. Files that fail to decode, or that are registered in the
    manifest but missing on disk, are returned as relative paths such as
    ``HR/Scene15.png``.

    Args:
        dataset_dir: Path to the dataset directory.
        reporter: Optional progress reporter.

    Returns:
        List of relative paths of unreadable or missing image files.
    """
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"

    rel_paths: list[Path] = []
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            for entry in manifest_data.get("pairs", []):
                hr_rel = entry.get("hr") or entry.get("HR")
                lr_rel = entry.get("lr") or entry.get("LR")
                if hr_rel:
                    rel_paths.append(Path(hr_rel))
                if lr_rel:
                    rel_paths.append(Path(lr_rel))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Could not read manifest for integrity scan: %s", e)
            rel_paths = []
    else:
        hr_dir = dataset_dir / "HR"
        lr_dir = dataset_dir / "LR"
        if hr_dir.is_dir():
            for p in list_images(hr_dir):
                rel_paths.append(Path("HR") / p.name)
        if lr_dir.is_dir():
            for p in list_images(lr_dir):
                rel_paths.append(Path("LR") / p.name)

    unreadable: list[str] = []
    if not rel_paths:
        return unreadable

    reporter = reporter or ProgressReporter()
    reporter.start(total=len(rel_paths), desc="Checking Image Integrity")

    for rel in rel_paths:
        abs_path = dataset_dir / rel
        if not abs_path.is_file() or cv2.imread(str(abs_path)) is None:
            unreadable.append(str(rel))
        reporter.update(1)

    reporter.finish()
    return unreadable


def save_health_report(dataset_dir: Path, report: dict) -> None:
    path = dataset_dir / HEALTH_REPORT_FILENAME
    try:
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError as e:
        log.warning("Failed to save health report to %s: %s", path, e)


def load_health_report(dataset_dir: Path) -> dict | None:
    path = dataset_dir / HEALTH_REPORT_FILENAME
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        report.setdefault("unreadable", [])
        return report
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Failed to load health report from %s: %s", path, e)
        return None


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
"""Percentile used by the fallback heuristic to distinguish full-range
from limited-range encodings. If the bottom 15% of frames average
below 10.0, the data is assumed full-range (0-255)."""

MAX_THRESHOLD: float = 25.0
"""Upper clamp for the computed adaptive threshold. Prevents the threshold
from exceeding 25.0 even if the detected gap is very large, avoiding
false positives on legitimately dark-but-valid content."""

STD_THRESHOLD: float = 3.0
"""Minimum pixel-value standard deviation required to consider an image
as containing visible content. Images below this threshold are
featureless (true black frames). Prevents low-mean but high-variance
dark scenes from being incorrectly flagged as black frames."""

SUSPICIOUS_STD_THRESHOLD: float = 8.0
"""Standard deviation threshold for the ``suspicious_frames`` list.
Images with mean below the adaptive threshold but std between
``STD_THRESHOLD`` and this value are flagged as suspicious (they may be
very dark content rather than true black frames)."""

MAX_PIXEL_BLACK_THRESHOLD: int = 20
"""Maximum pixel intensity anywhere in the image for the secondary
absolute-near-black check. If ``max_pixel < this`` even when std is
above ``STD_THRESHOLD``, the frame is still flagged as black."""

FULL_RANGE_FALLBACK: float = 3.5
"""Fallback threshold used when Otsu finds no frames below its threshold
and the data suggests a full-range (0-255) encoding. 3.5 is tight
enough to catch only truly black/near-black frames."""

LIMITED_RANGE_FALLBACK: float = 18.5
"""Fallback threshold used when Otsu finds no frames below its threshold
and the data suggests a limited-range (16-235) encoding. 18.5 corresponds
to the BT.709 black level (~16) plus a small margin."""


def _otsu_threshold(hist: np.ndarray) -> float:
    """Compute Otsu's optimal binary threshold from a 256-bin histogram.

    Finds the threshold that minimises intra-class variance (equivalently
    maximises inter-class variance) between the "dark" and "bright"
    populations.

    Args:
        hist: 256-element histogram array.

    Returns:
        Threshold value (0-255) separating the two classes.
    """
    total = hist.sum()
    if total == 0:
        return 3.0

    hist_n = hist.astype(np.float64) / total
    cum_sum = np.cumsum(hist_n)
    cum_mean = np.cumsum(hist_n * np.arange(256))
    mean_total = cum_mean[-1]

    best_t, best_v = 0, 0.0
    for t in range(256):
        w0, w1 = cum_sum[t], 1.0 - cum_sum[t]
        if w0 == 0 or w1 == 0:
            continue
        mu0 = cum_mean[t] / w0
        mu1 = (mean_total - cum_mean[t]) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_v:
            best_v, best_t = var, t

    return float(best_t)


def _compute_adaptive_threshold(image_means: list[float]) -> float:
    """Calculate a data-driven brightness threshold for black-frame detection.

    Uses Otsu's method on the full distribution of mean pixel intensities
    to find the optimal binary split between "dark" and "bright" frames.
    The result is clamped to ``MAX_THRESHOLD`` to avoid over-pruning
    legitimate dark content.

    Falls back to a conservative heuristic (``FULL_RANGE_FALLBACK`` /
    ``LIMITED_RANGE_FALLBACK``) when no frames fall below the Otsu
    threshold — indicating a clean dataset with no black frames.

    Args:
        image_means: List of mean pixel intensities per image.

    Returns:
        A float threshold value. Images with mean below this threshold
        are considered black frames.
    """
    if not image_means:
        return 3.0

    sorted_means = np.sort(image_means)
    log.info("Darkest frame: %.2f | Total frames: %d", sorted_means[0], len(sorted_means))

    hist, _ = np.histogram(sorted_means, bins=256, range=(0, 255))
    otsu_t = _otsu_threshold(hist)

    final = min(otsu_t, MAX_THRESHOLD)
    log.info("Otsu threshold: %.2f → clamped: %.2f", otsu_t, final)

    if np.sum(sorted_means < final) == 0:
        p15 = sorted_means[max(1, int(len(sorted_means) * DARK_PERCENTILE)) - 1]
        log.info("No frames below threshold. 15th percentile: %.2f", p15)
        if p15 < 10.0:
            log.info("Data leans Full Range (0-255). Fallback: %.2f", FULL_RANGE_FALLBACK)
            return FULL_RANGE_FALLBACK
        else:
            log.info("Data leans Limited Range (16-235). Fallback: %.2f", LIMITED_RANGE_FALLBACK)
            return LIMITED_RANGE_FALLBACK

    return final


def check_dataset_health(dataset_dir: Path,
                         reporter: Optional[ProgressReporter] = None,
                         ) -> dict:
    """Analyze dataset spatial properties, color channels, and detect black frames.

    Examines all images in both ``HR/`` and ``LR/`` subdirectories,
    collecting resolution and aspect-ratio distributions, channel counts,
    and mean pixel brightness. Uses an adaptive thresholding algorithm
    to identify completely black or near-black frames. Also validates
    LR dimensions against the expected scale factor from the manifest.

    Args:
        dataset_dir: Path to the dataset directory.
        reporter: Optional progress reporter.

    Returns:
        A dict with keys:
        ``total_pairs``, ``total_hr_images``, ``total_lr_images``,
        ``resolutions``, ``aspect_ratios``, ``channels``,
        ``computed_threshold``, ``black_frames``, ``suspicious_frames``,
        ``scale_mismatches``, ``unreadable``, ``frame_means``. On error,
        returns ``{"error": <message>}``.
    """
    hr_dir = dataset_dir / "HR"
    lr_dir = dataset_dir / "LR"
    if not hr_dir.is_dir():
        return {"error": "HR directory not found. Run validation/build first."}

    hr_files = list_images(hr_dir)
    lr_files = list_images(lr_dir) if lr_dir.is_dir() else []

    total_hr = len(hr_files)
    total_lr = len(lr_files)

    total_pairs = min(total_hr, total_lr) if total_lr > 0 else total_hr

    if total_hr == 0:
        return {"error": "No images found in HR directory to analyze."}

    resolutions = Counter()
    aspect_ratios = Counter()
    channels_summary = Counter()

    hr_image_means: list[float] = []
    hr_file_metadata: list[tuple[str, float, float, float]] = []

    reporter = reporter or ProgressReporter()

    # Phase 1: Analyze HR files
    reporter.start(total=total_hr, desc="Analyzing HR Images")
    for path in hr_files:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        h, w = img.shape[:2]
        resolutions[f"HR:{w}x{h}"] += 1
        aspect_ratios[f"HR:{round(w / h, 2)}"] += 1

        color_data = _extract_color_data(img, channels_summary)
        img_mean = float(np.mean(color_data))
        img_std = float(np.std(color_data))
        img_max = float(np.max(color_data))

        hr_image_means.append(img_mean)
        hr_file_metadata.append((f"HR/{path.name}", img_mean, img_std, img_max))
        reporter.update(1)
    reporter.finish()

    # Phase 2: Analyze LR files (means only, no resolution/channel tracking)
    lr_file_metadata: list[tuple[str, float, float, float]] = []
    lr_dims: dict[str, tuple[int, int]] = {}
    if total_lr > 0:
        reporter.start(total=total_lr, desc="Analyzing LR Images")
        for path in lr_files:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            h, w = img.shape[:2]
            lr_dims[path.name] = (w, h)

            color_data = _extract_color_data(img, channels_summary)
            img_mean = float(np.mean(color_data))
            img_std = float(np.std(color_data))
            img_max = float(np.max(color_data))

            lr_file_metadata.append((f"LR/{path.name}", img_mean, img_std, img_max))
            reporter.update(1)
        reporter.finish()

    # Phase 3: Compute adaptive threshold from HR distribution
    threshold = _compute_adaptive_threshold(hr_image_means)

    # Phase 4: Flag black and suspicious frames (both HR and LR)
    all_metadata = hr_file_metadata + lr_file_metadata
    black_frames: list[str] = []
    suspicious_frames: list[str] = []
    frame_means: dict[str, float] = {}

    for rel_path, img_mean, img_std, img_max in all_metadata:
        frame_means[rel_path] = round(img_mean, 2)
        is_below_threshold = img_mean < threshold
        is_featureless = img_std < STD_THRESHOLD
        is_absolutely_dark = img_max < MAX_PIXEL_BLACK_THRESHOLD

        if is_below_threshold and (is_featureless or is_absolutely_dark):
            black_frames.append(rel_path)
        elif is_below_threshold and img_std < SUSPICIOUS_STD_THRESHOLD:
            suspicious_frames.append(rel_path)

    # Phase 5: Scale validation
    manifest_path = dataset_dir / "manifest.json"
    scale = 4
    scale_mismatches: list[str] = []
    if manifest_path.is_file():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            scale = int(manifest_data.get("config", {}).get("scale", 4))
        except (json.JSONDecodeError, OSError):
            pass

    if total_hr > 0 and total_lr > 0 and scale > 1:
        hr_dims_map: dict[str, tuple[int, int]] = {}
        for path in hr_files:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                hr_dims_map[path.name] = (w, h)

        for name, (lr_w, lr_h) in lr_dims.items():
            if name not in hr_dims_map:
                continue
            hr_w, hr_h = hr_dims_map[name]
            expected_lr_w = round(hr_w / scale)
            expected_lr_h = round(hr_h / scale)
            if abs(lr_w - expected_lr_w) > 1 or abs(lr_h - expected_lr_h) > 1:
                scale_mismatches.append(
                    f"LR/{name}: expected {expected_lr_w}x{expected_lr_h}, "
                    f"got {lr_w}x{lr_h} (HR is {hr_w}x{hr_h}, scale={scale})"
                )

    return {
        "total_pairs": total_pairs,
        "total_hr_images": total_hr,
        "total_lr_images": total_lr,
        "resolutions": dict(resolutions),
        "aspect_ratios": dict(aspect_ratios),
        "channels": dict(channels_summary),
        "computed_threshold": round(threshold, 2),
        "black_frames": black_frames,
        "suspicious_frames": suspicious_frames,
        "scale_mismatches": scale_mismatches,
        "frame_means": frame_means,
        "unreadable": find_unreadable_images(dataset_dir, reporter=reporter),
    }


def _resolve_pair(dataset_dir: Path, rel: Path) -> list[Path]:
    """Resolve a relative path (e.g. ``HR/x.png``) to its full HR/LR pair.

    Uses ``manifest.json`` when available, otherwise matches the twin by
    stem name between ``HR/`` and ``LR/`` (mirroring ``pair_hr_lr``).

    Returns:
        List of absolute paths of the pair files that exist on disk.
    """
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            for entry in manifest_data.get("pairs", []):
                hr_rel = Path(entry.get("hr") or entry.get("HR") or "")
                lr_rel = Path(entry.get("lr") or entry.get("LR") or "")
                if hr_rel == rel:
                    return [dataset_dir / hr_rel, dataset_dir / lr_rel]
                if lr_rel == rel:
                    return [dataset_dir / hr_rel, dataset_dir / lr_rel]
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Could not read manifest for pair resolution: %s", e)

    side = rel.parts[0] if len(rel.parts) > 1 else ""
    if side == "HR":
        hr_dir, lr_dir = dataset_dir / "HR", dataset_dir / "LR"
        if not lr_dir.is_dir():
            return [dataset_dir / rel]
        for hr_p, lr_p in pair_hr_lr(hr_dir, lr_dir):
            if hr_p == dataset_dir / rel:
                return [hr_p, lr_p]
    elif side == "LR":
        hr_dir, lr_dir = dataset_dir / "HR", dataset_dir / "LR"
        if not hr_dir.is_dir():
            return [dataset_dir / rel]
        for hr_p, lr_p in pair_hr_lr(hr_dir, lr_dir):
            if lr_p == dataset_dir / rel:
                return [hr_p, lr_p]
    return [dataset_dir / rel]


def prune_pairs(dataset_dir: Path, rel_paths: list[str],
                reporter: Optional[ProgressReporter] = None) -> None:
    """Delete HR/LR image pairs and update the dataset manifest.

    Resolves each relative path (e.g. ``HR/Scene15.png``) to its full pair,
    deletes both sides from disk, then filters the entries out of
    ``manifest.json`` so it stays consistent with the filesystem state.

    Args:
        dataset_dir: Path to the dataset directory.
        rel_paths: Relative paths of files whose pairs should be removed.
        reporter: Optional progress reporter.

    Raises:
        RuntimeError: If any files could not be deleted.
    """
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    removed_rel: set[str] = set()

    reporter = reporter or ProgressReporter()
    reporter.start(total=len(rel_paths), desc="Removing Pairs")

    failed = []
    for rel in rel_paths:
        for abs_path in _resolve_pair(dataset_dir, Path(rel)):
            try:
                if abs_path.is_file():
                    abs_path.unlink()
                if str(abs_path).startswith(str(dataset_dir)):
                    removed_rel.add(str(abs_path.relative_to(dataset_dir)))
            except OSError as e:
                failed.append((str(abs_path), e))
        reporter.update(1)

    reporter.finish()

    if failed:
        msg = "; ".join(f"{p}: {e}" for p, e in failed)
        raise RuntimeError(f"Failed to delete some pair files: {msg}")

    # Sync and rewrite structural tracks inside manifest file
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            manifest_data["pairs"] = [
                p for p in manifest_data.get("pairs", [])
                if (p.get("hr") or p.get("HR") or "") not in removed_rel
                and (p.get("lr") or p.get("LR") or "") not in removed_rel
            ]

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=4)

        except (json.JSONDecodeError, OSError) as e:
            log.warning("Could not sync manifest.json adjustments: %s", e)

    # Stale health report is no longer valid after pruning
    health_path = dataset_dir / HEALTH_REPORT_FILENAME
    if health_path.exists():
        try:
            health_path.unlink()
        except OSError as e:
            log.warning("Could not remove stale health report: %s", e)


def prune_black_frames(dataset_dir: Path, black_filenames: list[str],
                       reporter: Optional[ProgressReporter] = None) -> None:
    """Delete black frame pairs from disk and update the dataset manifest.

    Accepts either bare filenames (e.g. ``"f0001.png"``) or prefixed relative
    paths (e.g. ``"HR/f0001.png"``, ``"LR/f0001.png"``). Both forms resolve
    to the full HR/LR pair via :func:`prune_pairs`.

    Args:
        dataset_dir: Path to the dataset directory.
        black_filenames: List of filenames or relative paths to remove.
        reporter: Optional progress reporter.

    Raises:
        RuntimeError: If any files could not be deleted.
    """
    rel_paths: list[str] = []
    for name in black_filenames:
        if name.startswith("HR/") or name.startswith("LR/"):
            rel_paths.append(name)
        else:
            rel_paths.append(f"HR/{name}")
    prune_pairs(dataset_dir, rel_paths, reporter=reporter)