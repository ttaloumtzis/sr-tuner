"""Dataset builder — orchestrates video -> HR/LR folder output."""
from .dataset_validator import validate
from .degrade import batch_degrade
from .video_extract import extract_frames
import json
from pathlib import Path

from sr_engine.utils.logging import get_logger

log = get_logger(__name__)

def build_from_video(
        video_path: Path,
        out_dir: Path,
        config: dict,
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

    # 1. Extract the video frames to HR subfolder
    hr_paths = extract_frames(
        video_path=video_path,
        out_dir=out_dir_hr,
        frame_rate=config.get("frame_rate"),
        start_time=config.get("start_time", 0.0),
        duration=config.get("duration"),
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
    )

    if len(hr_lr_pairs) < len(hr_paths):
        skipped = len(hr_paths) - len(hr_lr_pairs)
        log.warning("%d frame(s) failed to degrade and were skipped.", skipped)

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
    report = validate(out_dir)
    if not report.ok:
        # If verification fails, clean up the bad manifest to keep things unstable/invalidated
        if manifest_path.exists():
            manifest_path.unlink()

        # Structure problems list into a clean message
        error_msg = "\n- ".join(report.problems)
        raise RuntimeError(
            f"Dataset validation failed for '{out_dir}'! Found the following problems:\n- {error_msg}"
        )

    log.info("Successfully verified and built a stable dataset with %d pairs at: %s", report.num_pairs, out_dir)
    return out_dir



def build_from_preprocessed(
        dataset_dir: Path,
        config: dict,
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
    hr_paths = sorted(hr_dir.glob("*.png"))

    if not hr_paths:
        raise ValueError(f"No source PNG images found inside the HR directory: {hr_dir}")

    # Map the pairings together based on matching filenames
    pairs = []
    for hr_path in hr_paths:
        filename = hr_path.name
        lr_path = lr_dir / filename

        # Only include it in the manifest tracking loop if it actually exists in both places
        if lr_path.is_file():
            pairs.append({
                "hr": str(hr_path.relative_to(dataset_dir)),
                "lr": str(lr_path.relative_to(dataset_dir))
            })

    # 3. Write a tentative manifest.json file so the validator has its audit target
    manifest_data = {
        "config": {
            "scale": config.get("scale", 4),
            "frame_rate": config.get("frame_rate"),
            "video_source": "preprocessed_folder",
        },
        "pairs": pairs,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4, ensure_ascii=False)

    # 4. Trigger Deep Validation Scan
    report = validate(dataset_dir)
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