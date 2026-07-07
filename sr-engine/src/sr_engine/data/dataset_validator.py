"""Dataset validator — checks an existing HR/LR folder matches the manifest.json."""

import json
from dataclasses import dataclass, field
from pathlib import Path
import cv2
from tqdm import tqdm


@dataclass
class ValidationReport:
    """Result of validating a dataset directory."""
    ok: bool
    num_pairs: int = 0
    problems: list[str] = field(default_factory=list)


def validate(dataset_dir: Path) -> ValidationReport:
    """Validate that *dataset_dir* contains a well-formed HR/LR dataset tracking the manifest.

    Checks:
        - HR/ and LR/ subdirectories exist.
        - manifest.json exists and is parseable.
        - Every image pair cataloged in the manifest exists physically on disk.
        - All referenced files are valid PNG images.
        - HR image dimensions are exactly *scale* times LR dimensions.
        - No orphan files exist on disk that are missing from the manifest tracking log.

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
        scale = int(manifest_data.get("config", {}).get("scale", 4))
        manifest_pairs = manifest_data.get("pairs", [])
    except json.JSONDecodeError as e:
        problems.append(f"Failed to parse manifest.json: {e}")
        return ValidationReport(ok=False, problems=problems)

    # Build tracking sets of expected files from the manifest
    expected_hr = set()
    expected_lr = set()

    for pair in manifest_pairs:
        if "hr" in pair:
            expected_hr.add(Path(pair["hr"]).name)
        if "lr" in pair:
            expected_lr.add(Path(pair["lr"]).name)

    # 3. Integrity and Dimensional Scale Checks via Manifest Records
    num_pairs = 0

    for pair in tqdm(
        manifest_pairs,
        desc="🔍 Checking Manifest Alignment & Integrity",
        unit="pair"
    ):
        hr_rel = pair.get("hr")
        lr_rel = pair.get("lr")

        if not hr_rel or not lr_rel:
            problems.append(f"Malformed manifest track entry: missing path mappings in entry: {pair}")
            continue

        hr_img_path = dataset_dir / hr_rel
        lr_img_path = dataset_dir / lr_rel

        # A. Check for missing disk files registered in manifest
        if not hr_img_path.is_file():
            problems.append(f"Manifest alignment failure: File '{hr_rel}' is logged in manifest.json but missing from disk.")
            continue
        if not lr_img_path.is_file():
            problems.append(f"Manifest alignment failure: File '{lr_rel}' is logged in manifest.json but missing from disk.")
            continue

        # B. Read files using OpenCV to verify stability
        hr_img = cv2.imread(str(hr_img_path))
        lr_img = cv2.imread(str(lr_img_path))

        if hr_img is None:
            problems.append(f"Corrupted Image: HR file '{hr_rel}' is unreadable or malformed.")
            continue
        if lr_img is None:
            problems.append(f"Corrupted Image: LR file '{lr_rel}' is unreadable or malformed.")
            continue

        # C. Check scale metrics strictly
        hr_h, hr_w = hr_img.shape[:2]
        lr_h, lr_w = lr_img.shape[:2]

        if hr_h != lr_h * scale or hr_w != lr_w * scale:
            problems.append(
                f"Dimension mismatch on '{Path(hr_rel).name}': HR dimensions ({hr_w}x{hr_h}) "
                f"are not exactly {scale}x scale multiplier of LR dimensions ({lr_w}x{lr_h})."
            )
            continue

        num_pairs += 1

    # 4. Check for Orphaned Files (Files on disk that aren't in the manifest)
    disk_hr_files = {p.name for p in hr_dir.glob("*.png")}
    disk_lr_files = {p.name for p in lr_dir.glob("*.png")}

    orphaned_hr = disk_hr_files - expected_hr
    orphaned_lr = disk_lr_files - expected_lr

    for filename in orphaned_hr:
        problems.append(f"Orphaned asset: '{filename}' exists in HR/ directory but is missing from manifest.json.")
    for filename in orphaned_lr:
        problems.append(f"Orphaned asset: '{filename}' exists in LR/ directory but is missing from manifest.json.")

    # 5. Final decision matrix evaluation
    is_ok = len(problems) == 0 and num_pairs > 0
    if num_pairs == 0 and len(problems) == 0:
        problems.append("Dataset manifest is completely empty (0 registered frame tracking structures).")
        is_ok = False

    return ValidationReport(ok=is_ok, num_pairs=num_pairs, problems=problems)