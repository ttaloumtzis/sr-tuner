"""Tests for dataset validation."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sr_engine.data.dataset_validator import validate, ValidationReport


def _make_image(path: Path, w: int, h: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


class TestValidate:
    @pytest.fixture
    def dataset_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "dataset"
        hr_dir = d / "HR"
        lr_dir = d / "LR"
        hr_dir.mkdir(parents=True)
        lr_dir.mkdir(parents=True)
        scale = 4
        pairs = []
        for i in range(3):
            _make_image(hr_dir / f"frame_{i:04d}.png", 256 * scale, 128 * scale)
            _make_image(lr_dir / f"frame_{i:04d}.png", 256, 128)
            pairs.append({
                "hr": f"HR/frame_{i:04d}.png",
                "lr": f"LR/frame_{i:04d}.png",
            })
        manifest = {"config": {"scale": scale}, "pairs": pairs}
        with open(d / "manifest.json", "w") as f:
            json.dump(manifest, f)
        return d

    def test_valid_dataset_returns_ok(self, dataset_dir: Path):
        report = validate(dataset_dir)
        assert report.ok
        assert report.num_pairs == 3

    def test_missing_manifest_returns_not_ok(self, tmp_path: Path):
        d = tmp_path / "no_manifest"
        (d / "HR").mkdir(parents=True)
        (d / "LR").mkdir(parents=True)
        report = validate(d)
        assert not report.ok
        assert any("manifest" in p.lower() for p in report.problems)

    def test_corrupt_manifest_returns_not_ok(self, tmp_path: Path):
        d = tmp_path / "corrupt"
        (d / "HR").mkdir(parents=True)
        (d / "LR").mkdir(parents=True)
        with open(d / "manifest.json", "w") as f:
            f.write("not valid json")
        report = validate(d)
        assert not report.ok

    def test_scale_mismatch_detected(self, tmp_path: Path):
        d = tmp_path / "scale_mismatch"
        hr_dir = d / "HR"
        lr_dir = d / "LR"
        hr_dir.mkdir(parents=True)
        lr_dir.mkdir(parents=True)
        _make_image(hr_dir / "frame_0000.png", 256, 128)
        _make_image(lr_dir / "frame_0000.png", 256, 128)  # wrong scale (1x, not 4x)
        manifest = {"config": {"scale": 4}, "pairs": [{"hr": "HR/frame_0000.png", "lr": "LR/frame_0000.png"}]}
        with open(d / "manifest.json", "w") as f:
            json.dump(manifest, f)
        report = validate(d)
        assert not report.ok
        assert any("dimension" in p.lower() for p in report.problems)

    def test_missing_hr_file_detected(self, tmp_path: Path):
        d = tmp_path / "missing_hr"
        (d / "HR").mkdir(parents=True)
        (d / "LR").mkdir(parents=True)
        _make_image(d / "LR/frame_0000.png", 256, 128)
        manifest = {"config": {"scale": 4}, "pairs": [{"hr": "HR/frame_0000.png", "lr": "LR/frame_0000.png"}]}
        with open(d / "manifest.json", "w") as f:
            json.dump(manifest, f)
        report = validate(d)
        assert not report.ok

    def test_empty_manifest_returns_not_ok(self, tmp_path: Path):
        d = tmp_path / "empty"
        (d / "HR").mkdir(parents=True)
        (d / "LR").mkdir(parents=True)
        manifest = {"config": {"scale": 4}, "pairs": []}
        with open(d / "manifest.json", "w") as f:
            json.dump(manifest, f)
        report = validate(d)
        assert not report.ok

    def test_validation_report_dataclass(self):
        report = ValidationReport(ok=True, num_pairs=5, problems=[])
        assert report.ok
        assert report.num_pairs == 5
        assert len(report.problems) == 0
