"""Tests for data/dataset_validator.py — dataset validation logic."""

import json

import cv2
import numpy as np

from sr_engine.data.dataset_validator import validate


class TestValidate:
    """Tests for ``validate``."""

    def test_validate_healthy(self, minimal_dataset_with_manifest):
        """A complete dataset with manifest should pass."""
        report = validate(minimal_dataset_with_manifest)
        assert report.ok is True

    def test_validate_jpg_dataset(self, tmp_path):
        """JPEG pairs should be accepted alongside PNGs."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        hr_dir.mkdir()
        lr_dir.mkdir()
        cv2.imwrite(str(hr_dir / "a.jpg"), np.ones((64, 64, 3), dtype=np.uint8) * 200)
        cv2.imwrite(str(lr_dir / "a.jpg"), np.ones((16, 16, 3), dtype=np.uint8) * 200)
        cv2.imwrite(str(hr_dir / "b.png"), np.ones((64, 64, 3), dtype=np.uint8) * 200)
        cv2.imwrite(str(lr_dir / "b.png"), np.ones((16, 16, 3), dtype=np.uint8) * 200)
        manifest = {
            "config": {"scale": 4},
            "pairs": [
                {"hr": "HR/a.jpg", "lr": "LR/a.jpg"},
                {"hr": "HR/b.png", "lr": "LR/b.png"},
            ],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        report = validate(tmp_path)
        assert report.ok is True
        assert report.num_pairs == 2

    def test_validate_orphan_jpg_reported(self, tmp_path):
        """An orphaned JPEG missing from the manifest should be flagged."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        hr_dir.mkdir()
        lr_dir.mkdir()
        cv2.imwrite(str(hr_dir / "a.png"), np.ones((64, 64, 3), dtype=np.uint8) * 200)
        cv2.imwrite(str(lr_dir / "a.png"), np.ones((16, 16, 3), dtype=np.uint8) * 200)
        cv2.imwrite(str(hr_dir / "orphan.jpg"), np.ones((64, 64, 3), dtype=np.uint8) * 200)
        manifest = {
            "config": {"scale": 4},
            "pairs": [{"hr": "HR/a.png", "lr": "LR/a.png"}],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        report = validate(tmp_path)
        assert report.ok is False
        assert any("orphan.jpg" in p for p in report.problems)

    def test_validate_missing_hr(self, tmp_path):
        """A dataset without an HR directory should produce a problem."""
        lr = tmp_path / "LR"
        lr.mkdir(parents=True)
        report = validate(tmp_path)
        assert report.ok is False
        assert any("HR" in p for p in report.problems)

    def test_validate_missing_lr(self, tmp_path):
        """A dataset without an LR directory should produce a problem."""
        hr = tmp_path / "HR"
        hr.mkdir(parents=True)
        report = validate(tmp_path)
        assert report.ok is False
        assert any("LR" in p for p in report.problems)

    def test_validate_missing_manifest(self, tmp_path):
        """A dataset without manifest should report it as a problem."""
        hr = tmp_path / "HR"
        lr = tmp_path / "LR"
        hr.mkdir(parents=True)
        lr.mkdir(parents=True)
        report = validate(tmp_path)
        assert report.ok is False
        assert any("manifest" in p.lower() for p in report.problems)
