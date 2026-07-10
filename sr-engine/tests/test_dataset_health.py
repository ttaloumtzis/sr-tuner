"""Tests for data/dataset_health.py — health check and adaptive threshold."""

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from sr_engine.data.dataset_health import (
    _extract_color_data,
    _compute_adaptive_threshold,
    check_dataset_health,
    prune_black_frames,
)


class TestExtractColorData:
    def test_grayscale(self):
        img = np.ones((16, 16), dtype=np.uint8) * 128
        counter = Counter()
        result = _extract_color_data(img, counter)
        assert result.shape == (16, 16)
        assert counter["Grayscale (1 channel)"] == 1

    def test_rgb(self):
        img = np.ones((16, 16, 3), dtype=np.uint8) * 128
        counter = Counter()
        result = _extract_color_data(img, counter)
        assert result.shape == (16, 16, 3)
        assert counter["RGB (3 channels)"] == 1

    def test_rgba_drops_alpha(self):
        img = np.ones((16, 16, 4), dtype=np.uint8) * 128
        counter = Counter()
        result = _extract_color_data(img, counter)
        assert result.shape[2] == 3
        assert counter["RGBA (4 channels)"] == 1

    def test_unknown_channels(self):
        img = np.ones((16, 16, 5), dtype=np.uint8) * 128
        counter = Counter()
        result = _extract_color_data(img, counter)
        assert counter["Unknown (5 channels)"] == 1


class TestComputeAdaptiveThreshold:
    def test_empty_list(self):
        assert _compute_adaptive_threshold([]) == 3.0

    def test_single_value(self):
        result = _compute_adaptive_threshold([10.0])
        assert result > 10.0

    def test_low_values_full_range(self):
        values = [1.0, 1.2, 1.5, 1.8, 2.0, 2.1, 2.3, 50.0, 100.0, 150.0]
        threshold = _compute_adaptive_threshold(values)
        assert 1.0 <= threshold <= 25.0

    def test_bright_values_limited_range(self):
        values = [18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 100.0]
        threshold = _compute_adaptive_threshold(values)
        # percentile_15_score is around 21-24, which is > 10, so limited range fallback
        assert threshold > 10.0


class TestCheckDatasetHealth:
    def test_missing_hr_dir(self, tmp_path):
        result = check_dataset_health(tmp_path / "empty")
        assert "error" in result
        assert "HR directory not found" in result["error"]

    def test_empty_hr_dir(self, tmp_path):
        (tmp_path / "HR").mkdir()
        result = check_dataset_health(tmp_path)
        assert "error" in result
        assert "No images found" in result["error"]

    def test_valid_dataset(self, tmp_path, minimal_dataset):
        result = check_dataset_health(minimal_dataset)
        assert "error" not in result
        assert result["total_images"] == 3
        assert "resolutions" in result
        assert "aspect_ratios" in result
        assert "channels" in result
        assert "computed_threshold" in result
        assert "black_frames" in result


class TestPruneBlackFrames:
    def test_removes_files(self, tmp_path):
        for name in ["HR/frame_0000.png", "LR/frame_0000.png"]:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.ones((16, 16, 3), dtype=np.uint8) * 255)

        prune_black_frames(tmp_path, ["frame_0000.png"])
        assert not (tmp_path / "HR" / "frame_0000.png").exists()
        assert not (tmp_path / "LR" / "frame_0000.png").exists()

    def test_updates_manifest(self, tmp_path):
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        hr_dir.mkdir(parents=True)
        lr_dir.mkdir(parents=True)

        for i in range(3):
            cv2.imwrite(str(hr_dir / f"frame_{i:04d}.png"), np.ones((16, 16, 3), dtype=np.uint8) * 255)
            cv2.imwrite(str(lr_dir / f"frame_{i:04d}.png"), np.ones((16, 16, 3), dtype=np.uint8) * 255)

        import json
        manifest = {
            "pairs": [
                {"hr": f"HR/frame_{i:04d}.png", "lr": f"LR/frame_{i:04d}.png"}
                for i in range(3)
            ],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        prune_black_frames(tmp_path, ["frame_0001.png"])
        updated = json.loads((tmp_path / "manifest.json").read_text())
        assert len(updated["pairs"]) == 2
        assert all("frame_0001" not in p["hr"] for p in updated["pairs"])

    def test_raises_on_failed_deletion(self, tmp_path, monkeypatch):
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        hr_dir.mkdir(parents=True)
        lr_dir.mkdir(parents=True)
        (hr_dir / "bad.png").write_text("data")
        (lr_dir / "bad.png").write_text("data")

        from unittest.mock import patch
        original_unlink = Path.unlink

        def failing_unlink(self, *a, **kw):
            if self.name == "bad.png":
                raise OSError("Permission denied")
            return original_unlink(self, *a, **kw)

        with patch.object(Path, "unlink", failing_unlink):
            with pytest.raises(RuntimeError, match="Failed to delete"):
                prune_black_frames(tmp_path, ["bad.png"])
