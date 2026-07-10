"""Tests for data/dataset_builder.py — build_from_preprocessed, build_from_video."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from sr_engine.data.dataset_builder import build_from_preprocessed


class TestBuildFromPreprocessed:
    def test_missing_hr_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="missing"):
            build_from_preprocessed(tmp_path / "empty", {})

    def test_missing_lr_dir(self, tmp_path):
        (tmp_path / "HR").mkdir()
        with pytest.raises(FileNotFoundError, match="missing"):
            build_from_preprocessed(tmp_path, {})

    def test_empty_hr_dir(self, tmp_path):
        (tmp_path / "HR").mkdir()
        (tmp_path / "LR").mkdir()
        with pytest.raises(ValueError, match="No source PNG images"):
            build_from_preprocessed(tmp_path, {})

    def test_valid_dataset(self, minimal_dataset_with_manifest):
        result = build_from_preprocessed(minimal_dataset_with_manifest, {"scale": 4})
        assert result == minimal_dataset_with_manifest

    def test_validation_failure_cleans_up_manifest(self, tmp_path):
        import cv2
        import numpy as np

        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        hr_dir.mkdir()
        lr_dir.mkdir()

        cv2.imwrite(str(hr_dir / "frame_0000.png"), np.ones((128, 128, 3), dtype=np.uint8) * 200)
        cv2.imwrite(str(lr_dir / "frame_0000.png"), np.ones((16, 16, 3), dtype=np.uint8) * 200)

        with pytest.raises(RuntimeError, match="validation failed"):
            build_from_preprocessed(tmp_path, {"scale": 4})

        assert not (tmp_path / "manifest.json").exists()
