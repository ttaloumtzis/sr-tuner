"""Tests for data/dataset_builder.py — build_from_preprocessed, build_from_video."""

from unittest.mock import patch

import pytest

from sr_engine.data.dataset_builder import build_from_preprocessed, build_from_video


class TestBuildFromPreprocessed:
    """Tests for ``build_from_preprocessed``."""

    def test_missing_hr_dir(self, tmp_path):
        """A missing HR directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="missing"):
            build_from_preprocessed(tmp_path / "empty", {})

    def test_missing_lr_dir(self, tmp_path):
        """A missing LR directory should raise FileNotFoundError."""
        (tmp_path / "HR").mkdir()
        with pytest.raises(FileNotFoundError, match="missing"):
            build_from_preprocessed(tmp_path, {})

    def test_empty_hr_dir(self, tmp_path):
        """An empty HR directory should raise ValueError."""
        (tmp_path / "HR").mkdir()
        (tmp_path / "LR").mkdir()
        with pytest.raises(ValueError, match="No source PNG images"):
            build_from_preprocessed(tmp_path, {})

    def test_valid_dataset(self, minimal_dataset_with_manifest):
        """A valid dataset should return its path unchanged."""
        result = build_from_preprocessed(minimal_dataset_with_manifest, {"scale": 4})
        assert result == minimal_dataset_with_manifest

    def test_validation_failure_cleans_up_manifest(self, tmp_path):
        """A validation failure should remove any generated manifest."""
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


class TestBuildFromVideo:
    """Tests for ``build_from_video``."""

    def test_cleans_up_on_extraction_failure(self, sample_video, tmp_path):
        """Output directory should be removed when extraction fails."""
        with patch(
            "sr_engine.data.dataset_builder.extract_frames",
            side_effect=RuntimeError("Extraction failed"),
        ):
            with pytest.raises(RuntimeError, match="Extraction failed"):
                build_from_video(
                    video_path=sample_video,
                    out_dir=tmp_path / "out",
                    config={},
                )
        assert not (tmp_path / "out").exists()

    def test_cleans_up_on_degradation_failure(self, sample_video, tmp_path):
        """Output directory should be removed when degradation fails."""
        with patch(
            "sr_engine.data.dataset_builder.extract_frames",
            return_value=[tmp_path / "dummy" / "frame_0000.png"],
        ):
            with patch(
                "sr_engine.data.dataset_builder.batch_degrade",
                side_effect=RuntimeError("Degradation failed"),
            ):
                with pytest.raises(RuntimeError, match="Degradation failed"):
                    build_from_video(
                        video_path=sample_video,
                        out_dir=tmp_path / "out",
                        config={},
                    )
        assert not (tmp_path / "out").exists()

    def test_cleans_up_on_validation_failure(self, sample_video, tmp_path):
        """Output directory should be removed when validation fails."""
        out_dir = tmp_path / "out"
        hr_file = out_dir / "HR" / "frame_0000.png"
        lr_file = out_dir / "LR" / "frame_0000.png"
        (out_dir / "HR").mkdir(parents=True)
        (out_dir / "LR").mkdir(parents=True)

        with patch(
            "sr_engine.data.dataset_builder.extract_frames",
            return_value=[hr_file],
        ):
            with patch(
                "sr_engine.data.dataset_builder.batch_degrade",
                return_value=[(hr_file, lr_file)],
            ):
                with patch(
                    "sr_engine.data.dataset_builder.validate",
                    return_value=type("Report", (), {"ok": False, "problems": ["Corrupted data"]})(),
                ):
                    with pytest.raises(RuntimeError, match="validation failed"):
                        build_from_video(
                            video_path=sample_video,
                            out_dir=out_dir,
                            config={"scale": 4},
                        )
        assert not out_dir.exists()
