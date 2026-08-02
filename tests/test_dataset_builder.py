"""Tests for data/dataset_builder.py — build_from_preprocessed, build_from_video."""

import json
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from sr_engine.data.dataset_builder import build_from_preprocessed, build_from_video, build_manifest, inspect_dataset


class TestInspectDataset:
    """Tests for ``inspect_dataset``."""

    def test_detects_scale_from_first_pair(self, tmp_path):
        """Scale should be derived from the first readable HR/LR pair."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        hr_dir.mkdir()
        lr_dir.mkdir()
        for i in range(3):
            cv2.imwrite(str(hr_dir / f"f{i}.png"), np.ones((128, 128, 3), dtype=np.uint8) * 200)
            cv2.imwrite(str(lr_dir / f"f{i}.png"), np.ones((32, 32, 3), dtype=np.uint8) * 200)

        info = inspect_dataset(tmp_path)
        assert info["hr_count"] == 3
        assert info["lr_count"] == 3
        assert info["pair_count"] == 3
        assert info["scale_ratio"] == pytest.approx(4.0)
        assert info["scale_exact"] is True
        assert info["scale_w"] == pytest.approx(4.0)
        assert info["scale_h"] == pytest.approx(4.0)
        assert info["hr_size"] == {"width": 128, "height": 128}
        assert info["lr_size"] == {"width": 32, "height": 32}
        assert info["has_manifest"] is False
        assert info["warnings"] == []

    def test_jpg_dataset_supported(self, tmp_path):
        """Non-PNG formats should be counted and paired."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        hr_dir.mkdir()
        lr_dir.mkdir()
        cv2.imwrite(str(hr_dir / "f.jpg"), np.ones((64, 64, 3), dtype=np.uint8) * 200)
        cv2.imwrite(str(lr_dir / "f.jpg"), np.ones((16, 16, 3), dtype=np.uint8) * 200)

        info = inspect_dataset(tmp_path)
        assert info["hr_count"] == 1
        assert info["pair_count"] == 1
        assert info["scale_ratio"] == pytest.approx(4.0)

    def test_flickr2k_style_suffix_pairing(self, tmp_path):
        """LR files with an x<scale> suffix pair with plain HR names."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        hr_dir.mkdir()
        lr_dir.mkdir()
        cv2.imwrite(str(hr_dir / "000001.png"), np.ones((128, 128, 3), dtype=np.uint8) * 200)
        cv2.imwrite(str(lr_dir / "000001x2.png"), np.ones((64, 64, 3), dtype=np.uint8) * 200)

        info = inspect_dataset(tmp_path)
        assert info["pair_count"] == 1
        assert info["scale_ratio"] == pytest.approx(2.0)
        assert info["scale_exact"] is True
        assert info["hr_size"] == {"width": 128, "height": 128}

    def test_suffix_scale_fallback_when_unreadable(self, tmp_path):
        """Scale falls back to the LR suffix when dims cannot be read."""
        from conftest import _make_corrupt_image
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        hr_dir.mkdir()
        lr_dir.mkdir()
        # All images unreadable — scale can only come from the x4 suffix
        _make_corrupt_image(hr_dir / "f.png")
        _make_corrupt_image(lr_dir / "fx4.png")
        _make_corrupt_image(hr_dir / "g.png")
        _make_corrupt_image(lr_dir / "gx4.png")

        info = inspect_dataset(tmp_path)
        assert info["pair_count"] == 2
        assert info["scale_ratio"] == pytest.approx(4.0)
        assert info["scale_exact"] is True
        assert info["hr_size"] is None

    def test_cross_format_pairing(self, tmp_path):
        """HR png paired with LR jpg of the same stem."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        hr_dir.mkdir()
        lr_dir.mkdir()
        cv2.imwrite(str(hr_dir / "f.png"), np.ones((64, 64, 3), dtype=np.uint8) * 200)
        cv2.imwrite(str(lr_dir / "f.jpg"), np.ones((16, 16, 3), dtype=np.uint8) * 200)

        info = inspect_dataset(tmp_path)
        assert info["pair_count"] == 1
        assert info["scale_ratio"] == pytest.approx(4.0)

    def test_count_mismatch_warning(self, tmp_path):
        """Extra HR files without an LR match should be reported."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        hr_dir.mkdir()
        lr_dir.mkdir()
        for i in range(3):
            cv2.imwrite(str(hr_dir / f"f{i}.png"), np.ones((64, 64, 3), dtype=np.uint8) * 200)
            cv2.imwrite(str(lr_dir / f"f{i}.png"), np.ones((16, 16, 3), dtype=np.uint8) * 200)
        cv2.imwrite(str(hr_dir / "extra.png"), np.ones((64, 64, 3), dtype=np.uint8) * 200)

        info = inspect_dataset(tmp_path)
        assert info["pair_count"] == 3
        assert info["hr_count"] == 4
        assert any("no LR match" in w or "could not be matched" in w for w in info["warnings"])

    def test_corrupt_first_image_falls_back(self, tmp_path):
        """An unreadable first pair should not prevent scale detection."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        hr_dir.mkdir()
        lr_dir.mkdir()
        (hr_dir / "bad.png").write_bytes(b"not-an-image")
        (lr_dir / "bad.png").write_bytes(b"not-an-image")
        cv2.imwrite(str(hr_dir / "good.png"), np.ones((64, 64, 3), dtype=np.uint8) * 200)
        cv2.imwrite(str(lr_dir / "good.png"), np.ones((16, 16, 3), dtype=np.uint8) * 200)

        info = inspect_dataset(tmp_path)
        assert info["pair_count"] == 2
        assert info["scale_ratio"] == pytest.approx(4.0)

    def test_empty_dirs(self, tmp_path):
        """Missing HR/LR should yield warnings, not crash."""
        info = inspect_dataset(tmp_path / "empty")
        assert info["hr_count"] == 0
        assert info["pair_count"] == 0
        assert info["scale_ratio"] is None
        assert info["warnings"]

    def test_non_uniform_scale_warning(self, tmp_path):
        """Different width/height ratios should warn."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        hr_dir.mkdir()
        lr_dir.mkdir()
        cv2.imwrite(str(hr_dir / "f.png"), np.ones((128, 64, 3), dtype=np.uint8) * 200)
        cv2.imwrite(str(lr_dir / "f.png"), np.ones((32, 32, 3), dtype=np.uint8) * 200)

        info = inspect_dataset(tmp_path)
        assert any("Non-uniform" in w for w in info["warnings"])


class TestBuildManifest:
    """Tests for ``build_manifest``."""

    def test_writes_manifest_in_canonical_shape(self, tmp_path):
        """Manifest should match the build_from_preprocessed format."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        hr_dir.mkdir()
        lr_dir.mkdir()
        for i in range(2):
            cv2.imwrite(str(hr_dir / f"f{i}.png"), np.ones((64, 64, 3), dtype=np.uint8) * 200)
            cv2.imwrite(str(lr_dir / f"f{i}.png"), np.ones((16, 16, 3), dtype=np.uint8) * 200)

        result = build_manifest(tmp_path, 4.0)
        assert result == {"path": str(tmp_path), "scale": 4, "num_pairs": 2}

        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["config"]["scale"] == 4
        assert manifest["config"]["video_source"] == "preprocessed_folder"
        assert manifest["pairs"] == [
            {"hr": "HR/f0.png", "lr": "LR/f0.png"},
            {"hr": "HR/f1.png", "lr": "LR/f1.png"},
        ]

    def test_cross_format_pairs(self, tmp_path):
        """HR/foo.png should pair with LR/foo.jpg in the manifest."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        hr_dir.mkdir()
        lr_dir.mkdir()
        cv2.imwrite(str(hr_dir / "f.png"), np.ones((64, 64, 3), dtype=np.uint8) * 200)
        cv2.imwrite(str(lr_dir / "f.jpg"), np.ones((16, 16, 3), dtype=np.uint8) * 200)

        result = build_manifest(tmp_path, 4.0)
        assert result["num_pairs"] == 1
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["pairs"][0] == {"hr": "HR/f.png", "lr": "LR/f.jpg"}

    def test_near_integer_scale_rounded(self, tmp_path):
        """A near-integer scale (3.99) should be rounded to 4."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        hr_dir.mkdir()
        lr_dir.mkdir()
        cv2.imwrite(str(hr_dir / "f.png"), np.ones((64, 64, 3), dtype=np.uint8) * 200)
        cv2.imwrite(str(lr_dir / "f.png"), np.ones((16, 16, 3), dtype=np.uint8) * 200)

        result = build_manifest(tmp_path, 3.99)
        assert result["scale"] == 4

    def test_non_integer_scale_rejected(self, tmp_path):
        """A clearly non-integer scale should raise ValueError."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        hr_dir.mkdir()
        lr_dir.mkdir()
        cv2.imwrite(str(hr_dir / "f.png"), np.ones((64, 64, 3), dtype=np.uint8) * 200)
        cv2.imwrite(str(lr_dir / "f.png"), np.ones((16, 16, 3), dtype=np.uint8) * 200)

        with pytest.raises(ValueError, match="not a near-integer"):
            build_manifest(tmp_path, 2.5)

    def test_missing_dirs_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="missing"):
            build_manifest(tmp_path / "empty", 4.0)

    def test_no_pairs_raises(self, tmp_path):
        (tmp_path / "HR").mkdir()
        (tmp_path / "LR").mkdir()
        with pytest.raises(ValueError, match="No matching HR/LR"):
            build_manifest(tmp_path, 4.0)


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
        with pytest.raises(ValueError, match="No source images"):
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
