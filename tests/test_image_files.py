"""Tests for data/image_files.py — shared image discovery and pairing helpers."""

from pathlib import Path

import cv2
import numpy as np

from sr_engine.data.image_files import (
    SUPPORTED_IMAGE_EXTS,
    is_supported_image,
    list_images,
    lr_scale_from_suffix,
    natural_key,
    pair_hr_lr,
)


def _write(path: Path, w: int = 16, h: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), 128, dtype=np.uint8))


class TestIsSupportedImage:
    def test_supported_extensions(self, tmp_path):
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            assert is_supported_image(tmp_path / f"f{ext}")
        # extension matching is case-insensitive
        assert is_supported_image(tmp_path / "f.PNG")

    def test_unsupported_extensions(self, tmp_path):
        assert not is_supported_image(tmp_path / "f.txt")
        assert not is_supported_image(tmp_path / "f.gif")
        assert not is_supported_image(tmp_path / "f")


class TestNaturalKey:
    def test_digit_aware_ordering(self):
        names = ["frame_10.png", "frame_2.png", "frame_1.png"]
        sorted_names = sorted(names, key=lambda n: natural_key(Path(n)))
        assert sorted_names == ["frame_1.png", "frame_2.png", "frame_10.png"]

    def test_case_insensitive(self):
        assert natural_key(Path("B.png")) > natural_key(Path("a.png"))


class TestListImages:
    def test_returns_only_supported_images_naturally_sorted(self, tmp_path):
        d = tmp_path / "HR"
        for name in ("b.png", "a.jpg", "a.png", "10.png", "2.png", "note.txt", "sub.png"):
            if name == "note.txt":
                (d / name).write_text("x")
            else:
                _write(d / name)
        (d / "sub").mkdir()

        files = [p.name for p in list_images(d)]
        # Digit-prefixed names sort first (matches the frontend's padded-digit sortKey)
        assert files == ["2.png", "10.png", "a.jpg", "a.png", "b.png", "sub.png"]

    def test_missing_dir_returns_empty(self, tmp_path):
        assert list_images(tmp_path / "nope") == []


class TestPairHrLr:
    def test_exact_name_match(self, tmp_path):
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        for name in ("a.png", "b.jpg"):
            _write(hr_dir / name)
            _write(lr_dir / name)

        pairs = pair_hr_lr(hr_dir, lr_dir)
        assert [(p[0].name, p[1].name) for p in pairs] == [("a.png", "a.png"), ("b.jpg", "b.jpg")]

    def test_cross_format_stem_match(self, tmp_path):
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        _write(hr_dir / "a.png")
        _write(lr_dir / "a.jpg")  # different extension, unambiguous

        pairs = pair_hr_lr(hr_dir, lr_dir)
        assert len(pairs) == 1
        assert pairs[0][0].name == "a.png"
        assert pairs[0][1].name == "a.jpg"

    def test_ambiguous_stem_match_skipped(self, tmp_path):
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        _write(hr_dir / "a.png")
        _write(lr_dir / "a.jpg")
        _write(lr_dir / "a.webp")  # two candidates for stem "a" -> ambiguous

        assert pair_hr_lr(hr_dir, lr_dir) == []

    def test_unmatched_files_dropped(self, tmp_path):
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        _write(hr_dir / "a.png")
        _write(hr_dir / "only_hr.png")
        _write(lr_dir / "a.png")
        _write(lr_dir / "only_lr.png")

        pairs = pair_hr_lr(hr_dir, lr_dir)
        assert [p[0].name for p in pairs] == ["a.png"]

    def test_missing_dirs_return_empty(self, tmp_path):
        assert pair_hr_lr(tmp_path / "HR", tmp_path / "LR") == []

    def test_pairs_naturally_sorted(self, tmp_path):
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        for name in ("frame_10.png", "frame_2.png", "frame_1.png"):
            _write(hr_dir / name)
            _write(lr_dir / name)

        pairs = pair_hr_lr(hr_dir, lr_dir)
        assert [p[0].name for p in pairs] == ["frame_1.png", "frame_2.png", "frame_10.png"]


class TestLrScaleFromSuffix:
    """Tests for ``lr_scale_from_suffix``."""

    def test_x_suffix(self):
        assert lr_scale_from_suffix(Path("000001x2.png")) == 2
        assert lr_scale_from_suffix(Path("000001x4.png")) == 4
        assert lr_scale_from_suffix(Path("000001x8.png")) == 8

    def test_underscore_x_suffix(self):
        assert lr_scale_from_suffix(Path("000001_x2.png")) == 2

    def test_no_suffix(self):
        assert lr_scale_from_suffix(Path("000001.png")) is None
        assert lr_scale_from_suffix(Path("000001_LR.png")) is None


class TestPairHrLrSuffixMatching:
    """Pass 3 — suffix-tolerant matching (Flickr2K-style naming)."""

    def test_hr_plain_lr_x_scale_suffix(self, tmp_path):
        """HR/000001.png <-> LR/000001x2.png pairs correctly."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        for i in ("000001", "000002", "000003"):
            _write(hr_dir / f"{i}.png")
            _write(lr_dir / f"{i}x2.png")

        pairs = pair_hr_lr(hr_dir, lr_dir)
        assert len(pairs) == 3
        assert pairs[0][0].name == "000001.png"
        assert pairs[0][1].name == "000001x2.png"

    def test_lr_side_marker_suffix(self, tmp_path):
        """Bare _LR LR files do NOT pair with plain HR names (safety)."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        _write(hr_dir / "000001.png")
        _write(lr_dir / "000001_LR.png")

        assert pair_hr_lr(hr_dir, lr_dir) == []

    def test_both_side_markers(self, tmp_path):
        """HR/000001_HR.png <-> LR/000001_LR.png pairs correctly."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        _write(hr_dir / "000001_HR.png")
        _write(lr_dir / "000001_LR.png")
        _write(hr_dir / "000002_HR.png")
        _write(lr_dir / "000002_lr.png")

        pairs = pair_hr_lr(hr_dir, lr_dir)
        assert len(pairs) == 2
        assert pairs[0][0].name == "000001_HR.png"
        assert pairs[0][1].name == "000001_LR.png"

    def test_mixed_exact_and_suffix(self, tmp_path):
        """Exact-name pairs and suffixed pairs both work in one dataset."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        # exact-name pair
        _write(hr_dir / "a.png")
        _write(lr_dir / "a.png")
        # Flickr2K-style pair
        _write(hr_dir / "000001.png")
        _write(lr_dir / "000001x2.png")
        # both-side marker pair
        _write(hr_dir / "b_HR.png")
        _write(lr_dir / "b_LR.png")

        pairs = pair_hr_lr(hr_dir, lr_dir)
        assert len(pairs) == 3
        by_hr = {p[0].name: p[1].name for p in pairs}
        assert by_hr["a.png"] == "a.png"
        assert by_hr["000001.png"] == "000001x2.png"
        assert by_hr["b_HR.png"] == "b_LR.png"

    def test_ambiguous_suffix_skipped(self, tmp_path):
        """Both 000001x2.png and 000001x4.png -> no pair, never guessed."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        _write(hr_dir / "000001.png")
        _write(lr_dir / "000001x2.png")
        _write(lr_dir / "000001x4.png")

        assert pair_hr_lr(hr_dir, lr_dir) == []

    def test_exact_name_still_wins(self, tmp_path):
        """Pass 1 (exact name) takes precedence over suffix matching."""
        hr_dir = tmp_path / "HR"
        lr_dir = tmp_path / "LR"
        _write(hr_dir / "000001.png")
        _write(lr_dir / "000001.png")
        _write(lr_dir / "000001x2.png")

        pairs = pair_hr_lr(hr_dir, lr_dir)
        assert len(pairs) == 1
        assert pairs[0][1].name == "000001.png"
