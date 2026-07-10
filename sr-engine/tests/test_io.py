"""Tests for utils/io.py — image read/write, directory utilities."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from sr_engine.utils.io import read_image, write_image, ensure_dir


class TestReadImage:
    def test_reads_rgb_image(self, sample_image):
        img = read_image(sample_image)
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.dtype == np.uint8

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError, match="Could not read"):
            read_image(Path("/nonexistent.png"))

    def test_raises_on_corrupt_file(self, corrupt_image):
        with pytest.raises(FileNotFoundError, match="Could not read"):
            read_image(corrupt_image)

    def test_grayscale_still_reads_as_rgb(self):
        import tempfile
        gray = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        path = Path(tempfile.mktemp(suffix=".png"))
        cv2.imwrite(str(path), gray)
        img = read_image(path)
        assert img.ndim == 3
        assert img.shape[2] == 3
        path.unlink()


class TestWriteImage:
    def test_writes_rgb(self, tmp_path):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        path = tmp_path / "out.png"
        write_image(img, path)
        assert path.exists()
        reloaded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert reloaded is not None

    def test_writes_grayscale(self, tmp_path):
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        path = tmp_path / "gray.png"
        write_image(img, path)
        assert path.exists()

    def test_raises_on_bad_shape(self, tmp_path):
        img = np.random.randint(0, 256, (64, 64, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            write_image(img, tmp_path / "bad.png")

    def test_creates_parent_dir(self, tmp_path):
        img = np.ones((16, 16, 3), dtype=np.uint8) * 128
        path = tmp_path / "sub" / "nested" / "img.png"
        write_image(img, path)
        assert path.exists()


class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        path = tmp_path / "new_dir"
        result = ensure_dir(path)
        assert result == path
        assert path.is_dir()

    def test_returns_existing(self, tmp_path):
        (tmp_path / "existing").mkdir()
        result = ensure_dir(tmp_path / "existing")
        assert result.is_dir()
