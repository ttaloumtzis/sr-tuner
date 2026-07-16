"""Tests for image I/O operations."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from sr_engine.utils.io import read_image, write_image


class TestReadImage:
    """Tests for ``read_image``."""

    def test_read_rgb(self, tmp_path):
        """A valid PNG should be read as an RGB numpy array."""
        img_path = tmp_path / "test.png"
        ref = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))
        result = read_image(img_path)
        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64, 3)

    def test_read_missing_raises(self):
        """A missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_image(Path("/nonexistent/image.png"))

    def test_read_grayscale_converts(self, tmp_path):
        """A grayscale image should be returned as 3-channel RGB."""
        img_path = tmp_path / "gray.png"
        gray = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        cv2.imwrite(str(img_path), gray)
        result = read_image(img_path)
        assert result.shape[2] == 3


class TestWriteImage:
    """Tests for ``write_image``."""

    def test_write_png(self, tmp_path):
        """A numpy array should be written as a PNG file."""
        img_path = tmp_path / "out.png"
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        write_image(img, img_path)
        assert img_path.is_file()

    def test_written_file_is_valid(self, tmp_path):
        """The written PNG should be readable."""
        img_path = tmp_path / "valid.png"
        img = np.ones((16, 16, 3), dtype=np.uint8) * 128
        write_image(img, img_path)
        loaded = cv2.imread(str(img_path))
        assert loaded is not None

    def test_creates_parent_dirs(self, tmp_path):
        """Parent directories should be created if missing."""
        img_path = tmp_path / "sub" / "nested" / "out.png"
        img = np.ones((16, 16, 3), dtype=np.uint8)
        write_image(img, img_path)
        assert img_path.is_file()

    def test_clips_values(self, tmp_path):
        """Out-of-range float values should be clipped."""
        img_path = tmp_path / "clipped.png"
        img = np.array([[[-0.5, 1.5, 0.0]]], dtype=np.float32)
        write_image(img, img_path)
        loaded = cv2.imread(str(img_path))
        assert loaded is not None

    def test_grayscale_single_channel(self, tmp_path):
        """A 2D grayscale array should be written correctly."""
        img_path = tmp_path / "gray_out.png"
        img = np.ones((16, 16), dtype=np.uint8) * 128
        write_image(img, img_path)
        assert img_path.is_file()
