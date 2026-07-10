"""Tests for data/degrade.py — degradation pipeline functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from sr_engine.data.degrade import (
    _add_gaussian_noise,
    _add_poisson_noise,
    _apply_jpeg_compression,
    _degrade_image,
    batch_degrade,
    _init_worker,
)


class TestInitWorker:
    def test_sets_cv_threads(self):
        original = cv2.getNumThreads()
        _init_worker()
        assert cv2.getNumThreads() == 1
        cv2.setNumThreads(original)


class TestAddGaussianNoise:
    def test_shape_preserved(self):
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        result = _add_gaussian_noise(img, sigma_range=[10, 10])
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_output_in_range(self):
        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        result = _add_gaussian_noise(img, sigma_range=[30, 30])
        assert result.min() >= 0
        assert result.max() <= 255

    def test_zero_sigma_no_change(self):
        img = np.ones((16, 16, 3), dtype=np.uint8) * 128
        result = _add_gaussian_noise(img, sigma_range=[0, 0])
        np.testing.assert_array_equal(result, img)


class TestAddPoissonNoise:
    def test_shape_preserved(self):
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        result = _add_poisson_noise(img, scale_range=[0.5, 0.5])
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_output_in_range(self):
        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        result = _add_poisson_noise(img, scale_range=[0.5, 0.5])
        assert result.min() >= 0
        assert result.max() <= 255


class TestApplyJpegCompression:
    def test_shape_preserved(self):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = _apply_jpeg_compression(img, quality_range=[90, 90])
        assert result.shape == img.shape

    def test_lower_quality_more_artifacts(self):
        img = np.ones((64, 64, 3), dtype=np.uint8) * 200
        high_q = _apply_jpeg_compression(img.copy(), quality_range=[95, 95])
        low_q = _apply_jpeg_compression(img.copy(), quality_range=[10, 10])
        high_diff = np.abs(high_q.astype(np.float32) - img).mean()
        low_diff = np.abs(low_q.astype(np.float32) - img).mean()
        assert low_diff >= high_diff


class TestDegradeImage:
    @pytest.fixture
    def hr_image(self):
        return np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    def test_no_degradation(self, hr_image):
        result = _degrade_image(hr_image, scale=4)
        expected_h = 128 - (128 % 4)
        expected_w = 128 - (128 % 4)
        assert result.shape == (expected_h // 4, expected_w // 4, 3)

    def test_scale_2(self, hr_image):
        result = _degrade_image(hr_image, scale=2)
        expected_h = (128 - (128 % 2)) // 2
        expected_w = (128 - (128 % 2)) // 2
        assert result.shape[0] == expected_h
        assert result.shape[1] == expected_w

    def test_blur_applied(self):
        hr = np.zeros((64, 64, 3), dtype=np.uint8)
        hr[24:40, 24:40] = 255
        blur_kwargs = {"kernel_size": 15, "sigma": [20.0, 20.0], "prob": 1.0}
        result = _degrade_image(hr, scale=2, blur_kwargs=blur_kwargs)
        assert result is not None

    def test_resize_methods(self, hr_image):
        for method in ("bilinear", "bicubic", "lanczos"):
            result = _degrade_image(hr_image, scale=4, resize_method=method)
            assert result.shape == (32, 32, 3)

    def test_noise_kwargs_read_from_nested_structure(self, hr_image):
        noise_kwargs = {
            "gaussian": {"sigma_range": [5, 5], "prob": 1.0},
            "poisson": {"scale_range": [0.5, 0.5], "prob": 0.0},
        }
        result = _degrade_image(hr_image, scale=4, noise_kwargs=noise_kwargs)
        assert result is not None
        assert not np.array_equal(result, hr_image[:128, :128][::4, ::4])

    def test_poisson_noise_from_config(self, hr_image):
        noise_kwargs = {
            "gaussian": {"sigma_range": [1, 1], "prob": 0.0},
            "poisson": {"scale_range": [0.5, 0.5], "prob": 1.0},
        }
        result = _degrade_image(hr_image, scale=4, noise_kwargs=noise_kwargs)
        assert result is not None

    def test_jpeg_quality_range(self, hr_image):
        jpeg_kwargs = {"quality_range": [85, 85], "prob": 1.0}
        result = _degrade_image(hr_image, scale=4, jpeg_kwargs=jpeg_kwargs)
        assert result is not None


class TestBatchDegrade:
    def test_empty_hr_paths(self, tmp_path):
        result = batch_degrade([], tmp_path / "lr", scale=4, config={})
        assert result == []
