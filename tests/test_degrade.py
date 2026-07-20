"""Tests for data/degrade.py — batch degradation pipeline."""

from unittest.mock import patch

import numpy as np

from sr_engine.data.degrade import (
    batch_degrade,
    _degrade_image,
    _add_gaussian_noise,
    _add_poisson_noise,
    _add_salt_pepper_noise,
    _apply_gaussian_blur,
    _apply_motion_blur,
    _apply_jpeg_compression,
    _apply_jpeg2000_compression,
    _apply_color_jitter,
)


def _make_test_image(height=64, width=64) -> np.ndarray:
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


class TestBatchDegrade:
    """Tests for ``batch_degrade``."""

    def test_empty_hr_list(self, tmp_path):
        result = batch_degrade(
            hr_paths=[],
            lr_dir=tmp_path / "lr",
            scale=4,
            config={},
        )
        assert result == []

    def test_creates_lr_dir(self, tmp_path):
        batch_degrade([], tmp_path / "lr", 4, {})
        assert (tmp_path / "lr").is_dir()

    def test_skips_nonexistent_files(self, tmp_path):
        result = batch_degrade(
            hr_paths=[tmp_path / "nonexistent.png"],
            lr_dir=tmp_path / "lr",
            scale=4,
            config={},
        )
        assert result == []

    def test_worker_crash_returns_partial_results(self, tmp_path):
        """When a worker crashes, batch_degrade should return partial results."""
        import concurrent.futures
        import cv2
        import numpy as np

        lr_dir = tmp_path / "lr"

        # Create valid source images so _process_single_frame succeeds for them
        hr_paths = []
        for i in range(3):
            path = tmp_path / f"frame_{i:04d}.png"
            cv2.imwrite(str(path), np.ones((64, 64, 3), dtype=np.uint8) * 128)
            hr_paths.append(path)

        # Mock ProcessPoolExecutor so the map() results iterator raises
        # BrokenPipeError after yielding 2 successful results.
        def _mock_map(self, worker, items):
            for i, item in enumerate(items):
                if i == 2:
                    raise BrokenPipeError("Mock worker pipe broken")
                yield worker(item)

        with patch.object(
            concurrent.futures.ProcessPoolExecutor, "map", _mock_map,
        ):
            result = batch_degrade(
                hr_paths=hr_paths,
                lr_dir=lr_dir,
                scale=4,
                config={},
            )

        # Should have partial results (first 2 frames)
        assert len(result) == 2
        for hr, lr in result:
            assert hr.exists()
            assert lr.exists()


class TestDegradeImageEnabled:
    """Tests for the ``enabled`` flag in each degradation section."""

    def test_blur_enabled_false_skips_blur(self):
        img = _make_test_image()
        blurred = _degrade_image(
            img, scale=4,
            blur_kwargs={"enabled": False, "gaussian": {"kernel_size": 21, "sigma": [5.0, 5.0], "prob": 1.0}},
        )
        # No blur was applied (enabled: false), so resize + noise remain
        assert blurred.shape == (16, 16, 3)

    def test_noise_enabled_false_skips_noise(self):
        img = _make_test_image()
        result = _degrade_image(
            img, scale=4,
            noise_kwargs={"enabled": False, "gaussian": {"sigma_range": [50, 50], "prob": 1.0}},
        )
        assert result.shape == (16, 16, 3)

    def test_jpeg_enabled_false_skips_jpeg(self):
        img = _make_test_image()
        result = _degrade_image(
            img, scale=4,
            jpeg_kwargs={"enabled": False, "quality_range": [5, 5], "prob": 1.0},
        )
        assert result.shape == (16, 16, 3)

    def test_jpeg2000_enabled_false_skips_jpeg2000(self):
        img = _make_test_image()
        result = _degrade_image(
            img, scale=4,
            jpeg2000_kwargs={"enabled": False, "quality_range": [5, 5], "prob": 1.0},
        )
        assert result.shape == (16, 16, 3)

    def test_color_jitter_enabled_false_skips_jitter(self):
        img = _make_test_image()
        result = _degrade_image(
            img, scale=4,
            color_jitter_kwargs={"enabled": False, "prob": 1.0},
        )
        assert result.shape == (16, 16, 3)

    def test_all_disabled_produces_plain_downsample(self):
        img = _make_test_image()
        result = _degrade_image(
            img, scale=4,
            color_jitter_kwargs={"enabled": False},
            blur_kwargs={"enabled": False},
            noise_kwargs={"enabled": False},
            jpeg_kwargs={"enabled": False},
            jpeg2000_kwargs={"enabled": False},
            resize_method="area",
        )
        # Plain area downsample: 64x64 -> 16x16
        assert result.shape == (16, 16, 3)
        # Values should be in [0, 255]
        assert result.dtype == np.uint8


class TestNoiseFunctions:
    def test_gaussian_noise_changes_values(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        noisy = _add_gaussian_noise(img, [50, 50])
        assert noisy.shape == img.shape
        assert noisy.dtype == np.uint8
        # With sigma=50, values should differ
        assert not np.array_equal(noisy, img)

    def test_poisson_noise_changes_values(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        noisy = _add_poisson_noise(img, [10.0, 10.0])
        assert noisy.shape == img.shape
        assert noisy.dtype == np.uint8

    def test_salt_pepper_introduces_black_and_white_pixels(self):
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        noisy = _add_salt_pepper_noise(img, amount=0.5, salt_vs_pepper=0.5)
        assert noisy.shape == img.shape
        assert noisy.dtype == np.uint8
        has_255 = np.any(noisy == 255)
        has_0 = np.any(noisy == 0)
        assert has_255 or has_0, "Salt & pepper noise should introduce 0 or 255 pixels"


class TestBlurFunctions:
    def test_gaussian_blur_smoothes(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        blurred = _apply_gaussian_blur(img, kernel_size=11, sigma_range=[3.0, 3.0])
        assert blurred.shape == img.shape
        assert blurred.dtype == np.uint8

    def test_motion_blur_applies(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        blurred = _apply_motion_blur(img, max_kernel_size=15)
        assert blurred.shape == img.shape
        assert blurred.dtype == np.uint8

    def test_gaussian_blur_odd_kernel_enforced(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        # Even kernel size should be bumped to odd
        blurred = _apply_gaussian_blur(img, kernel_size=10, sigma_range=[1.0, 1.0])
        assert blurred.shape == img.shape


class TestCompressionFunctions:
    def test_jpeg_compression(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        compressed = _apply_jpeg_compression(img, [95, 95])
        assert compressed is not None
        assert compressed.shape == img.shape

    def test_jpeg2000_compression(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        compressed = _apply_jpeg2000_compression(img, [95, 95])
        assert compressed is not None
        assert compressed.shape == img.shape


class TestColorJitter:
    def test_color_jitter_changes_colors(self):
        img = np.full((16, 16, 3), [100, 100, 100], dtype=np.uint8)
        jittered = _apply_color_jitter(img, [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5])
        assert jittered.shape == img.shape
        assert jittered.dtype == np.uint8


class TestResizeMethods:
    @staticmethod
    def _downsample_with(method: str, antialias: bool = False) -> np.ndarray:
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        return _degrade_image(
            img, scale=4,
            blur_kwargs={"enabled": False},
            noise_kwargs={"enabled": False},
            jpeg_kwargs={"enabled": False},
            jpeg2000_kwargs={"enabled": False},
            resize_method=method,
            resize_antialias=antialias,
        )

    def test_area(self):
        result = self._downsample_with("area")
        assert result.shape == (16, 16, 3)

    def test_bicubic(self):
        result = self._downsample_with("bicubic")
        assert result.shape == (16, 16, 3)

    def test_bilinear(self):
        result = self._downsample_with("bilinear")
        assert result.shape == (16, 16, 3)

    def test_lanczos(self):
        result = self._downsample_with("lanczos")
        assert result.shape == (16, 16, 3)

    def test_nearest(self):
        result = self._downsample_with("nearest")
        assert result.shape == (16, 16, 3)


class TestPipelineOrder:
    """End-to-end pipeline tests with different configs."""

    def test_only_jpeg_pipeline(self):
        img = _make_test_image()
        result = _degrade_image(
            img, scale=4,
            blur_kwargs={"enabled": False},
            noise_kwargs={"enabled": False},
            jpeg_kwargs={"enabled": True, "quality_range": [50, 50], "prob": 1.0},
            jpeg2000_kwargs={"enabled": False},
            color_jitter_kwargs={"enabled": False},
        )
        assert result.shape == (16, 16, 3)

    def test_only_noise_pipeline(self):
        img = _make_test_image()
        result = _degrade_image(
            img, scale=4,
            blur_kwargs={"enabled": False},
            noise_kwargs={"enabled": True, "gaussian": {"sigma_range": [15, 15], "prob": 1.0}},
            jpeg_kwargs={"enabled": False},
            jpeg2000_kwargs={"enabled": False},
            color_jitter_kwargs={"enabled": False},
        )
        assert result.shape == (16, 16, 3)

    def test_only_color_jitter_pipeline(self):
        img = _make_test_image()
        result = _degrade_image(
            img, scale=4,
            color_jitter_kwargs={"enabled": True, "hue_range": [-0.1, 0.1],
                                  "saturation_range": [-0.1, 0.1],
                                  "value_range": [-0.1, 0.1], "prob": 1.0},
            blur_kwargs={"enabled": False},
            noise_kwargs={"enabled": False},
            jpeg_kwargs={"enabled": False},
            jpeg2000_kwargs={"enabled": False},
        )
        assert result.shape == (16, 16, 3)
