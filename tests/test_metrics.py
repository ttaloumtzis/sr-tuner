"""Tests for engine/metrics.py — PSNR, SSIM, LPIPS."""

import torch
import pytest

from sr_engine.engine.metrics import psnr, ssim, ms_ssim


class TestPSNR:
    """Tests for ``psnr``."""

    def test_identical(self):
        """PSNR should be very large for identical images."""
        img = torch.rand(3, 64, 64)
        value = psnr(img, img)
        assert value > 50.0

    def test_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        a = torch.rand(3, 64, 64)
        b = torch.rand(3, 32, 32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            psnr(a, b)

    def test_batched(self):
        """Batched inputs should return a scalar."""
        a = torch.rand(2, 3, 64, 64)
        b = torch.rand(2, 3, 64, 64)
        value = psnr(a, b)
        assert value.ndim == 0

    def test_lower_for_noisy(self):
        """A noisy image should have lower PSNR."""
        clean = torch.ones(3, 16, 16) * 0.5
        noisy = clean + torch.randn_like(clean) * 0.1
        assert psnr(clean, noisy) < psnr(clean, clean)


class TestSSIM:
    """Tests for ``ssim``."""

    def test_identical(self):
        """SSIM should be close to 1 for identical images."""
        img = torch.rand(3, 16, 16)
        value = ssim(img, img)
        assert value > 0.99

    def test_positive(self):
        """SSIM should be positive for similar images."""
        a = torch.rand(3, 16, 16)
        b = a + torch.randn_like(a) * 0.05
        value = ssim(a, b)
        assert value > 0.0


class TestMSSSIM:
    """Tests for ``ms_ssim``."""

    def test_identical(self):
        """MS-SSIM should be ~1 for identical images."""
        img = torch.rand(3, 64, 64)
        value = ms_ssim(img, img)
        assert value > 0.99

    def test_positive_for_similar(self):
        """MS-SSIM should be positive for similar images."""
        a = torch.rand(3, 64, 64)
        b = a + torch.randn_like(a) * 0.05
        value = ms_ssim(a, b)
        assert value > 0.0

    def test_in_range(self):
        """MS-SSIM should stay within [0, 1] for arbitrary pairs."""
        a = torch.rand(3, 48, 48)
        b = torch.rand(3, 48, 48)
        value = ms_ssim(a, b)
        assert 0.0 <= value <= 1.0

    def test_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        a = torch.rand(3, 64, 64)
        b = torch.rand(3, 32, 32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            ms_ssim(a, b)

    def test_tiny_image_degrades_to_single_scale(self):
        """Small images should not crash (levels are capped automatically)."""
        a = torch.rand(3, 16, 16)
        b = a + torch.randn_like(a) * 0.05
        value = ms_ssim(a, b)
        assert value > 0.0
