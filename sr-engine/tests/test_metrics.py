"""Tests for engine/metrics.py — PSNR, SSIM, LPIPS."""

import torch
import pytest

from sr_engine.engine.metrics import psnr, ssim


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
