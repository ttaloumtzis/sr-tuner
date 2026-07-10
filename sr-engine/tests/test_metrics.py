"""Tests for evaluation metrics: PSNR, SSIM, LPIPS."""

import pytest
import torch

from sr_engine.engine.metrics import psnr, ssim


class TestPSNR:
    def test_identical_images_returns_large_finite_value(self):
        img = torch.rand(3, 64, 64)
        val = psnr(img, img)
        assert val.isfinite(), "PSNR on identical images should be finite (clamped)"
        assert val.item() > 50.0, "PSNR on identical images should be > 50 dB"

    def test_different_images_returns_reasonable_value(self):
        img1 = torch.zeros(3, 64, 64)
        img2 = torch.ones(3, 64, 64) * 0.5
        val = psnr(img1, img2)
        assert val.isfinite()
        assert 0.0 < val.item() < 50.0

    def test_batch_input(self):
        batch = torch.rand(4, 3, 32, 32)
        val = psnr(batch, batch)
        assert val.isfinite()

    def test_shape_mismatch_raises(self):
        import pytest
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 32, 32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            psnr(img1, img2)


class TestSSIM:
    def test_identical_images_returns_one(self):
        img = torch.rand(3, 64, 64)
        val = ssim(img, img)
        assert (val - 1.0).abs() < 1e-4, "SSIM on identical images should be ~1.0"

    def test_different_images_returns_less_than_one(self):
        img1 = torch.zeros(3, 64, 64)
        img2 = torch.ones(3, 64, 64)
        val = ssim(img1, img2)
        assert val.item() < 1.0

    def test_batch_input(self):
        batch = torch.rand(4, 3, 32, 32)
        val = ssim(batch, batch)
        assert (val - 1.0).abs() < 1e-4

    def test_shape_mismatch_raises(self):
        import pytest
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 32, 32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            ssim(img1, img2)


class TestLPIPS:
    def test_lpips_import_available(self):
        pytest.importorskip("lpips", reason="lpips package not installed")
        from sr_engine.engine.metrics import lpips
        img1 = torch.rand(1, 3, 64, 64)
        img2 = torch.rand(1, 3, 64, 64)
        val = lpips(img1, img2, device="cpu")
        assert val.isfinite()
        assert val.item() >= 0
