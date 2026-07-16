"""Tests for models/losses.py — loss function implementations."""

import torch
import pytest

from sr_engine.models.losses import L1Loss, PerceptualLoss, GANLoss


class TestL1Loss:
    """Tests for Charbonnier L1 loss."""

    def test_identical(self):
        """Loss should be near-zero for identical tensors (Charbonnier eps)."""
        x = torch.tensor([1.0, 2.0, 3.0])
        loss = L1Loss()(x, x)
        assert loss.item() < 1e-5

    def test_positive_for_diff(self):
        """Loss should be positive for differing tensors."""
        pred = torch.ones(4)
        target = torch.zeros(4)
        loss = L1Loss()(pred, target)
        assert loss.item() > 0.0

    def test_batched(self):
        """Batched inputs should work."""
        pred = torch.randn(2, 3, 16, 16)
        target = torch.randn(2, 3, 16, 16)
        loss = L1Loss()(pred, target)
        assert loss.ndim == 0


class TestPerceptualLoss:
    """Tests for perceptual / VGG-based loss."""

    def test_forward_returns_scalar(self):
        """Forward pass should return a scalar tensor."""
        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 64, 64)
        loss = PerceptualLoss()
        out = loss(pred, target)
        assert out.ndim == 0

    def test_zero_for_identical(self):
        """Loss should be near zero for identical inputs."""
        x = torch.rand(1, 3, 64, 64)
        loss = PerceptualLoss()
        out = loss(x, x)
        assert out.item() < 1e-4


class TestGANLoss:
    """Tests for GAN loss (vanilla BCE and LSGAN)."""

    def test_vanilla_real(self):
        """Vanilla loss for real samples should be a scalar."""
        pred = torch.randn(4)
        loss = GANLoss(gan_type="vanilla")(pred, target_is_real=True)
        assert loss.ndim == 0

    def test_vanilla_fake(self):
        """Vanilla loss for fake samples should be a scalar."""
        pred = torch.randn(4)
        loss = GANLoss(gan_type="vanilla")(pred, target_is_real=False)
        assert loss.ndim == 0

    def test_lsgan_real(self):
        """LSGAN loss for real samples should be scalar."""
        pred = torch.randn(4)
        loss = GANLoss(gan_type="lsgan")(pred, target_is_real=True)
        assert loss.ndim == 0

    def test_lsgan_fake(self):
        """LSGAN loss for fake samples should be scalar."""
        pred = torch.randn(4)
        loss = GANLoss(gan_type="lsgan")(pred, target_is_real=False)
        assert loss.ndim == 0

    def test_unknown_gan_type_raises(self):
        """An unknown gan_type should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            GANLoss(gan_type="wgan")
