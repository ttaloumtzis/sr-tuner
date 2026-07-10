"""Tests for models/losses.py — L1Loss, PerceptualLoss, GANLoss."""

import pytest

from sr_engine.models.losses import L1Loss, PerceptualLoss, GANLoss


class TestL1Loss:
    def test_basic_forward(self):
        import torch
        loss_fn = L1Loss()
        pred = torch.randn(2, 3, 16, 16)
        target = torch.randn(2, 3, 16, 16)
        loss = loss_fn(pred, target)
        assert loss.ndim == 0
        assert loss > 0

    def test_identical_inputs(self):
        import torch
        loss_fn = L1Loss()
        x = torch.randn(2, 3, 16, 16)
        loss = loss_fn(x, x)
        assert loss.item() < 1e-4


class TestPerceptualLoss:
    def test_raises_on_unknown_layer(self):
        with pytest.raises(ValueError, match="Unknown VGG19 layer"):
            PerceptualLoss(layer_ids=["nonexistent"])

    def test_forward_shape(self):
        import torch
        loss_fn = PerceptualLoss(layer_ids=["relu5_4"])
        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 64, 64)
        loss = loss_fn(pred, target)
        assert loss.ndim == 0
        assert loss > 0


class TestGANLoss:
    @pytest.fixture
    def pred(self):
        import torch
        return torch.randn(4, 1)

    def test_vanilla_real(self, pred):
        loss_fn = GANLoss(gan_type="vanilla")
        loss = loss_fn(pred, target_is_real=True)
        assert loss.ndim == 0
        assert loss > 0

    def test_vanilla_fake(self, pred):
        loss_fn = GANLoss(gan_type="vanilla")
        loss = loss_fn(pred, target_is_real=False)
        assert loss.ndim == 0
        assert loss > 0

    def test_lsgan_real(self, pred):
        loss_fn = GANLoss(gan_type="lsgan")
        loss = loss_fn(pred, target_is_real=True)
        assert loss.ndim == 0
        assert loss > 0

    def test_lsgan_fake(self, pred):
        loss_fn = GANLoss(gan_type="lsgan")
        loss = loss_fn(pred, target_is_real=False)
        assert loss.ndim == 0
        assert loss > 0

    def test_raises_on_unknown_type(self):
        with pytest.raises(ValueError, match="Unsupported"):
            GANLoss(gan_type="wgan")
