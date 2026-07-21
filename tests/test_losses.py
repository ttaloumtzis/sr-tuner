"""Tests for models/losses.py — loss function implementations."""

import torch
import pytest

from sr_engine.models.losses import (
    L1Loss, L2Loss, PerceptualLoss, GANLoss,
    EdgeLoss, FrequencyLoss, StyleLoss, SSIMLoss,
    VGGFeatureExtractor,
    build_composite_loss, _migrate_legacy_loss_config, _default_loss_config,
)


class TestL1Loss:
    """Tests for Charbonnier L1 loss."""

    def test_identical(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        loss = L1Loss()(x, x)
        assert loss.item() < 1e-5

    def test_positive_for_diff(self):
        pred = torch.ones(4)
        target = torch.zeros(4)
        loss = L1Loss()(pred, target)
        assert loss.item() > 0.0

    def test_batched(self):
        pred = torch.randn(2, 3, 16, 16)
        target = torch.randn(2, 3, 16, 16)
        loss = L1Loss()(pred, target)
        assert loss.ndim == 0


class TestL2Loss:
    """Tests for MSE loss."""

    def test_identical(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        loss = L2Loss()(x, x)
        assert loss.item() < 1e-5

    def test_mse_value(self):
        pred = torch.tensor([2.0, 3.0, 4.0])
        target = torch.tensor([0.0, 0.0, 0.0])
        loss = L2Loss()(pred, target)
        expected = (4.0 + 9.0 + 16.0) / 3.0
        assert abs(loss.item() - expected) < 1e-5

    def test_batched(self):
        pred = torch.randn(2, 3, 16, 16)
        target = torch.randn(2, 3, 16, 16)
        loss = L2Loss()(pred, target)
        assert loss.ndim == 0


class TestEdgeLoss:
    """Tests for Sobel gradient-magnitude edge loss."""

    def test_identical(self):
        x = torch.rand(2, 3, 16, 16)
        loss = EdgeLoss()(x, x)
        assert loss.item() < 1e-5

    def test_different_positive(self):
        pred = torch.rand(2, 3, 16, 16)
        target = torch.rand(2, 3, 16, 16)
        loss = EdgeLoss()(pred, target)
        assert loss.item() > 0.0

    def test_output_shape(self):
        pred = torch.randn(2, 3, 16, 16)
        target = torch.randn(2, 3, 16, 16)
        loss = EdgeLoss()(pred, target)
        assert loss.ndim == 0


class TestFrequencyLoss:
    """Tests for FFT-based frequency loss."""

    def test_identical(self):
        x = torch.rand(2, 3, 16, 16)
        loss = FrequencyLoss()(x, x)
        assert loss.item() < 1e-5

    def test_different_positive(self):
        pred = torch.rand(2, 3, 16, 16)
        target = torch.rand(2, 3, 16, 16)
        loss = FrequencyLoss()(pred, target)
        assert loss.item() > 0.0

    def test_output_shape(self):
        pred = torch.randn(2, 3, 16, 16)
        target = torch.randn(2, 3, 16, 16)
        loss = FrequencyLoss()(pred, target)
        assert loss.ndim == 0


class TestStyleLoss:
    """Tests for Gram-matrix style loss."""

    def test_identical(self):
        x = torch.rand(1, 3, 64, 64)
        loss = StyleLoss(layers=["relu1_1"])
        out = loss(x, x)
        assert out.item() < 1e-4

    def test_different_positive(self):
        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 64, 64)
        loss = StyleLoss(layers=["relu1_1"])
        out = loss(pred, target)
        assert out.item() > 0.0


class TestSSIMLoss:
    """Tests for SSIM-based structural dissimilarity loss."""

    def test_identical(self):
        x = torch.rand(2, 3, 32, 32)
        loss = SSIMLoss()(x, x)
        assert loss.item() < 0.01

    def test_range(self):
        pred = torch.rand(2, 3, 32, 32)
        target = torch.zeros(2, 3, 32, 32)
        loss = SSIMLoss()(pred, target)
        assert 0.0 <= loss.item() <= 1.0

    def test_different_positive(self):
        pred = torch.ones(2, 3, 32, 32)
        target = torch.zeros(2, 3, 32, 32)
        loss = SSIMLoss()(pred, target)
        assert loss.item() > 0.01

    def test_constant_low(self):
        x = torch.ones(2, 3, 32, 32)
        loss = SSIMLoss()(x, x)
        assert loss.item() < 0.01


class TestPerceptualLoss:
    """Tests for perceptual / VGG-based loss."""

    def test_forward_returns_scalar(self):
        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 64, 64)
        loss = PerceptualLoss()
        out = loss(pred, target)
        assert out.ndim == 0

    def test_zero_for_identical(self):
        x = torch.rand(1, 3, 64, 64)
        loss = PerceptualLoss()
        out = loss(x, x)
        assert out.item() < 1e-4


class TestGANLoss:
    """Tests for GAN loss (vanilla BCE and LSGAN)."""

    def test_vanilla_real(self):
        pred = torch.randn(4)
        loss = GANLoss(gan_type="vanilla")(pred, target_is_real=True)
        assert loss.ndim == 0

    def test_vanilla_fake(self):
        pred = torch.randn(4)
        loss = GANLoss(gan_type="vanilla")(pred, target_is_real=False)
        assert loss.ndim == 0

    def test_lsgan_real(self):
        pred = torch.randn(4)
        loss = GANLoss(gan_type="lsgan")(pred, target_is_real=True)
        assert loss.ndim == 0

    def test_lsgan_fake(self):
        pred = torch.randn(4)
        loss = GANLoss(gan_type="lsgan")(pred, target_is_real=False)
        assert loss.ndim == 0

    def test_unknown_gan_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            GANLoss(gan_type="wgan")


class TestMigrateLegacyConfig:
    """Tests for ``_migrate_legacy_loss_config`` backward compat."""

    def test_empty_returns_default(self):
        result = _migrate_legacy_loss_config({})
        assert result["pixel"]["type"] == "l1"
        assert result["pixel"]["weight"] == 1.0
        assert result["perceptual"]["type"] == "vgg"

    def test_legacy_zero_weight(self):
        result = _migrate_legacy_loss_config({"perceptual_weight": 0.0})
        assert "pixel" in result
        assert "perceptual" not in result

    def test_legacy_positive_weight(self):
        result = _migrate_legacy_loss_config({"perceptual_weight": 0.1})
        assert result["perceptual"]["type"] == "vgg"
        assert result["perceptual"]["weight"] == 0.1
        assert "pixel" in result

    def test_legacy_with_layers(self):
        result = _migrate_legacy_loss_config({
            "perceptual_weight": 0.2,
            "perceptual_layers": ["relu3_4", "relu4_4"],
        })
        assert result["perceptual"]["layers"] == ["relu3_4", "relu4_4"]

    def test_new_format_passthrough(self):
        config = {
            "pixel": {"type": "l1", "weight": 1.0},
            "edge": {"type": "edge", "weight": 0.05},
        }
        result = _migrate_legacy_loss_config(config)
        assert result == config

    def test_new_format_with_legacy(self):
        result = _migrate_legacy_loss_config({
            "pixel": {"type": "l2", "weight": 1.0},
            "perceptual_weight": 0.1,
        })
        assert result["pixel"]["type"] == "l2"
        assert result["perceptual"]["type"] == "vgg"
        assert result["perceptual"]["weight"] == 0.1

    def test_none_returns_default(self):
        result = _migrate_legacy_loss_config(None)
        assert result["pixel"]["type"] == "l1"

    def test_default_config_structure(self):
        cfg = _default_loss_config()
        assert "pixel" in cfg
        assert "perceptual" in cfg
        assert cfg["pixel"]["type"] == "l1"
        assert cfg["perceptual"]["type"] == "vgg"
        assert cfg["perceptual"]["weight"] == 0.1


class TestBuildCompositeLoss:
    """Tests for ``build_composite_loss`` and ``CompositeLoss``."""

    def test_default_config(self):
        loss_fn = build_composite_loss(None, torch.device("cpu"))
        pred = torch.rand(2, 3, 16, 16)
        target = torch.rand(2, 3, 16, 16)
        total, components = loss_fn(pred, target)
        assert total.ndim == 0
        assert isinstance(components, dict)
        assert "pixel" in components
        assert total.item() > 0

    def test_custom_loss_config(self):
        config = {
            "pixel": {"type": "l2", "weight": 1.0},
            "edge": {"type": "edge", "weight": 0.1},
        }
        loss_fn = build_composite_loss(config, torch.device("cpu"))
        pred = torch.rand(2, 3, 16, 16)
        target = torch.rand(2, 3, 16, 16)
        total, components = loss_fn(pred, target)
        assert "pixel" in components
        assert "edge" in components

    def test_missing_pixel_raises(self):
        with pytest.raises(ValueError, match="pixel loss"):
            build_composite_loss(
                {"edge": {"type": "edge", "weight": 1.0}},
                torch.device("cpu"),
            )

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown loss type"):
            build_composite_loss(
                {"pixel": {"type": "l1", "weight": 1.0},
                 "bad": {"type": "bogus", "weight": 1.0}},
                torch.device("cpu"),
            )

    def test_legacy_config_auto_migrate(self):
        loss_fn = build_composite_loss(
            {"perceptual_weight": 0.0}, torch.device("cpu"),
        )
        pred = torch.rand(2, 3, 16, 16)
        target = torch.rand(2, 3, 16, 16)
        total, components = loss_fn(pred, target)
        assert "pixel" in components
        assert total.item() > 0

    def test_gradient_flow(self):
        loss_fn = build_composite_loss(
            {"pixel": {"type": "l1", "weight": 1.0}, "edge": {"type": "edge", "weight": 0.05}},
            torch.device("cpu"),
        )
        pred = torch.rand(2, 3, 16, 16, requires_grad=True)
        target = torch.rand(2, 3, 16, 16)
        total, _ = loss_fn(pred, target)
        total.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum().item() > 0

    def test_loss_names_property(self):
        config = {"pixel": {"type": "l1", "weight": 1.0}, "ssim": {"type": "ssim", "weight": 0.5}}
        loss_fn = build_composite_loss(config, torch.device("cpu"))
        assert "pixel" in loss_fn.loss_names
        assert "ssim" in loss_fn.loss_names

    def test_vgg_without_explicit_layers_uses_defaults(self):
        """VGG loss with no 'layers' key should use default layers (relu5_4)."""
        config = {
            "pixel": {"type": "l1", "weight": 1.0},
            "perceptual": {"type": "vgg", "weight": 0.1},
        }
        loss_fn = build_composite_loss(config, torch.device("cpu"))
        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 64, 64)
        total, components = loss_fn(pred, target)
        assert "perceptual" in components

    def test_vgg_default_and_style_explicit_shared(self):
        """Perceptual (default relu5_4) + style (explicit layers) should share VGG."""
        config = {
            "pixel": {"type": "l1", "weight": 1.0},
            "perceptual": {"type": "vgg", "weight": 0.1},
            "style": {
                "type": "style", "weight": 0.01,
                "layers": ["relu1_1", "relu1_2", "relu2_1", "relu2_2"],
            },
        }
        loss_fn = build_composite_loss(config, torch.device("cpu"))
        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 64, 64)
        total, components = loss_fn(pred, target)
        assert "perceptual" in components
        assert "style" in components


class TestVGGFeatureExtractor:
    """Tests for the shared VGG backbone."""

    def test_forward_returns_dict(self):
        extractor = VGGFeatureExtractor(["relu1_1", "relu5_4"])
        x = torch.rand(1, 3, 64, 64)
        out = extractor(x)
        assert isinstance(out, dict)
        assert "relu1_1" in out
        assert "relu5_4" in out

    def test_unknown_layer_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            VGGFeatureExtractor(["bogus"])

    def test_shared_between_perceptual_and_style(self):
        layers = ["relu1_1", "relu3_4"]
        vgg = VGGFeatureExtractor(layers)
        p = PerceptualLoss(layers=["relu1_1", "relu3_4"], vgg_extractor=vgg)
        s = StyleLoss(layers=["relu1_1", "relu3_4"], vgg_extractor=vgg)
        x = torch.rand(1, 3, 64, 64)
        px = p(x, x)
        sx = s(x, x)
        assert px.item() < 1e-4
        assert sx.item() < 1e-4
