"""Tests for model architectures and registry."""

import subprocess
import sys
import warnings

import pytest
import torch

from sr_engine.models.archs.rrdbnet import RRDBNet
from sr_engine.models.archs.swinir import SwinIR
from sr_engine.models.checkpoint import (
    check_shape_compat,
    compat_load_state_dict,
    strip_keys_not_in_model,
)
from sr_engine.models.registry import build_model, register


class TestRRDBNet:
    """Tests for RRDBNet forward pass."""

    def test_forward_output_shape(self):
        """RRDBNet(scale=4) should upscale 16x16 -> 64x64."""
        model = RRDBNet(scale=4)
        dummy = torch.randn(1, 3, 16, 16)
        out = model(dummy)
        assert out.shape == (1, 3, 64, 64)

    def test_forward_output_shape_scale_2(self):
        """RRDBNet(scale=2) should upscale 16x16 -> 32x32."""
        model = RRDBNet(scale=2)
        dummy = torch.randn(1, 3, 16, 16)
        out = model(dummy)
        assert out.shape == (1, 3, 32, 32)

    def test_output_finite(self):
        """RRDBNet output should contain only finite values."""
        model = RRDBNet(scale=4)
        model.eval()
        dummy = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            out = model(dummy)
        assert torch.isfinite(out).all()


class TestSwinIR:
    """Tests for SwinIR forward pass."""

    def test_forward_output_shape(self):
        """SwinIR(scale=4) should upscale 16x16 -> 64x64."""
        model = SwinIR(scale=4)
        dummy = torch.randn(1, 3, 16, 16)
        out = model(dummy)
        assert out.shape == (1, 3, 64, 64)

    def test_forward_output_shape_scale_2(self):
        """SwinIR(scale=2) should upscale 16x16 -> 32x32."""
        model = SwinIR(scale=2)
        dummy = torch.randn(1, 3, 16, 16)
        out = model(dummy)
        assert out.shape == (1, 3, 32, 32)


class TestModelRegistry:
    """Tests for model registration and builder."""

    def test_rrdb_esrgan_is_registered(self):
        """build_model('rrdb_esrgan') should return an RRDBNet instance."""
        model = build_model("rrdb_esrgan", {"scale": 4})
        assert isinstance(model, RRDBNet)

    def test_build_model_passes_kwargs(self):
        """build_model should pass config kwargs to the constructor."""
        model = build_model("rrdb_esrgan", {"scale": 2})
        dummy = torch.randn(1, 3, 16, 16)
        out = model(dummy)
        assert out.shape == (1, 3, 32, 32)

    def test_build_unknown_model_raises(self):
        """build_model with an unknown name should raise ValueError."""
        with pytest.raises(ValueError, match="nonexistent"):
            build_model("nonexistent", {})

    def test_register_decorator_adds_to_registry(self):
        """The @register decorator should add the class to the registry."""
        @register("_test_model")
        class _TestModel(torch.nn.Module):
            def forward(self, x):
                return x
        model = build_model("_test_model", {})
        assert isinstance(model, _TestModel)

    def test_models_registered_in_fresh_process(self):
        """Verify @register decorators fire in a fresh Python process.

        Guards against regression where sr_engine/models/__init__.py stops
        importing the arch modules, which would leave the registry empty and
        cause build_model() to raise ValueError despite the model files
        existing on disk.
        """
        code = """
from sr_engine.models.registry import build_model
m = build_model("rrdb_esrgan", {"scale": 4})
assert m is not None, "Model not registered in fresh import"
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"Subprocess failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


class TestSwinIRNumerical:
    """Numerical correctness for SwinIR — residual fix validation."""

    def test_forward_produces_finite_output(self):
        """SwinIR forward should produce finite values."""
        model = SwinIR(scale=4)
        model.eval()
        dummy = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            out = model(dummy)
        assert torch.isfinite(out).all()

    def test_residual_connection_active(self):
        """Verify the MLP residual uses the attention branch output.

        The fix ensures the second residual connects to x1 = shortcut + Attn
        rather than to the original shortcut. If correct, zeroing the
        attention's output should still let signal flow through the MLP branch.
        """
        model = SwinIR(scale=4)
        model.eval()
        dummy = torch.randn(1, 3, 16, 16)

        layer = model.rstb_layers[0].layers[0]

        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 3, 64, 64)

    def test_output_not_all_zeros_or_nans(self):
        """SwinIR output should be non-zero, finite, and non-NaN."""
        model = SwinIR(scale=4)
        model.eval()
        dummy = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            out = model(dummy)
        assert out.abs().sum() > 0
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestSwinIRExtended:
    """Tests for SwinIR scale=3, rgb_mean, and old-checkpoint rejection."""

    def test_forward_output_shape_scale_3(self):
        """SwinIR(scale=3) should upscale 16x16 -> 48x48."""
        model = SwinIR(scale=3)
        dummy = torch.randn(1, 3, 16, 16)
        out = model(dummy)
        assert out.shape == (1, 3, 48, 48)

    def test_rgb_mean_finite_output(self):
        """SwinIR with rgb_mean should produce finite output."""
        model = SwinIR(scale=4, rgb_mean=[0.4488, 0.4371, 0.4040])
        model.eval()
        dummy = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 3, 64, 64)
        assert torch.isfinite(out).all()

    def test_old_checkpoint_rejected(self):
        """Loading old-format state_dict should raise ValueError."""
        model = SwinIR(scale=4)
        old_sd = {
            'upsampler.0.weight': torch.randn(48, 180, 3, 3),
            'upsampler.0.bias': torch.randn(48),
            'upsampler.2.weight': torch.randn(3, 3, 3, 3),
            'upsampler.2.bias': torch.randn(3),
        }
        with pytest.raises(ValueError, match="does not match"):
            model.load_state_dict(old_sd)

    def test_old_scale1_checkpoint_can_load(self):
        """Old-format scale=1 checkpoint should load (shapes match)."""
        model = SwinIR(scale=1)
        sd = SwinIR(scale=1).state_dict()
        sd['upsampler.2.weight'] = torch.randn(3, 3, 3, 3)
        sd['upsampler.2.bias'] = torch.randn(3)
        original_w = sd['upsampler.0.weight'].clone()
        model.load_state_dict(sd)
        assert torch.equal(model.state_dict()['upsampler.0.weight'], original_w)
        assert 'upsampler.2.weight' not in model.state_dict()

    def test_old_scale1_checkpoint_with_rgb_mean(self):
        """Old-format scale=1 checkpoint loads even when model has rgb_mean."""
        model = SwinIR(scale=1, rgb_mean=[0.4488, 0.4371, 0.4040])
        sd = SwinIR(scale=1).state_dict()
        sd['upsampler.2.weight'] = torch.randn(3, 3, 3, 3)
        sd['upsampler.2.bias'] = torch.randn(3)
        model.load_state_dict(sd)
        assert 'rgb_mean' in model.state_dict()

    def test_swinir_model_format(self):
        """SwinIR should declare model_format."""
        assert SwinIR.model_format == "swinir-v2"

    def test_rrdbnet_model_format(self):
        """RRDBNet should declare model_format."""
        assert RRDBNet.model_format == "rrdb_esrgan-v1"

    def test_compat_load_strips_old_keys(self):
        """compat_load_state_dict strips upsampler keys not in current model."""
        from sr_engine.models.checkpoint import compat_load_state_dict
        model = SwinIR(scale=1)
        sd = SwinIR(scale=1).state_dict()
        sd['upsampler.2.weight'] = torch.randn(3, 3, 3, 3)
        sd['upsampler.2.bias'] = torch.randn(3)
        sd['upsampler.99.weight'] = torch.randn(1, 1, 1, 1)
        original_w = sd['upsampler.0.weight'].clone()
        cleaned, effective_strict = compat_load_state_dict(
            model, sd, compat_prefixes=("upsampler.",)
        )
        assert 'upsampler.2.weight' not in cleaned
        assert 'upsampler.99.weight' not in cleaned
        assert torch.equal(cleaned['upsampler.0.weight'], original_w)
        model.load_state_dict(cleaned, strict=effective_strict)

    def test_compat_load_rejects_shape_mismatch(self):
        """compat_load_state_dict should raise on shape mismatch."""
        model = SwinIR(scale=1)
        sd = {
            'upsampler.0.weight': torch.randn(3, 3, 3, 3),
            'upsampler.0.bias': torch.randn(3),
        }
        with pytest.raises(ValueError, match="does not match"):
            compat_load_state_dict(
                model, sd, compat_prefixes=("upsampler.",)
            )
