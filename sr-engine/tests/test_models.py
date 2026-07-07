"""Tests for model architectures and registry."""

import warnings

import pytest
import torch

from sr_engine.models.archs.rrdbnet import RRDBNet
from sr_engine.models.archs.swinir import SwinIR
from sr_engine.models.registry import build_model, register


class TestRRDBNet:
    def test_forward_output_shape(self):
        model = RRDBNet(scale=4)
        dummy = torch.randn(1, 3, 16, 16)
        out = model(dummy)
        assert out.shape == (1, 3, 64, 64)

    def test_forward_output_shape_scale_2(self):
        model = RRDBNet(scale=2)
        dummy = torch.randn(1, 3, 16, 16)
        out = model(dummy)
        assert out.shape == (1, 3, 32, 32)

    def test_output_finite(self):
        model = RRDBNet(scale=4)
        model.eval()
        dummy = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            out = model(dummy)
        assert torch.isfinite(out).all()


class TestSwinIR:
    def test_forward_output_shape(self):
        model = SwinIR(scale=4)
        dummy = torch.randn(1, 3, 16, 16)
        out = model(dummy)
        assert out.shape == (1, 3, 64, 64)

    def test_forward_output_shape_scale_2(self):
        model = SwinIR(scale=2)
        dummy = torch.randn(1, 3, 16, 16)
        out = model(dummy)
        assert out.shape == (1, 3, 32, 32)


class TestModelRegistry:
    def test_rrdb_esrgan_is_registered(self):
        model = build_model("rrdb_esrgan", {"scale": 4})
        assert isinstance(model, RRDBNet)

    def test_build_model_passes_kwargs(self):
        model = build_model("rrdb_esrgan", {"scale": 2})
        dummy = torch.randn(1, 3, 16, 16)
        out = model(dummy)
        assert out.shape == (1, 3, 32, 32)  # scale=2 -> 2x up

    def test_build_unknown_model_raises(self):
        with pytest.raises(ValueError, match="nonexistent"):
            build_model("nonexistent", {})

    def test_register_decorator_adds_to_registry(self):
        @register("_test_model")
        class _TestModel(torch.nn.Module):
            def forward(self, x):
                return x
        model = build_model("_test_model", {})
        assert isinstance(model, _TestModel)
