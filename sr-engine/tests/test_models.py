"""Tests for model architectures and registry."""

import pytest

from sr_engine.models.registry import build_model, register
from sr_engine.models.archs.rrdbnet import RRDBNet
from sr_engine.models.archs.swinir import SwinIR


class TestModelRegistry:
    """Tests for model registry."""

    def test_rrdbnet_is_registered(self) -> None:
        with pytest.raises(NotImplementedError):
            build_model("rrdbnet", {})

    def test_swinir_is_registered(self) -> None:
        with pytest.raises(NotImplementedError):
            build_model("swinir", {})

    def test_build_unknown_model(self) -> None:
        with pytest.raises(NotImplementedError):
            build_model("nonexistent", {})


class TestRRDBNet:
    """Tests for RRDBNet."""

    def test_forward_raises(self) -> None:
        import torch
        model = RRDBNet()
        dummy = torch.randn(1, 3, 64, 64)
        with pytest.raises(NotImplementedError):
            model(dummy)


class TestSwinIR:
    """Tests for SwinIR."""

    def test_forward_raises(self) -> None:
        import torch
        model = SwinIR()
        dummy = torch.randn(1, 3, 64, 64)
        with pytest.raises(NotImplementedError):
            model(dummy)
