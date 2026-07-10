"""Tests for device/kernels.py — backend-aware ops."""

import torch

from sr_engine.device.kernels import scaled_dot_product_attention, get_conv2d


class TestScaledDotProductAttention:
    """Tests for ``scaled_dot_product_attention``."""

    def test_basic_forward(self):
        """Forward pass with default parameters should not crash."""
        q = torch.randn(1, 2, 8, 16)
        k = torch.randn(1, 2, 8, 16)
        v = torch.randn(1, 2, 8, 16)
        out = scaled_dot_product_attention(q, k, v)
        assert out.shape == (1, 2, 8, 16)

    def test_with_mask(self):
        """Forward pass with an attention mask."""
        q = torch.randn(1, 2, 8, 16)
        k = torch.randn(1, 2, 8, 16)
        v = torch.randn(1, 2, 8, 16)
        mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
        out = scaled_dot_product_attention(q, k, v, attn_mask=mask)
        assert out.shape == (1, 2, 8, 16)

    def test_different_heads(self):
        """Multiple attention heads should work."""
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 4, 16, 32)
        v = torch.randn(2, 4, 16, 32)
        out = scaled_dot_product_attention(q, k, v)
        assert out.shape == (2, 4, 16, 32)

    def test_dropout(self):
        """Dropout should not affect output shape."""
        q = torch.randn(1, 2, 8, 16)
        k = torch.randn(1, 2, 8, 16)
        v = torch.randn(1, 2, 8, 16)
        out = scaled_dot_product_attention(q, k, v, dropout_p=0.1)
        assert out.shape == (1, 2, 8, 16)


class TestGetConv2d:
    """Tests for ``get_conv2d``."""

    def test_returns_module(self):
        """Should return an nn.Conv2d instance."""
        conv = get_conv2d(3, 64, kernel_size=3)
        assert isinstance(conv, torch.nn.Conv2d)

    def test_forward(self):
        """Forward pass should not crash."""
        conv = get_conv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(1, 3, 32, 32)
        out = conv(x)
        assert out.shape == (1, 16, 32, 32)
