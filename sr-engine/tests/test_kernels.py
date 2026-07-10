"""Tests for device/kernels.py — backend-agnostic op swaps."""

import torch
import pytest

from sr_engine.device.kernels import scaled_dot_product_attention, get_conv2d


class TestScaledDotProductAttention:
    def test_basic_forward(self):
        q = torch.randn(1, 4, 8, 16)
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)
        result = scaled_dot_product_attention(q, k, v)
        assert result.shape == (1, 4, 8, 16)

    def test_with_mask(self):
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        result = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        assert result.shape == (1, 2, 4, 8)


class TestGetConv2d:
    def test_basic_creation(self):
        conv = get_conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        assert conv.in_channels == 3
        assert conv.out_channels == 64
        assert conv.kernel_size == (3, 3)

    def test_forward(self):
        conv = get_conv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(1, 3, 32, 32)
        result = conv(x)
        assert result.shape == (1, 16, 32, 32)
