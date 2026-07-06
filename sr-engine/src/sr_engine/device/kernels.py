"""Backend-specific op swaps. All backend branching lives here."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext


def scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
) -> torch.Tensor:
    """
    Backend-aware scaled dot-product attention.

    Automatically dispatches to FlashAttention, Memory-Efficient Attention,
    or the Math implementation based on hardware support and availability.
    """

    # You can customize which kernels are allowed here if needed.
    # For example, if you want to strictly prefer FlashAttention:
    # kernel_context = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION)
    kernel_context = nullcontext()

    with kernel_context:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )


def get_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    bias: bool = True,
) -> nn.Conv2d:
    """Return the standard optimized Conv2d layer."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, padding, bias=bias
    )
