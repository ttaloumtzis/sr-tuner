"""Backend-specific op swaps. All backend branching lives here."""

import torch
import torch.nn as nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """Backend-aware scaled dot-product attention.

    Delegates to flash-attention when available, otherwise falls back to
    PyTorch's native ``scaled_dot_product_attention``.
    """
    raise NotImplementedError("TODO: implement backend-aware SDPA dispatch")


def get_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    bias: bool = True,
) -> nn.Conv2d:
    """Return a Conv2d layer, swapping in backend-specific implementations if available."""
    raise NotImplementedError("TODO: implement backend-specific conv2d if needed")
