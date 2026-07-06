"""SwinIR architecture — Swin Transformer-based image restoration."""

from typing import Any
import torch
import torch.nn as nn
from ..registry import register


@register("swinir")
class SwinIR(nn.Module):
    """SwinIR super-resolution network.

    Args:
        num_in_ch: Number of input channels.
        num_out_ch: Number of output channels.
        embed_dim: Embedding dimension.
        depths: Depth of each Swin Transformer stage.
        num_heads: Number of attention heads per stage.
        window_size: Local window size for attention.
        mlp_ratio: MLP expansion ratio.
        img_range: Image value range (1.0 or 255.0).
        upsampler: Upsampling method (``pixelshuffle``, ``pixelshuffledirect``, ``nearest+conv``).
        scale: Upsampling scale factor.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        embed_dim: int = 180,
        depths: list[int] | None = None,
        num_heads: list[int] | None = None,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        img_range: float = 1.0,
        upsampler: str = "pixelshuffle",
        scale: int = 4,
    ) -> None:
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.embed_dim = embed_dim
        self.depths = depths or [6, 6, 6, 6, 6, 6]
        self.num_heads = num_heads or [6, 6, 6, 6, 6, 6]
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.img_range = img_range
        self.upsampler = upsampler
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve input tensor *x*.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Super-resolved tensor of shape ``(B, C_out, H*scale, W*scale)``.
        """
        raise NotImplementedError("TODO: implement SwinIR forward pass")
