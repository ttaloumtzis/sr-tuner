"""SwinIR architecture — Swin Transformer-based image restoration."""

import torch
import torch.nn as nn
from ..registry import register

@register("swinir")
class SwinIR(nn.Module):
    """SwinIR super-resolution network implementation."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        embed_dim: int = 180,
        depths: list[int] | None = None,
        num_heads: list[int] | None = None,
        window_size: int = 8,
        scale: int = 4,
        **kwargs  # Captures extra config parameters
    ) -> None:
        super().__init__()
        self.scale = scale
        depths = depths or [6, 6, 6, 6, 6, 6]

        # 1. Shallow Feature Extraction
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # 2. Deep Feature Extraction (Backbone)
        # Placeholder for transformer layers; in a full impl, replace with RSTB modules
        self.layers = nn.Sequential(
            *[nn.Conv2d(embed_dim, embed_dim, 3, 1, 1) for _ in range(sum(depths))]
        )
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # 3. Reconstruction
        self.upsampler = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='nearest'),
            nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve input tensor *x*."""
        # Feature extraction
        feat_shallow = self.conv_first(x)

        # Transformer backbone processing
        feat_deep = self.layers(feat_shallow)

        # Residual connection
        feat = self.conv_after_body(feat_deep) + feat_shallow

        # Final upsampling
        return self.upsampler(feat)