"""RRDBNet architecture — Residual-in-Residual Dense Block network."""

from typing import Any
import torch
import torch.nn as nn
from ..registry import register


@register("rrdbnet")
class RRDBNet(nn.Module):
    """RRDB-based super-resolution network.

    Args:
        num_in_ch: Number of input channels.
        num_out_ch: Number of output channels.
        num_feat: Number of feature channels.
        num_block: Number of RRDB blocks.
        num_grow_ch: Channels to grow per dense layer inside each RRDB.
        scale: Upsampling scale factor.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        scale: int = 4,
    ) -> None:
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_block = num_block
        self.num_grow_ch = num_grow_ch
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve input tensor *x*.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Super-resolved tensor of shape ``(B, C_out, H*scale, W*scale)``.
        """
        raise NotImplementedError("TODO: implement RRDBNet forward pass")
