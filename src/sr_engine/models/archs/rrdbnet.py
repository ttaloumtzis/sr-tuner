"""RRDBNet architecture — Residual-in-Residual Dense Block network."""

import torch
import torch.nn as nn
from ..registry import register


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block with 3 dense layers and residual scaling."""

    def __init__(self, nf: int, gc: int = 32):
        """Initialise three convolutional dense layers.

        Args:
            nf: Number of feature channels.
            gc: Growth channel count per dense layer.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply three dense layers with residual scaling (``* 0.2``).

        Args:
            x: Input tensor ``(B, nf, H, W)``.

        Returns:
            Output tensor ``(B, nf, H, W)``.
        """
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.conv3(torch.cat([x, x1, x2], 1))
        return x3 * 0.2 + x


@register("rrdb_esrgan")
class RRDBNet(nn.Module):
    """RRDB-based ESRGAN super-resolution model.

    Stacks multiple RRDB blocks, applies a post-body convolution, and
    upsamples with nearest-neighbour interpolation plus a convolution.
    """

    model_format = "rrdb_esrgan-v1"

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        scale: int = 4,
        **kwargs,
    ):
        """Initialise the model with stacked RRDB blocks and upsampler.

        Args:
            num_in_ch: Number of input channels.
            num_out_ch: Number of output channels.
            num_feat: Base feature channel count.
            num_block: Number of RRDB blocks in the body.
            num_grow_ch: Growth channels per RRDB dense layer.
            scale: Super-resolution scale factor.
        """
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])

        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='nearest'),
            nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply first conv, RRDB body with residual, and upsampler.

        Args:
            x: Input tensor ``(B, C, H, W)``.

        Returns:
            Output tensor ``(B, C_out, H*scale, W*scale)``.
        """
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat)) + feat
        return self.upsample(body_feat)
