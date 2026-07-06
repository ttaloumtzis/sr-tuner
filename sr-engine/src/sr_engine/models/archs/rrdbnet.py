"""RRDBNet architecture — Residual-in-Residual Dense Block network."""

import torch
import torch.nn as nn
from ..registry import register


# A helper for the dense block logic
class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        # Standard RRDB structure: 3 dense layers + residual connection
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.conv3(torch.cat([x, x1, x2], 1))
        return x3 * 0.2 + x  # Residual scaling


@register("rrdb_esrgan")
class RRDBNet(nn.Module):
    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
        **kwargs  # <--- This will swallow the 'name' and 'type' keys
    ):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # Stack of RRDB blocks
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])

        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Simple upsampling (Upsample + Conv)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='nearest'),
            nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        # Residual-in-Residual connection
        body_feat = self.conv_body(self.body(feat)) + feat
        return self.upsample(body_feat)
