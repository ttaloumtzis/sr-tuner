"""SwinIR architecture — Swin Transformer-based image restoration."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import register


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    B = windows.shape[0] // (H // window_size * W // window_size)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        relative_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_bias = relative_bias.view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(-1, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinTransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 8,
                 shift_size: int = 0, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        shortcut = x

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm1(x).permute(0, 3, 1, 2).contiguous()

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_r, 0, pad_b))
        _, _, Hp, Wp = x.shape

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        x = x.permute(0, 2, 3, 1).contiguous()
        windows = window_partition(x, self.window_size)
        nW = windows.shape[0]

        attn_mask = self._compute_attention_mask(B, Hp, Wp, x.device) if self.shift_size > 0 else None
        attn_windows = self.attn(windows, mask=attn_mask)

        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x[:, :H, :W, :].permute(0, 3, 1, 2).contiguous()
        x = shortcut + x

        residual = x
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm2(x).permute(0, 3, 1, 2).contiguous()
        x = residual + self.mlp(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        return x

    def _compute_attention_mask(self, B: int, Hp: int, Wp: int, device: torch.device) -> torch.Tensor:
        ws = self.window_size
        shift = self.shift_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, ws)
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))


class RSTB(nn.Module):
    def __init__(self, dim: int, num_heads: int, depth: int, window_size: int = 8,
                 mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            shift_size = 0 if i % 2 == 0 else window_size // 2
            self.layers.append(SwinTransformerLayer(
                dim, num_heads, window_size, shift_size, mlp_ratio))
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        for layer in self.layers:
            x = layer(x)
        return self.conv(x) + shortcut


@register("swinir")
class SwinIR(nn.Module):
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        embed_dim: int = 180,
        depths: Optional[list[int]] = None,
        num_heads: Optional[list[int]] = None,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        img_range: float = 1.0,
        upsampler: str = "pixelshuffle",
        scale: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.img_range = img_range
        depths = depths or [6, 6, 6, 6, 6, 6]
        num_heads = num_heads or [6, 6, 6, 6, 6, 6]

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.rstb_layers = nn.ModuleList()
        for i, (d, nh) in enumerate(zip(depths, num_heads)):
            self.rstb_layers.append(RSTB(embed_dim, nh, d, window_size, mlp_ratio))

        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if upsampler == "pixelshuffle":
            self.upsampler = nn.Sequential(
                nn.Conv2d(embed_dim, num_out_ch * scale ** 2, 3, 1, 1),
                nn.PixelShuffle(scale),
                nn.Conv2d(num_out_ch, num_out_ch, 3, 1, 1),
            )
        else:
            self.upsampler = nn.Sequential(
                nn.Upsample(scale_factor=scale, mode='nearest'),
                nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.img_range

        shallow = self.conv_first(x)

        deep = shallow
        for rstb in self.rstb_layers:
            deep = rstb(deep)

        body = self.conv_after_body(deep) + shallow
        body = self.conv_before_upsample(body)

        out = self.upsampler(body)
        return out / self.img_range if self.img_range > 1 else out
