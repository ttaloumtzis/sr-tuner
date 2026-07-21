"""Loss functions for super-resolution training."""

import logging
import os
from urllib.error import URLError

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ── VGG19 layer index ────────────────────────────────────────────────────

_VGG19_LAYER_INDEX = {
    "relu1_1": 1, "relu1_2": 3,
    "relu2_1": 6, "relu2_2": 8,
    "relu3_1": 11, "relu3_2": 13, "relu3_3": 15, "relu3_4": 17,
    "relu4_1": 20, "relu4_2": 22, "relu4_3": 24, "relu4_4": 26,
    "relu5_1": 29, "relu5_2": 31, "relu5_3": 33, "relu5_4": 35,
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VGGFeatureExtractor(nn.Module):
    """Shared VGG19 backbone for perceptual and style losses.

    Loaded once and shared across all sub-losses that need VGG features.
    Frozen, eval mode, ImageNet normalization applied internally.
    """

    def __init__(self, layer_ids: list[str] | None = None) -> None:
        super().__init__()
        self.layer_ids = layer_ids or ["relu5_4"]
        unknown = [name for name in self.layer_ids if name not in _VGG19_LAYER_INDEX]
        if unknown:
            raise ValueError(
                f"Unknown VGG19 layer name(s): {unknown}. "
                f"Available: {sorted(_VGG19_LAYER_INDEX)}"
            )

        from torchvision.models import vgg19, VGG19_Weights
        try:
            import certifi
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        except ImportError:
            pass

        try:
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        except URLError as e:
            raise RuntimeError(
                "Failed to download VGG19 pretrained weights (required for perceptual loss). "
                "Check your internet connection or proxy settings. "
                f"Original error: {e}"
            ) from e

        max_idx = max(_VGG19_LAYER_INDEX[name] for name in self.layer_ids)
        self.vgg = nn.Sequential(*list(vgg.children())[: max_idx + 1]).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

        self._all_layers = sorted({_VGG19_LAYER_INDEX[n] for n in self.layer_ids})

        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = (x - self.mean) / self.std
        features: dict[str, torch.Tensor] = {}
        wanted = {_VGG19_LAYER_INDEX[n]: n for n in self.layer_ids}
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            if idx in wanted:
                features[wanted[idx]] = x
        return features


# ── Pixel losses ─────────────────────────────────────────────────────────

class L1Loss(nn.Module):
    """Charbonnier loss: ``mean(sqrt((pred - target)^2 + eps^2))``."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()


class L2Loss(nn.Module):
    """Mean squared error (MSE) loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


# ── Perceptual loss (VGG-based) ──────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """Perceptual (VGG19-based) loss.

    Accepts an optional shared ``VGGFeatureExtractor`` to avoid
    loading multiple VGG19 backbones when ``StyleLoss`` is also active.
    """

    def __init__(
        self,
        layers: list[str] | None = None,
        vgg_extractor: VGGFeatureExtractor | None = None,
    ) -> None:
        super().__init__()
        self.layer_ids = layers or ["relu5_4"]
        if vgg_extractor is not None:
            missing = [n for n in self.layer_ids if n not in vgg_extractor.layer_ids]
            if missing:
                raise ValueError(
                    f"Shared VGGFeatureExtractor missing requested layers: {missing}. "
                    f"It has: {vgg_extractor.layer_ids}"
                )
            self.extractor = vgg_extractor
        else:
            self.extractor = VGGFeatureExtractor(self.layer_ids)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_feats = self.extractor(target)
        pred_feats = self.extractor(pred)
        loss = 0.0
        for name in self.layer_ids:
            loss += F.l1_loss(pred_feats[name], target_feats[name])
        return loss


# ── Edge / Gradient loss ─────────────────────────────────────────────────

class EdgeLoss(nn.Module):
    """Sobel gradient-magnitude L1 loss.

    Penalises differences in edge structure by comparing gradient
    magnitudes of prediction and target.
    """

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("filter_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("filter_y", sobel_y.view(1, 1, 3, 3))

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.filter_x, padding=1)
        gy = F.conv2d(gray, self.filter_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        g_pred = self._gradient_magnitude(pred)
        g_target = self._gradient_magnitude(target)
        return F.l1_loss(g_pred, g_target)


# ── Frequency (FFT) loss ─────────────────────────────────────────────────

class FrequencyLoss(nn.Module):
    """FFT-based frequency-domain loss.

    Computes L1 loss on log-magnitude spectra, encouraging the
    model to match high-frequency detail. Complements spatial losses.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def _fft_mag(x: torch.Tensor) -> torch.Tensor:
            fft = torch.fft.rfft2(x, norm="backward")
            return torch.log(1 + fft.abs())

        mag_pred = _fft_mag(pred)
        mag_target = _fft_mag(target)
        return F.l1_loss(mag_pred, mag_target)


# ── Style (Gram) loss ────────────────────────────────────────────────────

class StyleLoss(nn.Module):
    """Gram-matrix style loss on VGG19 features.

    Measures style dissimilarity via L2 distance between Gram matrices
    of selected feature maps. Uses an optional shared ``VGGFeatureExtractor``.
    """

    def __init__(
        self,
        layers: list[str] | None = None,
        vgg_extractor: VGGFeatureExtractor | None = None,
    ) -> None:
        super().__init__()
        self.layer_ids = layers or [
            "relu1_2", "relu2_2", "relu3_4", "relu4_4", "relu5_2",
        ]
        if vgg_extractor is not None:
            missing = [n for n in self.layer_ids if n not in vgg_extractor.layer_ids]
            if missing:
                raise ValueError(
                    f"Shared VGGFeatureExtractor missing requested layers: {missing}. "
                    f"It has: {vgg_extractor.layer_ids}"
                )
            self.extractor = vgg_extractor
        else:
            self.extractor = VGGFeatureExtractor(self.layer_ids)

    @staticmethod
    def _gram(x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feats = x.view(B, C, H * W)
        gram = torch.bmm(feats, feats.transpose(1, 2))
        return gram / (C * H * W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_feats = self.extractor(target)
        pred_feats = self.extractor(pred)
        loss = 0.0
        for name in self.layer_ids:
            g_pred = self._gram(pred_feats[name])
            g_target = self._gram(target_feats[name])
            loss += F.mse_loss(g_pred, g_target)
        return loss


# ── SSIM loss ────────────────────────────────────────────────────────────

class SSIMLoss(nn.Module):
    """Structural dissimilarity loss: ``1 - SSIM``.

    Self-contained implementation using an 11x11 Gaussian window.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        self.window_size = window_size
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel_1d = gauss.view(1, 1, window_size, 1)
        kernel_2d = kernel_1d * kernel_1d.transpose(2, 3)
        self.register_buffer("window", kernel_2d)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pw = self.window_size // 2
        window = self.window.expand(x.shape[1], -1, -1, -1).contiguous()
        mu_x = F.conv2d(x, window, padding=pw, groups=x.shape[1])
        mu_y = F.conv2d(y, window, padding=pw, groups=y.shape[1])
        sigma_xx = F.conv2d(x * x, window, padding=pw, groups=x.shape[1]) - mu_x ** 2
        sigma_yy = F.conv2d(y * y, window, padding=pw, groups=y.shape[1]) - mu_y ** 2
        sigma_xy = F.conv2d(x * y, window, padding=pw, groups=x.shape[1]) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_xx + sigma_yy + C2))
        return ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1 - self._ssim(pred, target)


# ── LPIPS loss ───────────────────────────────────────────────────────────

class LPIPSLoss(nn.Module):
    """Learned Perceptual Image Patch Similarity loss.

    Wraps the ``lpips`` package. Falls back to VGG-based perceptual loss
    with a clear error message if ``lpips`` is not installed.
    """

    def __init__(self, net: str = "alex") -> None:
        super().__init__()
        try:
            import lpips as lpips_pkg
            self.lpips = lpips_pkg.LPIPS(net=net).eval()
            for p in self.lpips.parameters():
                p.requires_grad = False
        except ImportError:
            raise ImportError(
                "LPIPSLoss requires the 'lpips' package. Install it with: "
                "uv pip install lpips  or  pip install lpips"
            )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lpips(pred, target).mean()


# ── GAN loss (placeholder, not wired) ────────────────────────────────────

class GANLoss(nn.Module):
    """Standard (vanilla) or least-squares (LSGAN) adversarial loss.

    Currently defined but not wired into the composable loss system.
    Requires a discriminator model and alternating G/D training steps.
    """

    def __init__(self, gan_type: str = "vanilla") -> None:
        super().__init__()
        if gan_type not in ("vanilla", "lsgan"):
            raise ValueError(f"Unsupported gan_type: '{gan_type}'. Expected 'vanilla' or 'lsgan'.")
        self.gan_type = gan_type
        self.loss_fn = nn.BCEWithLogitsLoss() if gan_type == "vanilla" else nn.MSELoss()

    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = torch.full_like(pred, fill_value=1.0 if target_is_real else 0.0)
        return self.loss_fn(pred, target)


# ── Loss registry ────────────────────────────────────────────────────────

LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "l1": L1Loss,
    "l2": L2Loss,
    "vgg": PerceptualLoss,
    "edge": EdgeLoss,
    "style": StyleLoss,
    "fft": FrequencyLoss,
    "ssim": SSIMLoss,
    "lpips": LPIPSLoss,
}

_LOSS_PARAM_DEFAULTS: dict[str, dict] = {
    "vgg": {"layers": ["relu5_4"]},
    "style": {"layers": ["relu1_2", "relu2_2", "relu3_4", "relu4_4", "relu5_2"]},
    "l1": {"eps": 1e-6},
    "ssim": {"window_size": 11, "sigma": 1.5},
    "lpips": {"net": "alex"},
}

_PIXEL_LOSS_TYPES = {"l1", "l2"}
_VGG_BASED_TYPES = {"vgg", "style"}


def _migrate_legacy_loss_config(config: dict) -> dict:
    """Convert old flat config format to the new composable format.

    Old format (single loss):
      ``{"perceptual_weight": 0.1, "perceptual_layers": ["relu5_4"]}``

    New format:
      ``{"pixel": {"type": "l1", "weight": 1.0},
         "perceptual": {"type": "vgg", "weight": 0.1, "layers": ["relu5_4"]}}``
    """
    if not config:
        return _default_loss_config()

    has_legacy = "perceptual_weight" in config or "perceptual_layers" in config
    has_new = any(
        isinstance(v, dict) and "type" in v
        for v in config.values()
    )
    if has_new and not has_legacy:
        return config

    result: dict = {}
    needs_default_pixel = True

    for name, val in config.items():
        if name == "perceptual_weight":
            w = float(val)
            if w > 0:
                result["perceptual"] = {"type": "vgg", "weight": w}
            continue
        if name == "perceptual_layers":
            cfg = result.setdefault("perceptual", {"type": "vgg", "weight": 0.1})
            cfg["layers"] = val
            continue
        if isinstance(val, dict) and "type" in val:
            needs_default_pixel = False
            result[name] = val
            continue
        if isinstance(val, dict):
            needs_default_pixel = False
            result[name] = val

    if needs_default_pixel:
        result["pixel"] = {"type": "l1", "weight": 1.0}

    return result


def _default_loss_config() -> dict:
    return {
        "pixel": {"type": "l1", "weight": 1.0},
        "perceptual": {"type": "vgg", "weight": 0.1, "layers": ["relu5_4"]},
    }


def _vgg_layer_ids_from_config(config: dict) -> list[str]:
    """Collect all VGG layer IDs needed across all losses in the config."""
    needed: set[str] = set()
    for name, cfg in config.items():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("type") in _VGG_BASED_TYPES:
            layers = cfg.get("layers")
            if layers:
                needed.update(layers)
            else:
                defaults = _LOSS_PARAM_DEFAULTS.get(cfg["type"], {})
                needed.update(defaults.get("layers", []))
    return sorted(needed)


def build_composite_loss(
    config: dict | None,
    device: torch.device,
) -> nn.Module:
    """Build a composable loss module from a loss config dict.

    Args:
        config: Loss configuration dict (new or legacy format).
                ``None`` or empty dict uses default (L1 + VGG perceptual).
        device: Target device for all sub-losses.

    Returns:
        A ``CompositeLoss`` module whose ``forward()`` returns
        ``(total_loss_tensor, {name: component_value, ...})``.
    """
    if not config:
        config = _default_loss_config()
    else:
        config = _migrate_legacy_loss_config(config)

    vgg_layers = _vgg_layer_ids_from_config(config)
    shared_vgg: VGGFeatureExtractor | None = None
    if vgg_layers:
        shared_vgg = VGGFeatureExtractor(vgg_layers)

    losses: dict[str, nn.Module] = {}
    weights: dict[str, float] = {}

    for name, loss_cfg in config.items():
        if not isinstance(loss_cfg, dict) or "type" not in loss_cfg:
            raise ValueError(
                f"Loss '{name}' must be a dict with a 'type' field. Got: {loss_cfg}"
            )
        lt = loss_cfg["type"]
        if lt not in LOSS_REGISTRY:
            raise ValueError(
                f"Unknown loss type '{lt}' for loss '{name}'. "
                f"Available: {sorted(LOSS_REGISTRY)}"
            )
        cls = LOSS_REGISTRY[lt]
        params = dict(_LOSS_PARAM_DEFAULTS.get(lt, {}))
        for k, v in loss_cfg.items():
            if k not in ("type", "weight"):
                params[k] = v

        if lt in _VGG_BASED_TYPES and shared_vgg is not None:
            params["vgg_extractor"] = shared_vgg

        losses[name] = cls(**params)
        weights[name] = float(loss_cfg.get("weight", 1.0))

    has_pixel = any(lt in _PIXEL_LOSS_TYPES for lt in
                    [c.get("type") for c in config.values() if isinstance(c, dict)])
    if not has_pixel:
        raise ValueError(
            "At least one pixel loss (l1 or l2) is required. "
            "Add a pixel loss to your config or remove custom loss config to use defaults."
        )

    return CompositeLoss(losses, weights, shared_vgg, device)


class CompositeLoss(nn.Module):
    """Orchestrates multiple sub-losses and returns weighted total + components."""

    def __init__(
        self,
        losses: dict[str, nn.Module],
        weights: dict[str, float],
        shared_vgg: VGGFeatureExtractor | None,
        device: torch.device,
    ) -> None:
        super().__init__()
        self._losses = nn.ModuleDict(losses)
        self._weights = weights
        self._shared_vgg = shared_vgg
        self.to(device)
        for loss_mod in self._losses.values():
            loss_mod.eval() if _is_vgg_loss(loss_mod) else None

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = torch.zeros((), device=pred.device)
        components: dict[str, float] = {}
        for name, loss_mod in self._losses.items():
            val = loss_mod(pred, target)
            w = self._weights[name]
            total = total + w * val
            components[name] = val.item()
        return total, components

    @property
    def loss_names(self) -> list[str]:
        return list(self._losses.keys())


def _is_vgg_loss(module: nn.Module) -> bool:
    return isinstance(module, (PerceptualLoss, StyleLoss))
