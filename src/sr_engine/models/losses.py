"""Loss functions for super-resolution training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    """Charbonnier loss: a differentiable, robust approximation of L1.

    ``loss = mean(sqrt((pred - target)^2 + eps^2))``

    Behaves like L1 for large errors but stays smooth (and differentiable)
    near zero, which tends to train more stably than plain L1 for SR pixel
    loss. ``eps`` controls how "L2-like" the loss is near zero; smaller
    values approach true L1 more closely.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        """Configure the epsilon smoothing parameter.

        Args:
            eps: Small constant for numerical stability near zero.
        """
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the Charbonnier loss.

        Args:
            pred: Predicted tensor.
            target: Target tensor.

        Returns:
            Scalar loss value.
        """
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


# Standard VGG19 layer-name -> index mapping into torchvision's
# ``vgg19(...).features`` Sequential, using the layer immediately after
# each named activation (i.e. "relu3_1" is the output *after* that ReLU).
# This mirrors the layer naming convention used by most SRGAN/ESRGAN
# implementations.
_VGG19_LAYER_INDEX = {
    "relu1_1": 1,
    "relu1_2": 3,
    "relu2_1": 6,
    "relu2_2": 8,
    "relu3_1": 11,
    "relu3_2": 13,
    "relu3_3": 15,
    "relu3_4": 17,
    "relu4_1": 20,
    "relu4_2": 22,
    "relu4_3": 24,
    "relu4_4": 26,
    "relu5_1": 29,
    "relu5_2": 31,
    "relu5_3": 33,
    "relu5_4": 35,
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class PerceptualLoss(nn.Module):
    """Perceptual (VGG19-based) loss.

    Computes L1 distance between VGG19 feature activations of the
    prediction and target at the requested layers, and returns the
    (weighted) sum. Inputs are expected in ``[0, 1]`` range (matching
    ``PairedImageFolderDataset`` output) and are normalized internally
    with ImageNet statistics before being fed through VGG.

    Downloads pretrained ImageNet VGG19 weights on first use (requires
    internet access); the backbone is frozen (no gradient updates, eval
    mode) since it's only used as a fixed feature extractor.
    """

    def __init__(self, layer_ids: list[str] | None = None) -> None:
        """Load a truncated VGG19 feature extractor.

        Args:
            layer_ids: List of VGG19 layer names for feature extraction.
                       Defaults to ``["relu5_4"]``.

        Raises:
            ValueError: If any layer name is unknown.
        """
        super().__init__()

        self.layer_ids = layer_ids or ["relu5_4"]
        unknown = [name for name in self.layer_ids if name not in _VGG19_LAYER_INDEX]
        if unknown:
            raise ValueError(
                f"Unknown VGG19 layer name(s): {unknown}. "
                f"Available: {sorted(_VGG19_LAYER_INDEX)}"
            )

        from torchvision.models import vgg19, VGG19_Weights

        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        max_index = max(_VGG19_LAYER_INDEX[name] for name in self.layer_ids)

        self.vgg = nn.Sequential(*list(vgg.children())[: max_index + 1]).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def _extract_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the truncated VGG forward and collect named feature maps.

        Args:
            x: Input tensor ``(B, 3, H, W)`` normalised to ``[0, 1]``.

        Returns:
            Dict mapping layer names to their feature tensors.
        """
        x = (x - self.mean) / self.std

        features = {}
        wanted_indices = {_VGG19_LAYER_INDEX[name]: name for name in self.layer_ids}

        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            if idx in wanted_indices:
                features[wanted_indices[idx]] = x

        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between prediction and target.

        Args:
            pred: Predicted tensor ``(B, 3, H, W)`` in ``[0, 1]``.
            target: Target tensor ``(B, 3, H, W)`` in ``[0, 1]``.

        Returns:
            Scalar loss value.
        """
        with torch.no_grad():
            target_features = self._extract_features(target)

        pred_features = self._extract_features(pred)

        loss = 0.0
        for name in self.layer_ids:
            loss = loss + F.l1_loss(pred_features[name], target_features[name])

        return loss


class GANLoss(nn.Module):
    """Standard (vanilla) or least-squares (LSGAN) adversarial loss.

    Args:
        gan_type: ``"vanilla"`` uses BCE-with-logits (discriminator outputs
            raw logits). ``"lsgan"`` uses MSE (discriminator outputs are
            treated as un-squashed scores targeting 1.0/0.0).
    """

    def __init__(self, gan_type: str = "vanilla") -> None:
        """Configure the GAN loss type.

        Args:
            gan_type: ``"vanilla"`` or ``"lsgan"``.

        Raises:
            ValueError: If ``gan_type`` is not recognised.
        """
        super().__init__()

        if gan_type not in ("vanilla", "lsgan"):
            raise ValueError(
                f"Unsupported gan_type: '{gan_type}'. Expected 'vanilla' or 'lsgan'."
            )

        self.gan_type = gan_type
        self.loss_fn = nn.BCEWithLogitsLoss() if gan_type == "vanilla" else nn.MSELoss()

    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Compute the GAN loss.

        Args:
            pred: Discriminator output tensor.
            target_is_real: ``True`` for real samples, ``False`` for fake.

        Returns:
            Scalar loss value.
        """
        target = torch.full_like(pred, fill_value=1.0 if target_is_real else 0.0)
        return self.loss_fn(pred, target)
