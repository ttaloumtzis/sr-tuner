"""Loss functions for super-resolution training."""

import torch
import torch.nn as nn


class L1Loss(nn.Module):
    """Charbonnier / L1 loss for SR."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        raise NotImplementedError("TODO: implement L1Loss")


class PerceptualLoss(nn.Module):
    """Perceptual (VGG-based) loss."""

    def __init__(self, layer_ids: list[str] | None = None) -> None:
        super().__init__()
        raise NotImplementedError("TODO: implement PerceptualLoss")


class GANLoss(nn.Module):
    """Standard or relativistic GAN loss."""

    def __init__(self, gan_type: str = "vanilla") -> None:
        super().__init__()
        raise NotImplementedError("TODO: implement GANLoss")
