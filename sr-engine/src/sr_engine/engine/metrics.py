"""Evaluation metrics: PSNR, SSIM, LPIPS."""

import torch


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio between two image tensors.

    Args:
        img1: Reference image tensor.
        img2: Distorted image tensor.
        max_val: Maximum pixel value (1.0 for float images, 255.0 for uint8).

    Returns:
        PSNR value in decibels.
    """
    raise NotImplementedError("TODO: implement PSNR")


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    max_val: float = 1.0,
    window_size: int = 11,
) -> torch.Tensor:
    """Compute Structural Similarity Index between two image tensors."""
    raise NotImplementedError("TODO: implement SSIM")


def lpips(
    img1: torch.Tensor,
    img2: torch.Tensor,
    net: str = "alex",
    device: str = "cuda",
) -> torch.Tensor:
    """Compute Learned Perceptual Image Patch Similarity.

    Args:
        img1: Reference image tensor.
        img2: Distorted image tensor.
        net: Backbone network (``alex``, ``vgg``, ``squeeze``).
        device: Torch device string.

    Returns:
        LPIPS distance score.
    """
    raise NotImplementedError("TODO: implement LPIPS")
