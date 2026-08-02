"""Evaluation metrics: PSNR, SSIM, LPIPS."""

import torch
import torch.nn.functional as F


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio between two image tensors.

    Accepts either ``(C, H, W)`` single images or ``(B, C, H, W)`` batches;
    for batches, PSNR is computed per-image and then averaged.

    Args:
        img1: Reference image tensor.
        img2: Distorted image tensor.
        max_val: Maximum pixel value (1.0 for float images, 255.0 for uint8).

    Returns:
        PSNR value in decibels (a 0-dim tensor).

    Note:
        When the two images are identical (MSE == 0), true PSNR is
        infinite. To keep this usable in training logs, the MSE is
        clamped to a tiny epsilon instead of returning ``inf`` — this
        yields a large but finite number rather than a real "infinite"
        PSNR for perfect reconstructions.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    if img1.dim() == 4:
        mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    else:
        mse = torch.mean((img1 - img2) ** 2)

    mse = mse.clamp(min=1e-10)
    psnr_val = 10.0 * torch.log10((max_val ** 2) / mse)

    return psnr_val.mean()


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    """Build a normalized 2D Gaussian convolution kernel, replicated per channel
    for depthwise convolution (one independent kernel per input channel)."""
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    kernel_2d = g.outer(g)
    kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    max_val: float = 1.0,
    window_size: int = 11,
) -> torch.Tensor:
    """Compute Structural Similarity Index between two image tensors.

    Accepts either ``(C, H, W)`` single images or ``(B, C, H, W)`` batches.
    Uses the standard Gaussian-window SSIM formulation (Wang et al. 2004).
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    return _ssim_map(img1, img2, window_size, c1, c2).mean()


def ms_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    max_val: float = 1.0,
    window_size: int = 11,
    levels: int = 5,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Multi-Scale Structural Similarity (Wang et al. 2003).

    Accepts either ``(C, H, W)`` single images or ``(B, C, H, W)`` batches.
    Both images must be at least ``2 ** (levels - 1) * window_size`` on each
    spatial dimension; smaller inputs are handled by capping ``levels``.

    Returns the MS-SSIM value (a 0-dim tensor, in ``[0, 1]``).
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                               device=img1.device, dtype=img1.dtype)

    max_levels = weights.numel()
    min_size = min(img1.shape[2], img1.shape[3])
    # Each downsampling halves the size; each SSIM level needs window_size.
    feasible = (min_size // (2 ** (levels - 1))) >= window_size
    if not feasible:
        max_levels = min(max_levels, max(1, (min_size // window_size).bit_length()))
        weights = weights[:max_levels]
    weights = weights / weights.sum()

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    mcs: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []

    cur1 = img1
    cur2 = img2
    for level in range(weights.numel()):
        scores.append(_ssim_map(cur1, cur2, window_size, c1, c2))
        if level < weights.numel() - 1:
            mcs.append(_ssim_map(cur1, cur2, window_size, c1, c2, cs=True))
            # 2x downsampling with a small low-pass before strided sample.
            channels = cur1.shape[1]
            k = torch.ones(channels, 1, 2, 2, device=img1.device, dtype=img1.dtype) / 4.0
            cur1 = F.conv2d(cur1, k, stride=2, padding=0, groups=channels)
            cur2 = F.conv2d(cur2, k, stride=2, padding=0, groups=channels)

    # MS-SSIM = [SSIM at coarsest scale] * prod of [CS at finer scales].
    value = scores[-1]
    for cs in mcs:
        value = value * cs
    return value.mean()


def _ssim_map(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int,
    c1: float,
    c2: float,
    cs: bool = False,
) -> torch.Tensor:
    """SSIM map over a batch, optionally returning only the contrast term."""
    channels = img1.shape[1]
    window = _gaussian_window(window_size, sigma=1.5, channels=channels,
                              device=img1.device, dtype=img1.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    if cs:
        return cs_map.mean(dim=[1, 2, 3])
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean(dim=[1, 2, 3])


_lpips_model_cache: dict[tuple[str, str], "torch.nn.Module"] = {}


def lpips(
    img1: torch.Tensor,
    img2: torch.Tensor,
    net: str = "alex",
    device: str = "cuda",
) -> torch.Tensor:
    """Compute Learned Perceptual Image Patch Similarity.

    Requires the ``lpips`` package (``pip install lpips``) — imported
    lazily here since it's an optional dependency for whoever just wants
    PSNR/SSIM. Models are cached per (net, device) pair since loading them
    is relatively expensive and this may be called every eval step.

    Assumes *img1*/*img2* are in ``[0, 1]`` range (matching
    ``PairedImageFolderDataset`` output) and rescales internally to the
    ``[-1, 1]`` range the ``lpips`` package expects.

    Args:
        img1: Reference image tensor, ``(C, H, W)`` or ``(B, C, H, W)``.
        img2: Distorted image tensor, same shape as *img1*.
        net: Backbone network (``alex``, ``vgg``, ``squeeze``).
        device: Torch device string.

    Returns:
        LPIPS distance score (a 0-dim tensor, averaged over the batch).
    """
    import lpips as lpips_lib

    cache_key = (net, device)
    if cache_key not in _lpips_model_cache:
        _lpips_model_cache[cache_key] = lpips_lib.LPIPS(net=net).to(device).eval()
    model = _lpips_model_cache[cache_key]

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # lpips expects inputs in [-1, 1]; ours are in [0, 1].
    img1_scaled = (img1.to(device) * 2.0) - 1.0
    img2_scaled = (img2.to(device) * 2.0) - 1.0

    with torch.no_grad():
        distance = model(img1_scaled, img2_scaled)

    return distance.mean()