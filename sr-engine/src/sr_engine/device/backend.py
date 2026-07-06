"""Backend detection and capability flags for CUDA and ROCm."""

import torch
from torch import device as Device


def get_device() -> Device:
    """Detect and return the available compute device.

    Priority: CUDA > ROCm > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_rocm() -> bool:
    """Return True if the active CUDA device is a ROCm backend."""
    if not torch.cuda.is_available():
        return False
    return torch.version.hip is not None


def autocast_dtype() -> torch.dtype:
    """Return the recommended autocast dtype for the current backend.

    For ROCm returns bfloat16 if supported, otherwise float16.
    For CUDA returns bfloat16 if supported, otherwise float16.
    For CPU returns float32.
    """
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def supports_flash_attn() -> bool:
    """Return True if flash-attention is available on the current backend."""
    try:
        import flash_attn  # type: ignore[import-untyped]  # noqa: F401
        return True
    except ImportError:
        return False
