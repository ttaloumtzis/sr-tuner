"""Backend detection and capability flags for CUDA and ROCm."""

import torch

def get_device_name() -> str:
    """Returns 'cuda' for ROCm/NVIDIA or 'cpu'."""
    if torch.cuda.is_available():
        # ROCm devices report as 'cuda' in PyTorch
        return "cuda"
    return "cpu"

def get_device() -> torch.device:
    """Returns the torch.device object."""
    return torch.device(get_device_name())

def is_rocm() -> bool:
    """Return True if the active CUDA device is a ROCm backend."""
    if not torch.cuda.is_available():
        return False
    # Using getattr avoids static analysis 'missing reference' errors
    return getattr(torch.version, "hip", None) is not None

def autocast_dtype() -> torch.dtype:
    """Return the recommended autocast dtype for the current backend.

    Uses bfloat16 if supported by hardware (both ROCm/CUDA),
    otherwise falls back to float16.
    """
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def supports_flash_attn() -> bool:
    """Return True if PyTorch native Scaled Dot Product Attention is available."""
    # Check if the function exists and if the backend is initialized
    if not hasattr(torch.backends, 'cuda') or not torch.cuda.is_available():
        return False

    # SDPA is supported if the following functions exist
    try:
        # This confirms that the SDPA dispatching mechanism is present
        return torch.backends.cuda.flash_sdp_enabled() or \
            torch.backends.cuda.mem_efficient_sdp_enabled()
    except AttributeError:
        return False