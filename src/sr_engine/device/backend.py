"""Backend detection and capability flags for CUDA and ROCm."""

import torch

def get_device_name() -> str:
    """Detect the active compute device type.

    PyTorch reports both NVIDIA CUDA and AMD ROCm devices as ``"cuda"``.
    On systems without a GPU, returns ``"cpu"``.

    Returns:
        ``"cuda"`` if a GPU (NVIDIA or AMD) is available, ``"cpu"`` otherwise.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device() -> torch.device:
    """Return a ``torch.device`` for the active compute backend.

    Returns:
        ``torch.device("cuda")`` or ``torch.device("cpu")``.
    """
    return torch.device(get_device_name())


def is_rocm() -> bool:
    """Check whether the active CUDA device uses an AMD ROCm driver stack.

    Returns:
        ``True`` if ``torch.version.hip`` is set (indicating a ROCm build),
        ``False`` for NVIDIA CUDA or CPU-only.
    """
    if not torch.cuda.is_available():
        return False
    return getattr(torch.version, "hip", None) is not None


def autocast_dtype() -> torch.dtype:
    """Return the recommended automatic mixed-precision dtype.

    Uses ``bfloat16`` if the GPU supports it (both ROCm and CUDA),
    falls back to ``float16``, or ``float32`` on CPU-only systems.

    Returns:
        ``torch.bfloat16``, ``torch.float16``, or ``torch.float32``.
    """
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def supports_flash_attn() -> bool:
    """Check whether PyTorch's native scaled dot-product attention is available.

    Verifies that the CUDA backend is initialised and that at least one SDPA
    kernel (FlashAttention or Memory-Efficient Attention) is enabled.

    Returns:
        ``True`` if SDPA dispatch is available, ``False`` otherwise.
    """
    if not hasattr(torch.backends, 'cuda') or not torch.cuda.is_available():
        return False
    try:
        return (
            torch.backends.cuda.flash_sdp_enabled()
            or torch.backends.cuda.mem_efficient_sdp_enabled()
        )
    except AttributeError:
        return False