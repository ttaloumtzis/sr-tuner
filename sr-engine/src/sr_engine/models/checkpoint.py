"""Checkpoint save/load, EMA state handling, and export utilities."""

from pathlib import Path


def save_checkpoint(
    path: Path,
    state_dict: dict,
    ema_state_dict: dict | None = None,
    optimizer_state: dict | None = None,
    step: int = 0,
    config: dict | None = None,
    backend_info: dict | None = None,
) -> None:
    """Save a training checkpoint to *path*.

    Args:
        path: Destination file path (``.pt`` or ``.safetensors``).
        state_dict: Model state dictionary.
        ema_state_dict: Optional EMA model state dictionary.
        optimizer_state: Optional optimizer state dictionary.
        step: Global training step at save time.
        config: The resolved training configuration for reproducibility.
        backend_info: Device/backend metadata string.
    """
    raise NotImplementedError("TODO: implement save_checkpoint")


def load_checkpoint(
    path: Path,
    map_location: str | None = None,
    load_ema: bool = False,
) -> dict:
    """Load a checkpoint and return its contents as a dict.

    Args:
        path: Checkpoint file path.
        map_location: Torch device string to map tensors to.
        load_ema: If True, load the EMA state dict instead of the live one.

    Returns:
        A dict with at least ``state_dict``, ``step``, and ``config`` keys.
    """
    raise NotImplementedError("TODO: implement load_checkpoint")


def export_to_safetensors(
    checkpoint_path: Path,
    output_path: Path,
    load_ema: bool = False,
) -> None:
    """Export a PyTorch checkpoint to safetensors format."""
    raise NotImplementedError("TODO: implement safetensors export")


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    input_shape: tuple[int, ...] = (1, 3, 256, 256),
) -> None:
    """Export a model to ONNX format via ``torch.onnx.export``."""
    raise NotImplementedError("TODO: implement ONNX export")


def export_to_torchscript(
    checkpoint_path: Path,
    output_path: Path,
) -> None:
    """Export a model to TorchScript via ``torch.jit.trace``."""
    raise NotImplementedError("TODO: implement TorchScript export")
