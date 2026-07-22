"""Checkpoint save/load, EMA state handling, and export utilities."""

import logging
from pathlib import Path

import torch

from .registry import build_model

log = logging.getLogger(__name__)


def check_shape_compat(state_dict: dict, model_sd: dict, model_name: str = "model") -> None:
    """Raise ``ValueError`` if any shared key has a different shape.

    Args:
        state_dict: Checkpoint state dictionary.
        model_sd: Current model state dictionary.
        model_name: Name of the model (for error messages).
    """
    for k in list(state_dict.keys()):
        if k in model_sd and state_dict[k].shape != model_sd[k].shape:
            raise ValueError(
                f"Key '{k}' shape {state_dict[k].shape} in checkpoint does not "
                f"match {model_name} shape {model_sd[k].shape}. This checkpoint "
                "was saved from a different architecture version."
            )


def strip_keys_not_in_model(state_dict: dict, model_sd: dict,
                            prefixes: tuple[str, ...]) -> dict:
    """Remove keys from *state_dict* that start with *prefixes* but don't exist in *model_sd*.

    Args:
        state_dict: Checkpoint state dictionary (modified in place).
        model_sd: Current model state dictionary.
        prefixes: Module name prefixes to check (e.g. ``("upsampler.",)``).

    Returns:
        Cleaned state dictionary.
    """
    cleaned = dict(state_dict)
    for k in list(cleaned.keys()):
        if any(k.startswith(p) for p in prefixes) and k not in model_sd:
            del cleaned[k]
            log.info("Discarded key '%s' not in current model", k)
    return cleaned


def compat_load_state_dict(
    model: torch.nn.Module,
    state_dict: dict,
    strict: bool = True,
    compat_prefixes: tuple[str, ...] | None = None,
    optional_buffers: list[str] | None = None,
) -> tuple[dict, bool]:
    """Clean *state_dict* for backward-compatible loading.

    Strips keys under *compat_prefixes* not in *model*, checks shape
    compatibility, and handles missing *optional_buffers*.

    Returns ``(cleaned_state_dict, effective_strict)`` so the caller can
    pass them to ``model.load_state_dict()`` — this avoids recursion when
    the model has its own ``load_state_dict`` override.

    Args:
        model: Target model instance.
        state_dict: Checkpoint state dictionary.
        strict: Whether to enforce strict key matching.
        compat_prefixes: Module name prefixes whose unknown keys are
            safe to discard (e.g. ``("upsampler.",)``).
        optional_buffers: Buffer names that may be absent from older
            checkpoints (e.g. ``["rgb_mean"]``). When strict and a buffer
            is missing, ``strict`` is downgraded to ``False``.

    Returns:
        ``(cleaned_state_dict, effective_strict)``.
    """
    model_sd = model.state_dict()

    if compat_prefixes:
        state_dict = strip_keys_not_in_model(state_dict, model_sd, compat_prefixes)

    check_shape_compat(state_dict, model_sd, type(model).__name__)

    optional_buffers = optional_buffers or []
    if strict and any(b in model_sd and b not in state_dict for b in optional_buffers):
        strict = False

    return state_dict, strict


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
        config: The resolved model architecture configuration (must include
            a ``"name"`` key resolvable via the model registry), saved for
            reproducibility and so export utilities can rebuild the model.
        backend_info: Device/backend metadata string.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": state_dict,
        "ema_state_dict": ema_state_dict,
        "optimizer_state": optimizer_state,
        "step": step,
        "config": config,
        "backend_info": backend_info,
    }

    # Write to a temp file first and rename, so a crash/interrupt mid-write
    # can't leave a corrupted checkpoint at *path*.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


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
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        # Broad except — intentional across PyTorch versions.
        # Some PyTorch builds raise RuntimeError for weights-only
        # incompatibility, while others raise UnpicklingError or TypeError
        # (especially on Windows or with older torch versions). Catching
        # Exception here lets us retry with weights_only=False without a
        # version-gated dispatch.
        import warnings
        warnings.warn(
            f"Loading checkpoint with weights_only=False at '{path}'. "
            "This is unsafe for untrusted sources. Re-save with a current "
            "PyTorch version to silence this warning.",
            FutureWarning, stacklevel=2,
        )
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if load_ema:
        ema_state = checkpoint.get("ema_state_dict")
        if ema_state is None:
            raise ValueError(
                f"Checkpoint at '{path}' has no EMA state dict, but load_ema=True was requested."
            )
        checkpoint = {**checkpoint, "state_dict": ema_state}

    return checkpoint


def _build_model_from_checkpoint(checkpoint: dict) -> torch.nn.Module:
    """Rebuild a model instance from a loaded checkpoint dict and load its weights."""
    config = checkpoint.get("config")
    if not config or "name" not in config:
        raise ValueError(
            "Checkpoint has no usable 'config' (with a 'name' key) — cannot "
            "reconstruct the model architecture for export."
        )

    model = build_model(config["name"], config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def export_to_safetensors(
    checkpoint_path: Path,
    output_path: Path,
    load_ema: bool = False,
) -> None:
    """Export a PyTorch checkpoint to safetensors format."""
    from safetensors.torch import save_file

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", load_ema=load_ema)
    state_dict = checkpoint["state_dict"]

    # safetensors requires contiguous tensors on CPU.
    state_dict = {k: v.contiguous().cpu() for k, v in state_dict.items()}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(output_path))


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    input_shape: tuple[int, ...] = (1, 3, 256, 256),
) -> None:
    """Export a model to ONNX format via ``torch.onnx.export``."""
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model = _build_model_from_checkpoint(checkpoint)

    dummy_input = torch.randn(*input_shape)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
    )


def export_to_torchscript(
    checkpoint_path: Path,
    output_path: Path,
) -> None:
    """Export a model to TorchScript via ``torch.jit.trace``."""
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model = _build_model_from_checkpoint(checkpoint)

    # A reasonably generic trace input; SR models are typically fully
    # convolutional so this shape doesn't constrain inference-time inputs.
    dummy_input = torch.randn(1, 3, 256, 256)

    traced = torch.jit.trace(model, dummy_input)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))