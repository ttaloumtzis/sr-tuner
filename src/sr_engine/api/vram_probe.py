"""VRAM probe — measure real peak GPU memory for a given training configuration.

Runs a dry forward + loss + backward on a dummy batch in an isolated subprocess
so the measurement never touches a running training context and never leaves a
CUDA context behind in the main process.

Multiple consecutive dry steps, allocator tuning identical to the real training
worker, and a validation-phase no_grad forward are all included so the reported
``peak_reserved_mb`` reflects the OOM-driving metric (reserved, not allocated)
and captures fragmentation + MIOpen/cuDNN workspace that a single-step probe
misses.
"""

import multiprocessing
from typing import Any

import torch

from sr_engine.models.losses import build_composite_loss
from sr_engine.models.registry import build_model

_MB = 1024 * 1024
_N_STEPS = 8


def _apply_allocator_tuning() -> None:
    """Apply best-effort GPU allocator and cuDNN settings (mirrors workers.py)."""
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            from torch import version as torch_ver
            ver = tuple(int(x) for x in torch_ver.__version__.split("+")[0].split("."))
            if ver >= (2, 1):
                torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    except Exception:
        pass


def _probe_worker(params: dict, result_queue: multiprocessing.Queue) -> None:
    """Run the measurement in a spawned subprocess and put the result on a queue."""
    try:
        if not torch.cuda.is_available():
            result_queue.put({"error": "CUDA is not available on this machine"})
            return

        result = _measure(params)
        result_queue.put(result)
    except Exception as e:  # noqa: BLE001
        result_queue.put({"error": f"{type(e).__name__}: {e}"})
    finally:
        torch.cuda.empty_cache()


def _run_step(
    model: torch.nn.Module,
    lr: torch.Tensor,
    hr: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    amp_dtype: Any,
    amp_enabled: bool,
    loss_bf16: bool,
    grad_scaler: Any,
) -> None:
    """One full training step: forward + loss + backward + optimizer step."""
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        pred = model(lr)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=loss_bf16):
        loss, _ = loss_fn(pred, hr)
    if grad_scaler is not None:
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
    else:
        loss.backward()
        optimizer.step()


def _measure(params: dict) -> dict[str, Any]:
    """Run dry training steps + validation forward, return peak allocated/reserved."""
    _apply_allocator_tuning()

    model_name = params["model_name"]
    config = dict(params.get("config") or {})
    batch_size = int(params.get("batch_size", 1))
    patch_size = int(params.get("patch_size", 64))
    dtype_str = str(params.get("dtype", "float32")).lower()
    loss_config = params.get("loss_config")
    device = torch.device("cuda")

    if "name" not in config:
        config["name"] = model_name

    model = build_model(model_name, config).to(device)
    model.train()

    gc = str(params.get("gradient_checkpointing", "auto")).lower()
    if gc == "auto":
        gc_enabled = model_name == "swinir"
    else:
        gc_enabled = gc in ("true", "1", "yes")
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = gc_enabled

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4,
        betas=[0.9, 0.99],
    )

    loss_fn = build_composite_loss(loss_config, device)

    scale = int(config.get("scale", 4))
    num_in_ch = int(config.get("num_in_ch", 3))
    num_out_ch = int(config.get("num_out_ch", 3))
    lr = torch.rand(batch_size, num_in_ch, patch_size, patch_size, device=device)
    hr = torch.rand(batch_size, num_out_ch, patch_size * scale, patch_size * scale, device=device)

    amp_dtype = None
    if dtype_str == "bf16" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif dtype_str == "float16":
        amp_dtype = torch.float16
    amp_enabled = amp_dtype is not None
    grad_scaler = torch.amp.GradScaler() if amp_dtype == torch.float16 else None

    loss_bf16 = (
        amp_dtype is None
        and torch.cuda.is_bf16_supported()
        and str(params.get("loss_bf16", "true")).lower() in ("true", "1", "yes")
    )

    # Warmup so MIOpen/cuDNN autotuning (and its workspace) settles before we
    # sample the peak — otherwise the measured number is inflated.
    with torch.no_grad():
        model(lr)

    # --- Training-phase measurement: N consecutive dry steps ---
    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0

    for _ in range(_N_STEPS):
        torch.cuda.reset_peak_memory_stats()
        _run_step(model, lr, hr, optimizer, loss_fn, amp_dtype, amp_enabled, loss_bf16, grad_scaler)
        step_alloc = torch.cuda.max_memory_allocated() / _MB
        step_resv = torch.cuda.max_memory_reserved() / _MB
        if step_alloc > peak_allocated_mb:
            peak_allocated_mb = step_alloc
        if step_resv > peak_reserved_mb:
            peak_reserved_mb = step_resv

    # --- Validation-phase measurement: no_grad forward ---
    # Replicates the split-case validation loop: a forward at the training
    # batch size on CenterCrop'd patches + a batch-1 full-image forward.
    model.eval()
    with torch.no_grad():
        # Patch-based PSNR/SSIM style: forward at same batch/patch as training.
        torch.cuda.reset_peak_memory_stats()
        val_lr = lr.clone()
        _ = model(val_lr)
        val_alloc = torch.cuda.max_memory_allocated() / _MB
        val_resv = torch.cuda.max_memory_reserved() / _MB
        if val_alloc > peak_allocated_mb:
            peak_allocated_mb = val_alloc
        if val_resv > peak_reserved_mb:
            peak_reserved_mb = val_resv

        # Full-image val style: batch-1 forward at a real tiled-pass tile.
        # tile_size = patch_size, matching the trainer's _super_resolve_tensor.
        full_lr = torch.rand(1, num_in_ch, patch_size, patch_size, device=device)
        torch.cuda.reset_peak_memory_stats()
        _ = model(full_lr)
        full_alloc = torch.cuda.max_memory_allocated() / _MB
        full_resv = torch.cuda.max_memory_reserved() / _MB
        if full_alloc > peak_allocated_mb:
            peak_allocated_mb = full_alloc
        if full_resv > peak_reserved_mb:
            peak_reserved_mb = full_resv

    model.train()

    del model, optimizer, loss_fn, lr, hr, val_lr, full_lr
    torch.cuda.empty_cache()

    return {
        "arch": model_name,
        "dtype": dtype_str,
        "gradient_checkpointing": gc_enabled,
        "num_steps": _N_STEPS,
        "peak_allocated_mb": round(peak_allocated_mb, 1),
        "peak_reserved_mb": round(peak_reserved_mb, 1),
    }


def run_vram_probe(params: dict) -> dict[str, Any]:
    """Run the probe in a spawned subprocess and return its result dict."""
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_probe_worker, args=(params, result_queue), daemon=True)
    proc.start()
    proc.join(timeout=300)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"error": "VRAM probe timed out"}
    if result_queue.empty():
        return {"error": f"VRAM probe exited with code {proc.exitcode}"}
    return result_queue.get()
