"""Calibrate VRAM estimation constants against real probe measurements.

Runs the VRAM probe against a sweep of reference configurations and outputs
the measured ``peak_reserved_mb`` and ``peak_allocated_mb`` for each. Use the
output to re-fit ``ACTIVATION_REF_GB`` (model activations only), ``VGG_REF_GB``
(width-independent perceptual branch), ``BATCH_EXP``, ``PATCH_EXP``,
``WIDTH_EXP``, ``CHECKPOINT_FACTOR``, and ``CONTEXT_FLOOR_GB`` in
``vramEstimate.ts``. The estimator targets peak RESERVED memory (the
OOM-driving metric).

The probe's full-image validation term is a batch-1 forward at a single
``patch_size`` tile, matching the trainer's tiled ``_super_resolve_tensor``
pass (which runs on the selected device unless ``validation.offload_cpu`` is
set).

Usage:
    .venv/bin/python scripts/calibrate_vram.py
"""

import json
import sys
from pathlib import Path

# Add the project root to sys.path so we can import sr_engine.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sr_engine.api.vram_probe import run_vram_probe


def probe(label: str, params: dict) -> dict:
    print(f"\n--- {label} ---")
    result = run_vram_probe(params)
    if "error" in result:
        print(f"  ERROR: {result['error']}")
    else:
        print(f"  arch={result.get('arch')}")
        print(f"  dtype={result.get('dtype')}")
        print(f"  checkpointing={result.get('gradient_checkpointing')}")
        print(f"  steps={result.get('num_steps', '?')}")
        print(f"  peak_allocated_mb={result.get('peak_allocated_mb')}")
        print(f"  peak_reserved_mb={result.get('peak_reserved_mb')}")
    return result


def main() -> None:
    if not __import__("torch").cuda.is_available():
        print("CUDA is not available — cannot run probe. Exiting.")
        sys.exit(1)

    results: list[dict] = []

    # ── SwinIR reference configs ──────────────────────────────────────────
    common_swinir = dict(
        num_in_ch=3, num_out_ch=3, embed_dim=180, depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6], window_size=8, mlp_ratio=2.0,
        img_range=1.0, upsampler="pixelshuffle", scale=4,
    )

    # 1a: swinir medium, B=4, patch=64, fp32, no checkpointing (reference)
    r = probe("swinir medium B4 P64 fp32", dict(
        model_name="swinir",
        config=common_swinir,
        batch_size=4,
        patch_size=64,
        dtype="float32",
        scale=4,
        gradient_checkpointing="false",
    ))
    results.append({"label": "swinir_medium_B4_P64_fp32_nockpt", **r})

    # 1b: swinir medium, B=4, patch=64, fp32, auto checkpointing
    r = probe("swinir medium B4 P64 fp32 ckpt", dict(
        model_name="swinir",
        config=common_swinir,
        batch_size=4,
        patch_size=64,
        dtype="float32",
        scale=4,
        gradient_checkpointing="auto",
    ))
    results.append({"label": "swinir_medium_B4_P64_fp32_ckpt", **r})

    # 1c: swinir medium, B=16, patch=128, fp32, auto checkpointing
    r = probe("swinir medium B16 P128 fp32 ckpt", dict(
        model_name="swinir",
        config=common_swinir,
        batch_size=16,
        patch_size=128,
        dtype="float32",
        scale=4,
        gradient_checkpointing="auto",
    ))
    results.append({"label": "swinir_medium_B16_P128_fp32_ckpt", **r})

    # 1d: swinir medium, B=16, patch=128, fp32, no checkpointing
    r = probe("swinir medium B16 P128 fp32 nockpt", dict(
        model_name="swinir",
        config=common_swinir,
        batch_size=16,
        patch_size=128,
        dtype="float32",
        scale=4,
        gradient_checkpointing="false",
    ))
    results.append({"label": "swinir_medium_B16_P128_fp32_nockpt", **r})

    # ── RRDB reference configs ────────────────────────────────────────────
    common_rrdb = dict(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
        num_grow_ch=32, scale=4,
    )

    r = probe("rrdb B4 P64 fp32 nockpt", dict(
        model_name="rrdb_esrgan",
        config=common_rrdb,
        batch_size=4,
        patch_size=64,
        dtype="float32",
        scale=4,
        gradient_checkpointing="false",
    ))
    results.append({"label": "rrdb_B4_P64_fp32_nockpt", **r})

    r = probe("rrdb B4 P64 fp32 ckpt", dict(
        model_name="rrdb_esrgan",
        config=common_rrdb,
        batch_size=4,
        patch_size=64,
        dtype="float32",
        scale=4,
        gradient_checkpointing="auto",
    ))
    results.append({"label": "rrdb_B4_P64_fp32_ckpt", **r})

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n\n=== CALIBRATION SUMMARY ===")
    for entry in results:
        if "error" in entry:
            print(f"  {entry['label']}: ERROR {entry['error']}")
        else:
            rat = entry.get("peak_reserved_mb", 0) / max(entry.get("peak_allocated_mb", 1), 1)
            print(
                f"  {entry['label']}: "
                f"alloc={entry.get('peak_allocated_mb', '?')} MB  "
                f"resv={entry.get('peak_reserved_mb', '?')} MB  "
                f"ratio={rat:.2f}"
            )

    # Save raw results for refitting
    out_path = Path("calibration_results.json")
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()