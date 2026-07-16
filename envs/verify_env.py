#!/usr/bin/env python
"""Verify the sr-engine environment: device, backend, dtype support."""

import sys
import torch
from sr_engine.device.backend import get_device, is_rocm, autocast_dtype, supports_flash_attn


def main() -> None:
    """Print environment diagnostics and run a micro forward/backward pass."""
    print("=" * 60)
    print("sr-engine environment verification")
    print("=" * 60)

    try:
        device = get_device()
        print(f"  PyTorch version:  {torch.__version__}")
        print(f"  Detected device:  {device}")
        print(f"  CUDA available:   {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device name:      {torch.cuda.get_device_name(device)}")
        print(f"  ROCm backend:     {is_rocm()}")
        print(f"  Autocast dtype:   {autocast_dtype()}")
        print(f"  Flash attention:  {supports_flash_attn()}")
        print(f"  BF16 support:     {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")

        if torch.cuda.is_available():
            print("\n  Running micro fwd/bwd pass...")
            x = torch.randn(2, 3, 64, 64, device=device)
            conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
            out = conv(x)
            loss = out.sum()
            loss.backward()
            print("  Forward+backward: OK")

        print("\n  Status: PASS")
    except Exception as e:
        print(f"\n  Status: FAIL — {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
