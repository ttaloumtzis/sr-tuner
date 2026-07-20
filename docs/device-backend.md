# Device Backend

## Overview

The device abstraction layer provides a uniform interface for CUDA and ROCm backends. It handles detection, dtype selection, and kernel dispatch transparently — training and inference code never need to check the backend explicitly.

## Device Detection

`device/backend.py:get_device()`

```python
def get_device(preferred: str = "auto") -> torch.device:
```

Resolution logic:

| `preferred` | CUDA available | ROCm detected | Result |
|-------------|---------------|---------------|--------|
| `"auto"` | Yes | No | `cuda:0` |
| `"auto"` | Yes | Yes | `cuda:0` (ROCm reports as CUDA device) |
| `"auto"` | No | — | `cpu` |
| `"cuda"` | Yes | — | `cuda:0` |
| `"cuda"` | No | — | Error |
| `"cpu"` | — | — | `cpu` |

### Backend Detection Functions

```python
def is_rocm() -> bool:
    """True if PyTorch was built with ROCm support."""
    return hasattr(torch.version, 'hip') and torch.version.hip is not None

def get_device_name() -> str:
    """GPU name string (e.g., 'NVIDIA GeForce RTX 4090', 'AMD Radeon RX 7900 XTX')."""

def get_vram() -> int:
    """Total VRAM in MB."""

def get_vram_used() -> int:
    """Currently used VRAM in MB."""
```

## Mixed Precision

`device/backend.py:autocast_dtype()`

```python
def autocast_dtype(device: torch.device) -> torch.dtype:
    """Returns bfloat16 if supported, float16 otherwise."""
```

Returns the optimal autocast dtype for the device:

| Backend | BF16 support | Autocast dtype |
|---------|-------------|----------------|
| CUDA (compute 8.0+): A100, H100 | Yes | `bfloat16` |
| CUDA (compute 7.0+): V100, RTX 30xx | Partial | `bfloat16` |
| CUDA (compute <7.0): GTX 10xx | No | `float16` |
| ROCm (MI200+) | Yes | `bfloat16` |
| ROCm (older) | No | `float16` |
| CPU | — | `float32` |

### Usage in Training

```python
device = get_device("auto")
dtype = autocast_dtype(device)

with torch.amp.autocast(device_type=device.type, dtype=dtype):
    sr = model(lr)
    loss = criterion(sr, hr)
```

## Flash Attention

`device/backend.py:supports_flash_attn()`

```python
def supports_flash_attn(device: torch.device) -> bool:
    """Check if PyTorch's scaled_dot_product_attention uses Flash Attention."""
```

Detection checks:
- CUDA compute capability >= 8.0 (Ampere or newer)
- PyTorch version >= 2.0
- ROCm: requires ROCm 5.7+ and PyTorch built with ROCm support

Flash Attention accelerates SwinIR's self-attention by 2-3× on compatible hardware without numerical changes.

## Backend-Aware Kernels

`device/kernels.py` provides backend-specific implementations:

### Scaled Dot-Product Attention

```python
def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0):
    """Uses PyTorch 2.0's native SDPA if available, falls back manually."""
    if HAS_TORCH_SDPA:
        return F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    else:
        # Manual implementation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if attn_mask is not None:
            scores += attn_mask
        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, dropout_p)
        return torch.matmul(attn, v)
```

### Convolution

```python
def get_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    """Returns standard Conv2d (no backend-specific variant needed currently)."""
    return nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
```

Currently both backends use PyTorch's standard `nn.Conv2d`. The indirection exists for future integration with backend-specific libraries like `torch.backends.cudnn` or `torch.backends.mkldnn`.

## Environment Diagnostics

### `env check`

`cli/cmd_env.py:cmd_env_check()` collects and displays system information:

```
PyTorch version:  2.5.0+cu124
Detected device:  cuda:0
CUDA/ROCm avail:  True
Device name:      NVIDIA GeForce RTX 4090
VRAM total:       24564 MB
VRAM used:        1024 MB
BF16 support:     True
ROCm backend:     False
Autocast dtype:   torch.bfloat16
Flash attention:  True
```

Used for:
- Verifying the correct PyTorch build is installed (CUDA vs ROCm)
- Checking VRAM availability before starting training
- Debugging installation issues

### `env bench`

`cli/cmd_env.py:cmd_env_bench()` runs a micro-benchmark:

```python
# Pseudo-code
model = build_model(model_name, config).to(device)
dummy_input = torch.randn(batch_size, 3, 128, 128).to(device)
dtype = autocast_dtype(device)

for _ in range(iterations):
    with torch.amp.autocast(device_type=device.type, dtype=dtype):
        output = model(dummy_input)
        loss = output.sum()
    loss.backward()
```

Reports average forward+backward time per iteration. Useful for comparing:
- RRDB vs SwinIR throughput on the same hardware
- CUDA vs ROCm performance for the same model
- Effect of batch size and input resolution on throughput

## Build Backends

The `envs/build.sh` script configures PyTorch for the target backend:

| `--backend` | PyTorch source | CUDA/ROCm version |
|-------------|---------------|-------------------|
| `cpu` | PyPI CPU-only | None |
| `cuda` | PyPI CUDA | CUDA 12.4 |
| `rocm` | PyPI ROCm | ROCm 6.2 |

### Platform Support

| Backend | Linux | macOS | Windows |
|---------|-------|-------|---------|
| CPU     | ✓     | ✓     | ✓       |
| CUDA    | ✓     | —     | ✓       |
| ROCm    | ✓     | —     | —       |

- **ROCm is Linux-only.** Use the CUDA or CPU backend on Windows.
- On Windows, use `.\envs\build.ps1 -Backend cpu` or `.\envs\build.ps1 -Backend cuda`.

Verification via `envs/verify_env.py`:
- Checks PyTorch imports without error
- Validates CUDA/ROCm availability matches the requested backend
- Runs a tiny forward pass to confirm device execution
