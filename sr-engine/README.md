# sr-engine

Super-resolution engine for training and running super-resolution models on
video/image data, with first-class support for both NVIDIA (CUDA) and AMD
(ROCm) GPUs.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Linux (ROCm) or Linux/Windows (CUDA) — CPU-only mode works anywhere

## Quick start (build script)

```bash
# CPU-only (no GPU)
./envs/build.sh --backend cpu

# NVIDIA CUDA
./envs/build.sh --backend cuda

# AMD ROCm
./envs/build.sh --backend rocm
```

The build script creates a `.venv`, installs dependencies with the right PyTorch
index, and runs `envs/verify_env.py` to confirm the setup works.

## Manual install

```bash
uv venv
uv sync --extra cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

Replace `cpu` with `cuda` or `rocm` (and the matching index URL) as needed.

After install, activate the venv or use `uv run srtool ...`.

## Rebuilding

If you delete `.venv/`, just re-run the build script or the `uv venv` + `uv sync`
commands above. The lock file (`uv.lock`) is checked in so resolutions are
deterministic.

## CLI reference

```
srengine [--version] <command> [options]
```

| Command | Subcommand | Description |
|---|---|---|
| `dataset build` | | Build dataset from video or validate preprocessed dir |
| `dataset validate` | | Check an HR/LR dataset directory is well-formed |
| `train run` | | Train a super-resolution model |
| `infer run` | | Run inference on an image or video |
| `model export` | | Export checkpoint to ONNX, safetensors, or TorchScript |
| `model info` | | Print checkpoint summary |
| `env check` | | Detect device, backend, and dtype support |
| `env bench` | | Run micro-benchmark (forward + backward) |

## Project structure

```
configs/         - YAML configuration files
envs/            - Build scripts and Dockerfiles
src/sr_engine/   - Python package
tests/           - Test suite
```
