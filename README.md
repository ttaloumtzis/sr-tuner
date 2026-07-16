# sr-engine

Super-resolution engine for training and running super-resolution models on video/image data, with first-class support for both NVIDIA (CUDA) and AMD (ROCm) GPUs.

## Requirements

- Python 3.11 (strictly pinned)
- [uv](https://docs.astral.sh/uv/) (package manager)
- Linux (ROCm) or Linux/Windows (CUDA) — CPU-only mode works anywhere

## Quick Setup

```bash
# CPU-only (no GPU)
./envs/build.sh --backend cpu

# NVIDIA CUDA
./envs/build.sh --backend cuda

# AMD ROCm
./envs/build.sh --backend rocm
```

One command creates `.venv`, pins the right PyTorch index, installs deps, and runs `envs/verify_env.py` to confirm everything works.

Manual alternative:
```bash
uv venv
uv sync
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

## Quick Start

```bash
# Initialize a workspace
srengine workspace init

# Build a dataset from video
srengine dataset build --input video.mp4

# Train an RRDB model
srengine train run --dataset ./datasets/my_set --model rrdb_esrgan

# Run inference on an image
srengine infer run --model checkpoints/model.pth --input input.png --output output.png
```

## What's Next?

| Topic | Doc |
|---|---|
| System architecture and design patterns | [docs/architecture.md](docs/architecture.md) |
| Full CLI reference | [docs/cli-reference.md](docs/cli-reference.md) |
| Training deep dive (trainer, models, metrics, config) | [docs/training.md](docs/training.md) |
| Data pipeline (dataset build, degradation, validation) | [docs/data-pipeline.md](docs/data-pipeline.md) |
| Inference and model export | [docs/inference.md](docs/inference.md) |
| Workspace and project management | [docs/workspace.md](docs/workspace.md) |
| Device abstraction (CUDA/ROCm) | [docs/device-backend.md](docs/device-backend.md) |
| GUI bridge protocol | [docs/gui_bridge.md](docs/gui_bridge.md) |
| Development guide (testing, adding models, etc.) | [docs/development.md](docs/development.md) |

## Project Structure

```
envs/                 - Build scripts and Dockerfiles
src/sr_engine/        - Python package
  cli/                - Click CLI commands
  data/               - Dataset building, validation, transforms, degradation
  device/             - CUDA/ROCm backend abstraction
  engine/             - Inference, trainer, metrics, tiling
  models/             - RRDB, SwinIR, checkpointing, losses, registry
  utils/              - Config loader, I/O, logging
    configs/          - YAML configuration files
tests/                - Test suite
```
