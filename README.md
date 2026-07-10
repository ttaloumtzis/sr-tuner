# sr-tuner

Super-resolution training toolkit for video and image upscaling. Provides a complete pipeline from raw video to trained models, with first-class GPU support for both NVIDIA (CUDA) and AMD (ROCm).

## Components

| Component | Description |
|-----------|-------------|
| **sr-engine** | Core Python package: dataset building, training, inference, model export, GUI bridge |
| [sr-engine/README.md](sr-engine/README.md) | Setup and quick start |

## Quick Setup

```bash
cd sr-engine
./envs/build.sh --backend cpu     # CPU-only
# or ./envs/build.sh --backend cuda   # NVIDIA CUDA
# or ./envs/build.sh --backend rocm   # AMD ROCm

uv run srengine env check          # Verify installation
```

Requirements: Python 3.11, [uv](https://docs.astral.sh/uv/), Linux (ROCm) or Linux/Windows (CUDA).

## Quick Start

```bash
cd sr-engine
uv run srengine workspace init                        # Set up workspace
uv run srengine dataset build --input video.mp4        # Build dataset
uv run srengine train run --dataset my_set --model swinir  # Train model
uv run srengine infer run --model model.pth --input image.png --output sr.png  # Upscale
```

## Documentation

| Topic | Location |
|-------|----------|
| Architecture & design patterns | [sr-engine/docs/architecture.md](sr-engine/docs/architecture.md) |
| CLI reference (all commands) | [sr-engine/docs/cli-reference.md](sr-engine/docs/cli-reference.md) |
| Training deep dive | [sr-engine/docs/training.md](sr-engine/docs/training.md) |
| Data pipeline | [sr-engine/docs/data-pipeline.md](sr-engine/docs/data-pipeline.md) |
| Inference & export | [sr-engine/docs/inference.md](sr-engine/docs/inference.md) |
| Workspace management | [sr-engine/docs/workspace.md](sr-engine/docs/workspace.md) |
| Device backend (CUDA/ROCm) | [sr-engine/docs/device-backend.md](sr-engine/docs/device-backend.md) |
| GUI bridge protocol | [sr-engine/docs/gui_bridge.md](sr-engine/docs/gui_bridge.md) |
| Development guide | [sr-engine/docs/development.md](sr-engine/docs/development.md) |

## Supported Models

| Model | Architecture | Type |
|-------|-------------|------|
| `rrdb_esrgan` | RRDBNet (23 RRDB blocks, 64 channels) | CNN-based |
| `swinir` | SwinIR (6-stage, 6-head, 180-dim embed) | Transformer-based |
