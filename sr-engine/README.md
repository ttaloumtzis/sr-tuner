# sr-engine

Super-resolution engine for training and running super-resolution models on
video/image data, with first-class support for both NVIDIA (CUDA) and AMD
(ROCm) GPUs.

## Requirements

- Python 3.11 (strictly pinned)
- [uv](https://docs.astral.sh/uv/) (package manager)
- Linux (ROCm) or Linux/Windows (CUDA) — CPU-only mode works anywhere

## Setup

Two paths, same result — choose based on how much control you want.

### Quick start (build script)

```bash
# CPU-only (no GPU)
./envs/build.sh --backend cpu

# NVIDIA CUDA
./envs/build.sh --backend cuda

# AMD ROCm
./envs/build.sh --backend rocm
```

One command. Creates `.venv`, pins the right PyTorch index, installs deps,
and runs `envs/verify_env.py` to confirm everything works. Re-run any time.

### Manual install

```bash
uv venv
uv sync
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

Replace the index URL for CUDA or ROCm as needed.
Use this when you want full control or need to debug dependency issues.

| Approach | Speed | Control | Best for |
|---|---|---|---|
| `build.sh` | Fast | Low | First-time setup, CI, rebuilds |
| `uv sync` | Medium | High | Debugging, custom extras, CI |

After install, activate the venv or use `uv run srengine ...`.

> **Standalone aliases vs `srengine` umbrella:** The pyproject.toml also registers
> `dataset`, `env`, `infer`, `model`, `train`, `workspace`, and `project` as
> standalone entry points. These bypass the `srengine` parent group entirely —
> they auto-detect the workspace from CWD and ignore the `--workspace` global flag.
> Use `srengine <cmd>` when you need explicit `--workspace PATH` control. Use the
> standalone aliases for convenience when already inside a workspace directory.

## CLI reference

```
srengine [--version] [--workspace PATH] <command> [options]
```

### Commands overview

| Command | Subcommand | Description |
|---|---|---|
| `dataset build` | | Build dataset from video or validate preprocessed dir |
| `dataset validate` | | Run deep validation on an existing dataset |
| `dataset health` | | Profile resolution, aspect ratio, black-frame distribution |
| `train run` | | Train a model (RRDB or SwinIR) |
| `infer run` | | Run inference on an image or video |
| `model export` | | Export checkpoint to ONNX / safetensors / TorchScript |
| `model info` | | Print checkpoint summary |
| `env check` | | Detect device, backend, dtype, flash-attn support |
| `env bench` | | Micro-benchmark (forward + backward pass) |

### Detailed usage

#### `dataset build` — video vs preprocessed

```bash
# From raw video — extracts frames, applies degradation
srengine dataset build --input video.mp4 --output ./datasets/my_set

# From an existing preprocessed folder — re-validates + relinks
srengine dataset build --input ./datasets/my_set
```

| Source | `--out` required | Behaviour |
|---|---|---|
| Video file (`.mp4`, `.avi`, etc.) | No (inside workspace) | Auto-resolves to `<ws>/datasets/<stem>/` |
| Video file (`.mp4`, `.avi`, etc.) | Yes (outside workspace) | Extract frames → apply degradation → write HR/LR pairs |
| Directory (existing dataset) | No | Validate structure, rebuild manifest |

Behind the scenes, the pipeline reads the dataset config YAML
(`src/sr_engine/utils/configs/datasets/video_pairs.yaml`) and applies the
degradation model defined there (blur → noise → downscale → JPEG compression).

#### `dataset validate` — structural integrity

```bash
srengine dataset validate --path ./datasets/my_set
```

Checks that `HR/` and `LR/` exist, image sizes match, and every file is readable.
Reports total valid pairs and any problems found.

#### `dataset health` — data quality

```bash
srengine dataset health --path ./datasets/my_set
srengine dataset health --path ./datasets/my_set --yes   # auto-delete black frames
```

Profiles every frame: resolution breakdown, aspect ratios, color spaces.
Detects completely black frames and optionally prunes them from the manifest
and filesystem. Without `--yes`, prompts before deletion.

#### `train run` — CLI flags vs YAML config

```bash
# Minimal — uses default train config + CLI overrides
srengine train run --dataset ./datasets/my_set --model swinir

# Explicit config file — full control
srengine train run \
  --config ./src/sr_engine/utils/configs/train/base.yaml \
  --dataset ./datasets/my_set \
  --model rrdb_esrgan

# CLI overrides take precedence over the config file
srengine train run \
  --dataset ./datasets/my_set \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --max-epochs 20 \
  --validation-split 0.15
```

| Parameter | Config file key | CLI flag |
|---|---|---|
| Batch size | `batch_size` | `--batch-size` |
| Learning rate | `learning_rate` | `--learning-rate` |
| Max epochs | `max_epochs` | `--max-epochs` |
| Validation split | `validation.split` | `--validation-split` |
| Validation on/off | `validation.enabled` | `--validation-enabled / --no-validation-enabled` |

Defaults from `src/sr_engine/utils/configs/train/base.yaml` if no config file is given.

#### Available models

| Model name | Architecture | Config |
|---|---|---|
| `rrdb_esrgan` | RRDBNet (Residual-in-Residual Dense Block) | `src/sr_engine/utils/configs/models/rrdb_esrgan.yaml` |
| `swinir` | SwinIR (Swin Transformer) | `src/sr_engine/utils/configs/models/swinir.yaml` |

RRDB is a CNN-based architecture (23 RRDB blocks, 64 feature channels).
SwinIR is a transformer-based architecture (6-stage, 6-head, 180 embed dim).
Use `srengine env bench --model swinir` or `--model rrdb_esrgan` to compare
throughput on your hardware.

#### `infer run` — image vs video

```bash
# Single image
srengine infer run \
  --model checkpoints/model.pth \
  --input input.png \
  --output output.png

# Video — processes every frame
srengine infer run \
  --model checkpoints/model.pth \
  --input video.mp4 \
  --output upscaled.mp4 \
  --tile 512 --overlap 64
```

| Input type | Detected by | Behaviour |
|---|---|---|
| Image (`.png`, `.jpg`, etc.) | File suffix | Single-frame super-resolution |
| Video (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`) | File suffix | Frame-by-frame upscaling |

Tiling (`--tile`, `--overlap`) trades VRAM for speed. Tiling off (`--tile 0`)
is fastest on high-VRAM GPUs. Tiling on (`--tile 512 --overlap 64`) avoids
OOM on 8 GB cards.

#### `model export` — three formats

```bash
# ONNX
srengine model export --model-name swinir --ckpt model.pth --format onnx --out model.onnx

# SafeTensors
srengine model export --model-name rrdb_esrgan --ckpt model.pth --format safetensors --out model.safetensors

# TorchScript
srengine model export --model-name swinir --ckpt model.pth --format torchscript --out model.pt

# Short form
srengine model export -m swinir -c model.pth -f onnx -o model.onnx
```

| Format | Use case |
|---|---|
| `onnx` | Cross-platform inference, ONNX Runtime |
| `safetensors` | Safe weight distribution, HuggingFace |
| `torchscript` | C++ `libtorch` deployment, no-Python inference |

#### `model info` — checkpoint inspection

```bash
srengine model info --model model.pth
# Checkpoint: model.pth
#   Step:      45000
#   Config:    {'name': 'swinir', 'type': 'swinir', ...}
```

#### `env check` — hardware detection

```bash
srengine env check
# PyTorch version:  2.5.0+cu124
# Detected device:  cuda:0
# CUDA/ROCm avail:  True
# Device name:      NVIDIA GeForce RTX 4090
# VRAM total:       24564 MB
# BF16 support:     True
# ROCm backend:     False
# Autocast dtype:   torch.bfloat16
# Flash attention:  True
```

Useful in CI or when choosing between CUDA/ROCm builds.

#### `env bench` — model throughput comparison

```bash
# Benchmark RRDB on default device
srengine env bench

# Benchmark SwinIR explicitly
srengine env bench --model swinir
```

Runs a forward+backward pass on a 128×128 dummy batch. Use to compare
RRDB vs SwinIR throughput, or CUDA vs ROCm performance.

Exits with code 0 on success, code 1 on failure (e.g. model not found).

### Configuration layering

Configs resolve in this order (later wins):

1. **Built-in defaults** — `src/sr_engine/utils/configs/default.yaml`
2. **Module defaults** — `train/base.yaml`, model YAMLs, dataset YAMLs
3. **Explicit `--config` file** — overrides module defaults
4. **CLI flags** (`--batch-size`, `--learning-rate`, etc.) — override everything

Example: running with a custom config and overriding just the batch size:

```bash
srengine train run \
  --config my_custom.yaml \
  --batch-size 32 \
  --dataset ./datasets/my_set
```

## Workspace Management

Organize datasets, projects, and training sessions with a structured workspace:

```bash
# Initialize a workspace in the current directory
srengine workspace init
# Or use the standalone alias (auto-detects from CWD):
workspace init

# Workspace is auto-detected from CWD for all subsequent commands
srengine workspace check   # validate workspace health
srengine workspace info    # show summary

# Create a project
srengine project create my_experiment
# Or standalone:
project create my_experiment

# List projects
srengine project list

# The workspace layout:
# workspace/
#   .sr_workspace               # auto-detection marker
#   datasets/<name>/HR/ LR/     # dataset pool
#   projects/<name>/
#     configs/                   # experiment configs
#     checkpoints/               # epoch_XXX.pt files
#     metrics/                   # *.jsonl files (machine mode)
```

Once initialized, commands auto-resolve paths relative to the workspace:

```bash
srengine train run --project my_experiment --dataset my_set --model swinir
# dataset resolves to: <workspace>/datasets/my_set/
# checkpoints go to:   <workspace>/projects/my_experiment/checkpoints/
```

Without a workspace, all paths are literal (existing behavior, unchanged).

## Machine Mode (GUI Integration)

The `--machine` flag enables realtime metrics output for GUI consumption:

```bash
srengine train run \
  --project my_experiment \
  --dataset my_set \
  --model swinir \
  --machine \
  --experiment-id run_001 \
  --metrics-frequency 1
```

This writes a `<project>/metrics/<experiment-id>.jsonl` file with one JSON
object per line. The GUI (Godot, Gradio, etc.) tails this file for live
loss, LR, PSNR, and SSIM plots.

Message types written to the `.jsonl` file:

| type | When | Fields |
|---|---|---|
| `step` | Every N batches | epoch, batch, loss_total, loss_pixel, loss_perceptual, lr |
| `validate` | After validation | epoch, psnr, ssim |
| `phase` | State transitions | training / validation / saving / complete |
| `done` | Training finished | elapsed_seconds, total_epochs |

## Config Dump

Print the final merged config (defaults + config file + CLI overrides) without
running the command:

```bash
srengine train run --dataset ./data --model swinir --dump-config
srengine dataset build --input video.mp4 --dump-config
```

Useful for debugging or inspecting what the GUI-generated config resolves to.

## New CLI Commands

| Command | Description | Standalone alias |
|---|---|---|
| `srengine workspace init --path .` | Initialize a workspace | `workspace init` |
| `srengine workspace info` | Show workspace summary | `workspace info` |
| `srengine workspace check` | Validate workspace health | `workspace check` |
| `srengine project create <name>` | Create a project in the workspace | `project create <name>` |
| `srengine project list` | List workspace projects | `project list` |

## New Flags

| Flag | Applies to | Description |
|---|---|---|
| `--workspace PATH` | `srengine` (global) | Explicit workspace path |
| `--project NAME` | `train run` | Project name for path resolution |
| `--machine` | `train run` | Enable JSONL metrics output |
| `--experiment-id TEXT` | `train run` | Experiment identifier |
| `--metrics-frequency N` | `train run` | Log metrics every N batches |
| `--dump-config` | `train run`, `dataset build` | Print final merged config |
| `--num-workers N` | `train run` | Dataloader worker count |
| `--patch-size N` | `train run` | Training patch size |
| `--save-per-epoch N` | `train run` | Save checkpoint every N epochs |

## Project structure

```
envs/                 - Build scripts and Dockerfiles
src/sr_engine/        - Python package
  cli/                - Click CLI commands
  data/               - Dataset building, validation, transforms, degradation
  device/             - CUDA/ROCm backend abstraction
  engine/             - Inference, trainer, metrics, tiling
  models/             - RRDB, SwinIR, checkpointing, losses, registry
    archs/            - Model architectures (rrdbnet.py, swinir.py)
  utils/              - Config loader, I/O, logging
    configs/          - YAML configuration files
      models/
        rrdb_esrgan.yaml
        swinir.yaml
      train/
        base.yaml
      datasets/
        video_pairs.yaml
tests/                - Test suite
```
