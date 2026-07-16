# CLI Reference

## Overview

```
srengine [--version] [--workspace PATH] <command> [options]
```

Standalone aliases exist for each command group — they bypass the `srengine` parent and auto-detect the workspace from CWD:

```
train        infer        dataset        model
env          workspace    project        serve
```

Use `srengine <cmd>` when you need explicit `--workspace PATH` control. Use standalone aliases for convenience when already inside a workspace directory.

## Global Flags

| Flag | Env var | Description |
|------|---------|-------------|
| `--workspace PATH` | `SRENGINE_WORKSPACE` | Explicit workspace path |
| `--version` | — | Print version and exit |

## Command Tree

```
srengine
├── dataset
│   ├── build                  Video → HR/LR dataset, or validate preprocessed
│   ├── validate               Deep structural validation
│   └── health                 Profile dataset, detect/prune black frames
├── train
│   └── run                    Train SR model
├── infer
│   └── run                    Inference on image or video
├── model
│   ├── create-instance        Create named model instance in a project
│   ├── list-instances         List instances in a project
│   ├── list-runs              List training runs for an instance
│   ├── export                 Export to ONNX/safetensors/TorchScript
│   └── info                   Display checkpoint or instance info
├── env
│   ├── check                  Environment diagnostics
│   └── bench                  Micro-benchmark model throughput
├── workspace
│   ├── init                   Initialize workspace directory tree
│   ├── info                   Show workspace summary
│   └── check                  Validate workspace health
├── project
│   ├── create                 Create project in workspace
│   └── list                   List workspace projects
└── serve
    └── start                  Start GUI socket server
```

## dataset

### dataset build

Build a dataset from a video file, or re-validate a preprocessed directory.

```bash
# From raw video — extracts frames, applies degradation
srengine dataset build --input video.mp4

# From an existing preprocessed folder — re-validates + relinks
srengine dataset build --input ./datasets/my_set

# Explicit output directory
srengine dataset build --input video.mp4 --output ./custom/dataset

# Only JPEG compression, area downsampling
srengine dataset build -i video.mp4 --degradations jpeg --resize-method area

# Color jitter + JPEG2000 only
srengine dataset build -i video.mp4 -d color-jitter,jpeg2000

# Noise only (all sub-types)
srengine dataset build -i video.mp4 -d noise
```

| Source | `--output` required | Behaviour |
|--------|---------------------|-----------|
| Video file (`.mp4`, `.avi`, etc.) | No (inside workspace) | Auto-resolves to `<ws>/datasets/<stem>/` |
| Video file | Yes (outside workspace) | Extract frames → apply degradation → write HR/LR pairs |
| Directory (existing dataset) | No | Validate structure, rebuild manifest |

The pipeline reads the dataset config YAML and applies the selected degradation stages. Each stage can be toggled independently.

| Flag | Default | Description |
|------|---------|-------------|
| `--input PATH` | required | Input video file or preprocessed dataset directory |
| `--output PATH` | auto | Output dataset directory |
| `--config PATH` | built-in | Dataset config YAML path |
| `--degradations TEXT` | config | Comma-separated enabled degradations: `blur,noise,jpeg,jpeg2000,color-jitter`. Overrides per-section `enabled` fields from config |
| `--resize-method TEXT` | config | Downsampling method: `area`, `bicubic`, `bilinear`, `lanczos`, `nearest` (default: `area`). Overrides config |
| `--dump-config` | false | Print final merged config and exit |

### dataset validate

Check structural integrity of a dataset.

```bash
srengine dataset validate --path ./datasets/my_set
```

Checks that `HR/` and `LR/` directories exist, image sizes match (scale factor ratio), and every file is readable. Reports total valid pairs and any problems found.

| Flag | Default | Description |
|------|---------|-------------|
| `--path PATH` | required | Dataset directory path |

### dataset health

Profile dataset resolution, aspect ratio, color distribution, and detect black frames.

```bash
srengine dataset health --path ./datasets/my_set
srengine dataset health --path ./datasets/my_set --yes   # auto-delete black frames
```

Profiles every frame: resolution breakdown, aspect ratios, color spaces. Detects completely black frames and optionally prunes them from the manifest and filesystem. Without `--yes`, prompts before deletion.

| Flag | Default | Description |
|------|---------|-------------|
| `--path PATH` | required | Dataset directory path |
| `--yes` | false | Skip confirmation prompt for black frame deletion |

## train

### train run

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

| Flag | Config key | Default | Description |
|------|------------|---------|-------------|
| `--dataset PATH` | — | required | Dataset path or workspace dataset name |
| `--model TEXT` | — | required | Model name (`rrdb_esrgan` or `swinir`) |
| `--config PATH` | — | built-in | Training config YAML path |
| `--batch-size N` | `batch_size` | config | Batch size |
| `--learning-rate F` | `learning_rate` | config | Learning rate |
| `--max-epochs N` | `max_epochs` | config | Maximum epochs |
| `--validation-split F` | `validation.split` | config | Validation split ratio |
| `--validation-enabled` | `validation.enabled` | config | Enable/disable validation |
| `--no-validation-enabled` | | | |
| `--patch-size N` | — | config | Training patch size |
| `--num-workers N` | — | config | Dataloader worker count |
| `--save-per-epoch N` | — | config | Save checkpoint every N epochs |
| `--project TEXT` | — | — | Project name (requires workspace) |
| `--machine` | — | false | Enable JSONL metrics output |
| `--experiment-id TEXT` | — | auto | Experiment identifier for metrics |
| `--metrics-frequency N` | — | config | Log metrics every N batches |
| `--dump-config` | — | false | Print final merged config and exit |
| `--resume PATH` | — | — | Resume from checkpoint |

## infer

### infer run

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

| Input type | Detected by |
|------------|-------------|
| Image (`.png`, `.jpg`, etc.) | File suffix — single-frame SR |
| Video (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`) | File suffix — frame-by-frame upscaling |

| Flag | Default | Description |
|------|---------|-------------|
| `--model PATH` | required | Model checkpoint path |
| `--input PATH` | required | Input image or video path |
| `--output PATH` | required | Output path |
| `--tile N` | 0 (off) | Tile size for tiled inference (0 = no tiling) |
| `--overlap N` | 64 | Tile overlap in pixels |
| `--device TEXT` | auto | Device (`cuda`, `cpu`, or `auto`) |

Tiling trades VRAM for speed. Tiling off is fastest on high-VRAM GPUs. Tiling on (`--tile 512`) avoids OOM on 8 GB cards.

## model

### model export

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

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--model-name TEXT` | `-m` | required | Model architecture name |
| `--ckpt PATH` | `-c` | required | Checkpoint path |
| `--format TEXT` | `-f` | required | Export format (`onnx`, `safetensors`, `torchscript`) |
| `--out PATH` | `-o` | required | Output path |

| Format | Use case |
|--------|----------|
| `onnx` | Cross-platform inference, ONNX Runtime |
| `safetensors` | Safe weight distribution, HuggingFace |
| `torchscript` | C++ `libtorch` deployment, no-Python inference |

### model info

```bash
srengine model info --model model.pth
# Checkpoint: model.pth
#   Step:      45000
#   Config:    {'name': 'swinir', 'type': 'swinir', ...}
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model PATH` | required | Checkpoint path |
| `--instance TEXT` | — | Model instance name (workspace) |

### model create-instance / list-instances / list-runs

```bash
# Create a named model instance in a project
srengine model create-instance --project my_project --name my_model --model swinir

# List instances in a project
srengine model list-instances --project my_project

# List training runs for an instance
srengine model list-runs --instance my_project/my_model
```

## env

### env check

Detect hardware capabilities:

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

### env bench

Micro-benchmark model forward+backward pass:

```bash
# Benchmark RRDB on default device
srengine env bench

# Benchmark SwinIR explicitly
srengine env bench --model swinir
```

Runs on a 128x128 dummy batch. Exits 0 on success, 1 on failure (model not found, etc.).

| Flag | Default | Description |
|------|---------|-------------|
| `--model TEXT` | `rrdb_esrgan` | Model to benchmark |
| `--device TEXT` | auto | Device (`cuda`, `cpu`, `auto`) |
| `--batch-size N` | 1 | Batch size |
| `--iterations N` | 10 | Number of iterations |

## workspace

```bash
# Initialize a workspace in the current directory
srengine workspace init

# Show workspace summary
srengine workspace info

# Validate workspace health
srengine workspace check
```

| Flag | Default | Description |
|------|---------|-------------|
| `--path PATH` | CWD | Path to initialize |

## project

```bash
# Create a project
srengine project create my_experiment

# List projects
srengine project list
```

## serve

```bash
# Start GUI socket server
srengine serve start --port 8765
```

| Flag | Default | Description |
|------|---------|-------------|
| `--port N` | 8765 | TCP port for GUI clients |
| `--host TEXT` | 127.0.0.1 | Bind address |

## Config Dump

Print the final merged config (defaults + config file + CLI overrides) without running the command:

```bash
srengine train run --dataset ./data --model swinir --dump-config
srengine dataset build --input video.mp4 --dump-config
```

Useful for debugging or inspecting what the GUI-generated config resolves to.

## Machine Mode

The `--machine` flag enables realtime metrics output via JSONL files for GUI consumption:

```bash
srengine train run \
  --project my_experiment \
  --dataset my_set \
  --model swinir \
  --machine \
  --experiment-id run_001 \
  --metrics-frequency 1
```

Writes `<project>/metrics/<experiment-id>.jsonl` with one JSON object per line. Message types:

| type | When | Fields |
|------|------|--------|
| `step` | Every N batches | epoch, batch, loss_total, loss_pixel, loss_perceptual, lr |
| `validate` | After validation | epoch, psnr, ssim |
| `phase` | State transitions | training / validation / saving / complete |
| `done` | Training finished | elapsed_seconds, total_epochs |
