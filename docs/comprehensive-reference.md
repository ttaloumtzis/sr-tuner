# sr-engine — Comprehensive Reference Manual

# Executive Summary

**sr-engine** is a modular, production-grade super-resolution (SR) training and inference toolkit for video and image data. It provides a complete pipeline: video frame extraction → synthetic degradation → dataset building → model training → inference → model export — all with first-class support for NVIDIA CUDA and AMD ROCm GPUs.

**Why it matters:** Super-resolution is a critical enabler across video restoration, medical imaging, satellite imagery, surveillance, and media production. sr-engine reduces the gap between research frameworks (which prioritize experimentation over production readiness) and enterprise deployment by offering a unified, well-architected system with a CLI, a Godot GUI bridge, and a clear extension path.

**Key takeaways:**

- **Two architectures:** RRDB (Residual-in-Residual Dense Block, the CNN workhorse from ESRGAN) and SwinIR (Transformer-based with windowed self-attention). Both are composable, configurable, and exportable.
- **Configurable degradation pipeline:** 6 stages (color jitter, blur, downsample, noise, JPEG, JPEG2000) with per-stage probability gating, mutually exclusive sub-types, and CLI quick-select — no YAML editing needed for common variations.
- **4-level config merge:** Builtin YAML defaults → workspace overrides → config file → CLI flags — every knob exposed at every level.
- **GPU abstraction:** A single `get_device()` call returns the optimal backend. Mixed precision (bf16/fp16) and flash attention detection are automatic.
- **GUI bridge:** TCP/NDJSON server with subprocess lifecycle management, real-time progress streaming, and a complete C# Godot client implementation.
- **No lock-in:** Models export to ONNX, SafeTensors, and TorchScript. The config system and model registry make adding new architectures a matter of one file + one decorator.

**Target audience:** ML engineers, data engineers, systems integrators, researchers, and DevOps/SRE engineers building or operating super-resolution systems.

---

# Table of Contents

1. [Introduction](#introduction)
2. [Historical Context](#historical-context)
3. [Terminology and Definitions](#terminology-and-definitions)
4. [Conceptual Foundations](#conceptual-foundations)
5. [System Architecture](#system-architecture)
6. [Core Components](#core-components)
   - 6.1 [CLI Layer](#61-cli-layer)
   - 6.2 [Workspace System](#62-workspace-system)
   - 6.3 [Config System](#63-config-system)
   - 6.4 [Device Abstraction Layer](#64-device-abstraction-layer)
   - 6.5 [Data Pipeline](#65-data-pipeline)
   - 6.6 [Model Registry and Architectures](#66-model-registry-and-architectures)
   - 6.7 [Training Engine](#67-training-engine)
   - 6.8 [Inference Engine](#68-inference-engine)
   - 6.9 [Loss Functions](#69-loss-functions)
   - 6.10 [Metrics System](#610-metrics-system)
   - 6.11 [Checkpointing and Export](#611-checkpointing-and-export)
   - 6.12 [GUI Bridge](#612-gui-bridge)
   - 6.13 [Progress Reporting](#613-progress-reporting)
   - 6.14 [Tiling System](#614-tiling-system)
7. [Internal Mechanics](#internal-mechanics)
   - 7.1 [Training Loop Lifecycle](#71-training-loop-lifecycle)
   - 7.2 [Degradation Pipeline Execution](#72-degradation-pipeline-execution)
   - 7.3 [Config Merge Resolution](#73-config-merge-resolution)
   - 7.4 [Device Detection Flow](#74-device-detection-flow)
   - 7.5 [GUI Bridge Request Lifecycle](#75-gui-bridge-request-lifecycle)
   - 7.6 [Tiled Inference Flow](#76-tiled-inference-flow)
8. [Data Model](#data-model)
9. [APIs / Interfaces](#apis--interfaces)
10. [Installation and Setup](#installation-and-setup)
11. [Configuration Reference](#configuration-reference)
12. [Usage Guide](#usage-guide)
13. [Implementation Guide](#implementation-guide)
14. [Security Analysis](#security-analysis)
15. [Performance Analysis](#performance-analysis)
16. [Reliability and Resilience](#reliability-and-resilience)
17. [Scalability](#scalability)
18. [Operational Guide](#operational-guide)
19. [Troubleshooting](#troubleshooting)
20. [Common Mistakes](#common-mistakes)
21. [Best Practices](#best-practices)
22. [Comparative Analysis](#comparative-analysis)
23. [Real-World Case Studies](#real-world-case-studies)
24. [Advanced Topics](#advanced-topics)
25. [Future Outlook](#future-outlook)
26. [References](#references)

---

# Introduction

## Purpose

sr-engine provides a complete, production-ready platform for:

1. **Building paired HR/LR datasets** from raw video with configurable synthetic degradation
2. **Training super-resolution models** (RRDB, SwinIR) with pixel, perceptual, and adversarial losses
3. **Running inference** on images and video, with tiled inference for VRAM-limited GPUs
4. **Exporting models** to ONNX, SafeTensors, and TorchScript for deployment
5. **Integrating with GUI applications** via a TCP/NDJSON protocol and subprocess job management

## Scope

This document covers the entire sr-engine codebase: every module, every configuration option, every CLI command, and every internal mechanism. It is intended as the single definitive reference for working with the system.

## Audience

- **ML engineers** training or fine-tuning SR models
- **Data engineers** building and maintaining training datasets
- **Systems integrators** embedding SR capabilities into larger platforms
- **DevOps engineers** operating SR pipelines in production
- **Researchers** experimenting with architectures, losses, or degradation strategies
- **Maintainers** extending sr-engine with new models, data sources, or backends

## Assumptions

- Python 3.11–3.13 and `uv` package manager
- Familiarity with PyTorch concepts (tensors, modules, optimizers, dataloaders)
- Basic image processing knowledge (color spaces, convolution, compression)
- Command-line familiarity (bash, environment variables)
- No prior super-resolution domain knowledge required

## Document Goals

After reading this document, you should be able to:

1. Understand every component of sr-engine and how they interact
2. Configure the system for any reasonable use case
3. Diagnose and fix any common failure mode
4. Extend the system with new models, losses, data sources, or commands
5. Deploy sr-engine in production with confidence
6. Benchmark and optimize performance for your hardware

---

# Historical Context

## Origins of Super-Resolution

Super-resolution (SR) is the problem of reconstructing a high-resolution image from one or more low-resolution observations. It is an ill-posed inverse problem: many HR images can produce the same LR image after downsampling and degradation.

### Classical Era (Pre-2014)

- **Interpolation-based**: Bicubic, Lanczos, nearest-neighbor upsampling — fast but no new information is added.
- **Reconstruction-based**: Iterative methods enforcing priors (total variation, sparse coding) — better quality, high computational cost.
- **Example-based**: Dictionary learning on external databases (Yang et al., 2010) — quality depends on dictionary coverage.

### Deep Learning Revolution (2014–2020)

| Year | Model | Key Innovation |
|------|-------|---------------|
| 2014 | SRCNN | First CNN for SR — 3 layers, bicubic pre-upsampling |
| 2016 | VDSR, DRRN | Very deep networks with residual learning |
| 2017 | SRGAN, ESRGAN | Perceptual + adversarial losses for photorealistic SR |
| 2018 | RCAN | Channel attention mechanism |
| 2020 | Real-ESRGAN | High-order degradation model for real-world blind SR |

### Transformer Era (2021–Present)

- **SwinIR** (2021): First Swin Transformer applied to image restoration. Achieves state-of-the-art PSNR/SSIM with windowed self-attention.
- **HAT** (2023): Hybrid Attention Transformer combining channel attention and window attention.
- **DAT** (2023): Dual Aggregation Transformer.

## sr-engine's Design Heritage

sr-engine builds on the classic synthetic degradation model (BSRGAN/Real-ESRGAN) and implements two proven architectures (RRDB/ESRGAN and SwinIR). Its engineering innovations include:

- **GPU-agnostic backend**: Runtime CUDA vs. ROCm detection without configuration changes
- **Process-pool degradation**: Parallel CPU execution for high-throughput dataset building
- **CLI-first design**: All functionality accessible from the command line
- **GUI bridge as a first-class citizen**: TCP/NDJSON server with subprocess management for Godot integration
- **4-level config merge**: Builtin defaults → workspace overrides → file → CLI flags

Previous solutions (MATLAB toolboxes, standalone Python scripts, research frameworks like BasicSR) required manual venv management, CUDA/ROCm awareness, and had no GUI integration path. sr-engine packages all of this into a single, well-architected system.

---

# Terminology and Definitions

| Term | Definition |
|------|------------|
| **HR** | High-Resolution — the original, undegraded image (ground truth) |
| **LR** | Low-Resolution — the degraded counterpart of HR, simulating real-world low-quality input |
| **Scale factor** | Integer ratio `HR_dimension / LR_dimension` (typically 2, 3, or 4) |
| **Patch** | A cropped sub-region of an image used for training (e.g., 128×128 LR → 512×512 HR at 4× scale) |
| **Degradation** | The process of transforming HR to LR via blur, noise, downsampling, compression, etc. |
| **Blind SR** | Super-resolution where the degradation model is unknown — the model must generalize from diverse training degradations |
| **Synthetic degradation** | Algorithmically generated LR from HR, used to create paired training data |
| **ProcessPoolExecutor** | `concurrent.futures` mechanism for true parallel CPU execution across multiple processes |
| **NDJSON** | Newline-Delimited JSON — one JSON object per line, used in the GUI bridge protocol |
| **Workspace** | A directory tree (marked by `.sr_workspace`) containing datasets, projects, configs, and job manifests |
| **Project** | A named experiment directory within a workspace, with configs, checkpoints, and metrics |
| **Model instance** | A named model configuration within a project, with checkpoint and run history |
| **Manifest** | `manifest.json` — the index file mapping HR/LR pairs in a dataset directory |
| **SAVE_PER_EPOCH** | Config parameter controlling validation frequency every N epochs |
| **RRDB** | Residual-in-Residual Dense Block — the CNN building block of ESRGAN |
| **RSTB** | Residual Swin Transformer Block — the Transformer building block of SwinIR |
| **W-MSA** | Window Multi-Head Self-Attention (non-overlapping windows) |
| **SW-MSA** | Shifted Window Multi-Head Self-Attention (cross-window connections via cyclic shift) |
| **PixelShuffle** | Efficient sub-pixel convolution upsampling (also known as depth-to-space) |
| **Charbonnier loss** | A differentiable L1 approximation: `sqrt(diff^2 + eps^2)` — smoother than L1 at zero |
| **Perceptual loss** | Feature-space L1 distance using VGG19 activations — encourages perceptually similar output |
| **GAN loss** | Adversarial loss for ESRGAN-style training — discriminator distinguishes SR from HR |
| **SDPA** | Scaled Dot-Product Attention (`F.scaled_dot_product_attention` in PyTorch 2.0+) |
| **Flash Attention** | A fused kernel for efficient attention computation (integrated in PyTorch SDPA) |
| **bf16** | Brain floating-point 16 — mixed-precision format supported on Ampere+ and MI200+ |
| **fp16** | IEEE half-precision float — supported on all CUDA GPUs |
| **ONNX** | Open Neural Network Exchange — cross-platform model format |
| **SafeTensors** | A safe weight-storage format (no pickle, no arbitrary code execution) |
| **TorchScript** | PyTorch's JIT-compiled model format for C++ `libtorch` inference |
| **INTER_AREA** | OpenCV interpolation method — best anti-aliasing for downscaling |
| **Autocast** | PyTorch automatic mixed precision — selects dtype per operation |
| **SSIM** | Structural Similarity Index Measure — perceptual image quality metric |
| **PSNR** | Peak Signal-to-Noise Ratio — pixel-wise quality metric in dB |
| **LPIPS** | Learned Perceptual Image Patch Similarity — deep-feature-based perceptual metric |

---

# Conceptual Foundations

## The Super-Resolution Inverse Problem

Super-resolution is fundamentally an inverse problem: given an LR observation `y`, find the most plausible HR image `x` such that:

```
y = D(x) + η
```

Where `D` is the degradation function (downsampling + blur + noise + compression) and `η` is additive noise. The problem is **ill-posed** because:
- Infinite HR images map to the same LR input (information is destroyed during degradation)
- The degradation function `D` is generally unknown in real-world applications

Super-resolution models learn a function `f_θ(y) ≈ x` that approximates the inverse of `D`. The quality of this approximation depends on:
1. **Model capacity** (architecture expressiveness, parameter count)
2. **Training data diversity** (how well the training degradation space covers real-world degradation)
3. **Loss function** (what "good" means — pixel accuracy, perceptual quality, or both)

## Why Blind SR Requires Diverse Training Degradation

A model trained on bicubic-only downsampling learns to reverse only bicubic interpolation. Real-world LR images suffer from a combination of:

- **Optical blur** from lens defocus, motion, or atmospheric turbulence
- **Sensor noise** (Gaussian, Poisson/shot, salt-and-pepper)
- **Downsampling artifacts** from camera sensor sampling patterns
- **Compression artifacts** from JPEG, JPEG2000, HEVC, or AV1 codecs
- **Color shifts** from different camera sensors, white balance, or color grading

The degradation pipeline in sr-engine models these effects through a configurable sequence of stages. The model learns to reverse any combination it has seen during training. The **probability gating** per stage ensures diverse combinations: some training examples may have all six degradations, others may have only blur + downsample.

## The Tradeoff Between Diversity and Consistency

Each degradation stage has a "probability of application" (`prob` in the config). These values control the expected frequency of each degradation type in the training distribution:

- **High prob (1.0):** Applied to nearly every training example — the model becomes specialized at reversing this degradation
- **Low prob (0.3):** Seen less frequently — the model learns to handle it but won't over-specialize
- **Zero prob (0.0):** Never applied — the model will fail if this degradation appears at inference time

The probability values must be tuned based on the deployment target. For general-purpose SR, moderate probabilities (0.3–0.7) for most stages work well. For domain-specific SR (e.g., satellite imagery), match the probabilities to the known imaging pipeline.

## Why Multiple Model Architectures

sr-engine provides two architectures with fundamentally different design philosophies:

| Property | RRDB (ESRGAN) | SwinIR |
|----------|---------------|--------|
| **Core operation** | Convolution (local, translation-equivariant) | Self-attention (global, content-adaptive) |
| **Parameter efficiency** | Good at moderate scales (2×–4×) | Excellent at all scales |
| **Memory footprint** | Low — pure conv, constant memory per pixel | Higher — attention scales as O(window²) |
| **Speed** | Fast — pure conv, GPU-optimized | Slower — window operations, masking overhead |
| **Receptive field** | Limited by depth (23 RRDB blocks = large) | Global within each window, cross-window via shifting |
| **Best for** | High-throughput, low-latency, production deployment | Maximum quality, research, large-scale SR |

Training both and comparing results is a common workflow: RRDB for the throughput/quality frontier, SwinIR for the quality ceiling.

## Mixed Precision Training

Modern GPUs achieve 2-4× training throughput with half-precision (fp16/bf16) vs. fp32. sr-engine automatically selects the best dtype:

- **bf16** on Ampere+ CUDA GPUs (A100, RTX 3090, RTX 4090) and MI200+ ROCm GPUs — wider dynamic range than fp16, no loss scaling needed
- **fp16** on older CUDA GPUs (V100, RTX 2080) — requires gradient scaling
- **fp32** on CPU — no half-precision benefit

The `torch.amp.autocast` context manager handles the dtype selection automatically per operation. Loss-sensitive operations (e.g., reductions) run in fp32; compute-heavy operations (convolutions, matrix multiplies) run in the lower-precision dtype.

---

# System Architecture

## High-Level Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                           CLI Layer (cli/)                            │
│  main.py │ cmd_train.py │ cmd_infer.py │ cmd_dataset.py               │
│  cmd_model.py │ cmd_env.py │ cmd_serve.py │ workspace_commands.py     │
│  helpers.py (progress/callback resolution)                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐      │
│  │            Config Layer (utils/config.py)                     │      │
│  │  4-level merge: builtin → workspace → file → CLI flags       │      │
│  └─────────────────────┬───────────────────────────────────┘      │
│                        │                                         │
│  ┌─────────────────────▼───────────────────────────────────┐      │
│  │            Workspace Layer (workspace.py)                 │      │
│  │  Auto-discovery, marker resolution, project/model CRUD,   │      │
│  │  dataset path resolution, config layering                 │      │
│  └─────────────────────┬───────────────────────────────────┘      │
│                        │                                         │
├────────────────────────┼─────────────────────────────────────────┤
│                        │                                         │
│  ┌─────────────────────▼──────────┐  ┌──────────────────────┐    │
│  │      Data Pipeline (data/)     │  │    Engine (engine/)  │    │
│  │                                │  │                      │    │
│  │  video_extract.py              │  │  trainer.py          │    │
│  │  degrade.py                    │  │  inference.py        │    │
│  │  dataset_builder.py            │  │  tiling.py           │    │
│  │  dataset_validator.py          │  │  metrics.py          │    │
│  │  dataset_health.py             │  │  metrics_stream.py   │    │
│  │  datasets.py (PyTorch Dataset) │  │                      │    │
│  │  transforms.py                 │  │                      │    │
│  └────────────────────────────────┘  └──────────────────────┘    │
│                        │                                         │
│  ┌─────────────────────▼───────────────────────────────────┐      │
│  │            Models Layer (models/)                         │      │
│  │  registry.py │ checkpoint.py │ losses.py                  │      │
│  │  archs/rrdbnet.py │ archs/swinir.py                       │      │
│  └─────────────────────┬───────────────────────────────────┘      │
│                        │                                         │
│  ┌─────────────────────▼───────────────────────────────────┐      │
│  │            Device Layer (device/)                         │      │
│  │  backend.py │ kernels.py                                  │      │
│  │  CUDA/ROCm detection, bf16/fp16 selection, SDPA, flash   │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                       │
├──────────────────────────────────────────────────────────────────────┤
│                  GUI Bridge Layer (gui_bridge/)                        │
│  server.py │ jobs.py │ protocol.py                                   │
│  TCP/NDJSON server, subprocess manager, job manifests               │
└──────────────────────────────────────────────────────────────────────┘
```

## Architectural Goals

| Goal | Mechanism |
|------|-----------|
| **Separation of concerns** | Layers depend strictly downward — CLI → Config → Workspace → Data/Engine → Models → Device |
| **Configurability** | Every parameter exposed at 4 levels with recursive merge |
| **GPU agnosticism** | Device layer abstracts CUDA vs. ROCm — no `if cuda:` checks in model/training code |
| **Extensibility** | Decorator-based model registry, strategy pattern for progress reporting, callback pattern for training events |
| **Observability** | ProgressReporter, TrainerCallback, MetricsStream — abstract interfaces with multiple implementations |
| **No PyTorch dependency at install time** | PyTorch installed separately via `envs/build.sh` with backend-specific index — avoids CUDA-dependency for ROCm users |
| **Determinism** | Seeded RNG per worker process, configurable global seed, fixed degradation pipeline order |

## Dependency Direction

```
cli/ ─────► utils/config.py ──► workspace.py ──► data/ ──► models/ ──► device/
  │              │                                  │          │
  └──────────────┴──────────────────────────────────┴──────────┘
                              │
                         gui_bridge/
                              │
                         subprocess → cli commands
```

## Module Dependency Graph (Detailed)

```
cli/cmd_train.py
  ├── cli/helpers.py           (resolve_reporter, resolve_callbacks, resolve_cancel_check)
  ├── utils/config.py          (DefaultConfigs, merge_overrides)
  ├── utils/progress.py        (ProgressReporter)
  ├── workspace.py             (Workspace — resolve_dataset, get_model_instance)
  ├── engine/trainer.py        (Trainer class)
  ├── models/checkpoint.py     (load_checkpoint for resume)
  └── gui_bridge/protocol.py   (SocketReporter, SocketCallback)

cli/cmd_dataset.py
  ├── cli/helpers.py
  ├── utils/config.py
  ├── utils/progress.py
  ├── workspace.py
  ├── data/dataset_builder.py   (build_from_video, build_from_preprocessed)
  ├── data/dataset_validator.py (validate)
  └── data/dataset_health.py    (check_dataset_health, prune_black_frames)

engine/trainer.py
  ├── models/registry.py        (build_model)
  ├── models/losses.py          (L1Loss, PerceptualLoss)
  ├── models/checkpoint.py      (save_checkpoint, load_checkpoint)
  ├── engine/metrics.py         (psnr, ssim)
  ├── engine/metrics_stream.py  (MetricsStream)
  ├── data/datasets.py          (PairedImageFolderDataset)
  ├── data/transforms.py        (Compose, RandomCrop, RandomFlip, RandomRotate)
  └── utils/progress.py         (ProgressReporter)

gui_bridge/server.py
  ├── gui_bridge/jobs.py        (JobManager, JobManifest)
  └── workspace.py              (Workspace)

gui_bridge/jobs.py
  └── gui_bridge/protocol.py    (connect_control_socket, make_json_sender)
```

---

# Core Components

## 6.1 CLI Layer

**Location:** `src/sr_engine/cli/`

### Purpose

Provides the primary user-facing interface — a tree of Click commands that expose every sr-engine capability.

### Structure

| File | Group | Commands |
|------|-------|----------|
| `main.py` | `srengine` (root) | Not a command itself — registers all sub-groups |
| `cmd_train.py` | `train` | `run` |
| `cmd_infer.py` | `infer` | `run` |
| `cmd_dataset.py` | `dataset` | `build`, `validate`, `health` |
| `cmd_model.py` | `model` | `create-instance`, `list-instances`, `list-runs`, `export`, `info` |
| `cmd_env.py` | `env` | `check`, `bench` |
| `cmd_serve.py` | `serve` | `start` |
| `workspace_commands.py` | `workspace`, `project` | `init`, `info`, `check`, `create`, `list` |
| `helpers.py` | (shared utilities) | `resolve_workspace`, `resolve_reporter`, `resolve_callbacks`, `resolve_cancel_check`, `parse_config_overrides` |

### Entry Points

From `pyproject.toml`:

```toml
[project.scripts]
srengine = "sr_engine.cli.main:cli"
workspace = "sr_engine.cli.workspace_commands:workspace"
project = "sr_engine.cli.workspace_commands:project"
dataset = "sr_engine.cli.cmd_dataset:dataset"
env = "sr_engine.cli.cmd_env:env"
infer = "sr_engine.cli.cmd_infer:infer"
model = "sr_engine.cli.cmd_model:model"
serve = "sr_engine.cli.cmd_serve:serve"
train = "sr_engine.cli.cmd_train:train"
```

Each standalone alias bypasses the `srengine` parent group and auto-detects the workspace from CWD. Use `srengine <cmd>` when explicit `--workspace PATH` control is needed.

### Helpers Module (`cli/helpers.py`)

Three critical resolver functions that implement the **Strategy Pattern**:

#### `resolve_reporter()`
```python
def resolve_reporter(machine: bool = False) -> ProgressReporter:
```
- If `SRENGINE_GUI_SOCKET` env var is set: returns `SocketReporter` (sends NDJSON to GUI)
- If `machine=True`: returns `TqdmReporter` (terminal) — the metrics go to JSONL via `MetricsStreamCallback` instead
- Otherwise: returns `TqdmReporter`

#### `resolve_callbacks()`
```python
def resolve_callbacks(machine: bool = False) -> list[TrainerCallback]:
```
- If `SRENGINE_GUI_SOCKET` is set: returns `[SocketCallback]` for GUI streaming
- If `machine=True`: sets up `MetricsStreamCallback` internally (in Trainer)
- Otherwise: returns `[]`

#### `resolve_cancel_check()`
```python
def resolve_cancel_check() -> Callable[[], bool]:
```
- If `SRENGINE_GUI_SOCKET` is set: installs a SIGTERM handler and returns `was_cancelled` callback
- Otherwise: returns `lambda: False` (no-op)

### Inputs/Outputs

| Input | Source | Example |
|-------|--------|---------|
| CLI arguments | User | `srengine train run --dataset my_set --model swinir --batch-size 4` |
| Environment variables | OS | `SRENGINE_WORKSPACE=/data/ws`, `SRENGINE_GUI_SOCKET=...` |
| YAML config files | Filesystem | `--config my_train.yaml` |

| Output | Destination | Example |
|--------|-------------|---------|
| Terminal output | stdout/stderr | Progress bars, log messages, tables |
| Metrics JSONL | Filesystem | `<project>/metrics/<experiment_id>.jsonl` |
| NDJSON events | TCP socket (GUI) | `{"type":"step","epoch":1,...}` |

### Key Internal Logic

**Workspace resolution** (in every command):
1. Check `--workspace PATH` flag
2. Check `SRENGINE_WORKSPACE` env var
3. Walk up from CWD looking for `.sr_workspace` marker
4. If none found and workspace is required: raise `FileNotFoundError`

**Config resolution** (in `cmd_train.py` and `cmd_dataset.py`):
1. Load builtin defaults from `utils/configs/`
2. Merge workspace overrides from `<workspace>/configs/`
3. Merge `--config` YAML file if provided
4. Merge CLI flag overrides (highest priority)
5. Optionally `--dump-config` to print and exit

### Failure Modes

| Symptom | Cause | Resolution |
|---------|-------|------------|
| `Error: No such option: --foo` | Typo or unknown flag | Check CLI reference for valid flags |
| `FileNotFoundError: Workspace not found` | No workspace context | Run `srengine workspace init` or set `SRENGINE_WORKSPACE` |
| `ClickException` on missing required argument | Required `--dataset` or `--model` omitted | Supply the missing argument |

---

## 6.2 Workspace System

**Location:** `src/sr_engine/workspace.py`

### Purpose

Provides structured project organization, path auto-resolution, config layering, and model instance management. Eliminates the need for absolute paths once a workspace is initialized.

### Responsibilities

- Workspace auto-discovery (walk up from CWD)
- Directory creation for datasets, projects, configs, jobs
- Project CRUD (create, list, get)
- Model instance CRUD (create, get, list)
- Checkpoint listing per instance
- Run directory creation (timestamp-based)
- Dataset path resolution (absolute → CWD-relative → workspace-relative)
- Builtin config copying to workspace

### Workspace Directory Structure

```
<workspace_root>/
├── .sr_workspace                  # JSON marker: {"version": 1, "created": "..."}
├── datasets/                      # Dataset pool
│   └── <name>/
│       ├── HR/                    # High-resolution frames (PNG)
│       ├── LR/                    # Low-resolution frames (PNG)
│       └── manifest.json          # Pairs index
├── models/                        # Named model instances
│   └── <instance_name>/
│       ├── config.yaml            # Frozen model-architecture config
│       ├── versions/              # Versioned checkpoints (v1/, v2/, ...)
│       ├── checkpoints/           # Training checkpoints (epoch_*.pt)
│       └── runs/                  # run_<timestamp>/ directories
├── experiments/                   # Experiment artifacts
├── jobs/                          # GUI bridge job manifests
│   └── <job_id>.json              # Persisted job state
├── configs/                       # User-overridable configs
│   ├── train/                     # Overrides builtin train config
│   ├── datasets/                  # Overrides builtin dataset config
│   └── models/                    # Overrides builtin model configs
```

### Auto-Discovery Algorithm

```python
@classmethod
def discover(cls) -> Workspace | None:
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        marker = parent / ".sr_workspace"
        if marker.is_file():
            return cls(path=parent)
    return None
```

Walks up from CWD to root (/) checking for `.sr_workspace`. Stops at the first match. Returns `None` if no marker is found.

### Dataset Path Resolution

```python
def resolve_dataset(self, name_or_path: Path) -> Path:
    # 1. Absolute path — use as-is
    if name_or_path.is_absolute():
        return name_or_path
    # 2. CWD-relative — try
    resolved_cwd = name_or_path.resolve()
    if resolved_cwd.exists():
        return resolved_cwd
    # 3. Workspace-relative — try <ws>/datasets/<name>
    resolved_ws = (self.path / "datasets" / name_or_path).resolve()
    if resolved_ws.exists():
        return resolved_ws
    raise FileNotFoundError(...)
```

### Model Instance System

Model instances are named configurations that track checkpoint history and training runs:

```
<workspace>/models/<instance>/
├── config.yaml          # Frozen model architecture config (immutable after creation)
├── checkpoints/         # epoch_XXX.pt — sorted by mtime desc
│   ├── epoch_010.pt
│   └── epoch_020.pt
└── runs/                # timestamp-based run directories
    └── run_20250516_120000/
        ├── config.yaml
        ├── metrics.jsonl
        └── checkpoint.pt
```

### Failure Modes

| Symptom | Cause | Resolution |
|---------|-------|------------|
| `FileNotFoundError: Dataset not found` | Dataset path not resolvable | Check the path or create the dataset first |
| `FileExistsError: Project already exists` | Duplicate project name | Use a different name or work with the existing project |
| `FileNotFoundError: Model instance not found` | Wrong instance name | Run `model list-instances --instance <p>` to see available instances |

---

## 6.3 Config System

**Location:** `src/sr_engine/utils/config.py`

### Purpose

Provides a 4-level hierarchical configuration merge system that lets users set any parameter at any level of specificity.

### Responsibilities

- Load YAML config files from multiple locations
- Recursively merge configs with user overrides taking precedence
- Validate required keys
- Save configs as YAML
- Provide access to builtin default configs
- Apply CLI flag overrides on top of merged configs

### 4-Level Precedence

| Level | Source | Mechanism | Priority |
|-------|--------|-----------|----------|
| 1 | Builtin defaults | `utils/configs/*.yaml` | Lowest |
| 2 | Workspace overrides | `<workspace>/configs/**/*.yaml` | |
| 3 | Config file | `--config PATH` CLI flag | |
| 4 | CLI flags | `--batch-size`, `--lr`, etc. | **Highest** |

### Config Merge Algorithm

```python
def merge_overrides(base: dict, overrides: dict) -> dict:
    """Recursive dict merge. Lists and scalars are replaced, dicts recurse."""
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_overrides(result[key], value)
        else:
            result[key] = value
    return result
```

This means:
- Nested dicts are merged recursively (you can override just one key in a nested config)
- Scalar values are replaced entirely (no "partial float" merging)
- Lists are replaced entirely (not appended)

### DefaultConfigs Class

```python
class DefaultConfigs:
    def __init__(self, workspace: Workspace | None = None):
        self.builtins = self._load_builtins()
        self.workspace = workspace

    def get_train_config(self, config_path: str | None = None) -> dict:
        cfg = self.builtins["train/base.yaml"]          # Level 1
        cfg = self._ws_or_builtin("train/base.yaml", cfg)  # Level 2
        if config_path:
            cfg = merge_overrides(cfg, load_yaml(config_path))  # Level 3
        return cfg

    @staticmethod
    def apply_cli_overrides(cfg: dict, cli_kwargs: dict) -> dict:
        return merge_overrides(cfg, cli_kwargs)  # Level 4
```

### Builtin Config Files

| File | Key Parameters | Purpose |
|------|---------------|---------|
| ~~`default.yaml`~~ | *(deleted — values distributed to specific configs)* | |
| `train/base.yaml` | `max_epochs`, `learning_rate`, `batch_size`, `patch_size`, `losses`, `validation` | Training hyperparameters |
| `datasets/video_pairs.yaml` | `scale`, `frame_rate`, `degradation` (full 6-stage config) | Degradation pipeline parameters |
| `models/swinir.yaml` | `embed_dim`, `depths`, `num_heads`, `window_size` | SwinIR architecture parameters |
| `models/rrdb_esrgan.yaml` | `num_feat`, `num_block`, `num_grow_ch` | RRDB architecture parameters |

### Config Validation

```python
def validate_config(config: dict) -> None:
    """Validate required keys exist and have expected types."""
    required_keys = ["scale", "batch_size", "learning_rate"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
```

Called at CLI entry point before any work begins. Prevents runtime failures due to missing configuration.

### Failure Modes

| Symptom | Cause | Resolution |
|---------|-------|------------|
| `FileNotFoundError: No such config file` | `--config` path is wrong | Check the path |
| `yaml.YAMLError` | Malformed YAML | Validate YAML syntax with `yamllint` or Python `yaml.safe_load` |
| `ValueError: Missing required config key` | Required key not in merged config | Add the key to your config file or use default |
| Unexpected config behavior | Wrong key name (typo silently ignored) | Use `--dump-config` to see the resolved config |

---

## 6.4 Device Abstraction Layer

**Location:** `src/sr_engine/device/backend.py`, `device/kernels.py`

### Purpose

Provides a uniform interface for CUDA and ROCm GPU backends. Training and inference code never need to check the backend explicitly.

### Responsibilities

- Detect available GPU hardware (CUDA vs. ROCm vs. CPU)
- Select optimal dtype for mixed precision (bf16 vs. fp16 vs. fp32)
- Check flash attention support
- Provide backend-aware kernel implementations
- Return device properties (VRAM, name, compute capability)

### Device Detection

```python
def get_device(preferred: str = "auto") -> torch.device:
```

| `preferred` | CUDA available | ROCm detected | Result |
|-------------|---------------|---------------|--------|
| `"auto"` | Yes | No | `cuda:0` |
| `"auto"` | Yes | Yes | `cuda:0` (ROCm reports as CUDA) |
| `"auto"` | No | — | `cpu` |
| `"cuda"` | Yes | — | `cuda:0` |
| `"cuda"` | No | — | RuntimeError |
| `"cpu"` | — | — | `cpu` |

### Backend Detection Functions

```python
def is_rocm() -> bool:
    return hasattr(torch.version, 'hip') and torch.version.hip is not None

def get_device_name() -> str:
    return torch.cuda.get_device_name(0)

def get_vram() -> int:
    return torch.cuda.get_device_properties(0).total_memory // 1048576

def get_vram_used() -> int:
    return torch.cuda.memory_allocated(0) // 1048576
```

### Mixed Precision Selection

```python
def autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    # bf16 is available on Ampere+ (compute 8.0+) and ROCm 5.7+
    if _check_bf16_support():
        return torch.bfloat16
    return torch.float16
```

| Backend | BF16 support | Autocast dtype |
|---------|-------------|----------------|
| CUDA compute 8.0+ (A100, H100, RTX 3090, RTX 4090) | Yes | `bfloat16` |
| CUDA compute 7.0+ (V100, RTX 2080) | Partial | `bfloat16` or `float16` |
| CUDA compute <7.0 (GTX 1080) | No | `float16` |
| ROCm MI200+ | Yes | `bfloat16` |
| ROCm older | No | `float16` |
| CPU | — | `float32` |

### Flash Attention Support

```python
def supports_flash_attn(device: torch.device) -> bool:
```

Detection checks:
- CUDA compute capability >= 8.0
- `torch.backends.cuda.flash_sdp_enabled()` returns True
- For ROCm: requires ROCm 5.7+ with compatible GPU

### Backend-Aware Kernels

```python
def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0):
    """Uses PyTorch 2.0 SDPA if available, falls back to manual."""
    if hasattr(F, 'scaled_dot_product_attention'):
        return F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    # Manual fallback
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if attn_mask is not None:
        scores += attn_mask
    attn = torch.softmax(scores, dim=-1)
    attn = F.dropout(attn, dropout_p)
    return torch.matmul(attn, v)

def get_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    return nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
```

Currently no backend-specific convolution needed — both CUDA and ROCm use PyTorch's standard `nn.Conv2d`. The indirection exists for future integration with backend-specific libraries.

### Internal Logic

```
get_device("auto")
  │
  ├── is CUDA available?
  │     ├── Yes → is it ROCm? → return cuda:0
  │     └── No  → return cpu
  │
  └── (specific requested → validate availability → return or raise)
```

### Failure Modes

| Symptom | Cause | Resolution |
|---------|-------|------------|
| `RuntimeError: CUDA requested but not available` | `--device cuda` on CPU-only system | Install CUDA PyTorch, use `--device cpu`, or switch `--backend cuda` build |
| `torch.cuda.OutOfMemoryError` | Insufficient VRAM | Reduce batch size, enable tiling, switch to smaller model |
| BF16 training errors | GPU doesn't support bf16 | sr-engine falls back to fp16 automatically — check `env check` output |
| Flash attention silently not used | GPU compute capability < 8.0 | No action needed — SDPA uses memory-efficient or vanilla path |

---

## 6.5 Data Pipeline

**Location:** `src/sr_engine/data/`

### Overview

The data pipeline converts raw video into paired HR/LR image datasets for super-resolution training. It consists of six submodules:

| Submodule | Function | Responsibility |
|-----------|----------|---------------|
| `video_extract.py` | `extract_frames()` | OpenCV-based frame extraction from video files |
| `degrade.py` | `batch_degrade()`, `_degrade_image()` | Parallel HR→LR synthetic degradation |
| `dataset_builder.py` | `build_from_video()`, `build_from_preprocessed()` | Pipeline orchestrator |
| `dataset_validator.py` | `validate()` | Structural and dimensional validation |
| `dataset_health.py` | `check_dataset_health()`, `prune_black_frames()` | Quality profiling and black frame removal |
| `datasets.py` | `PairedImageFolderDataset` | PyTorch Dataset for training |
| `transforms.py` | `RandomCrop`, `CenterCrop`, `RandomFlip`, `RandomRotate`, `Compose` | Training-time augmentation |

### 6.5.1 Video Extraction (`video_extract.py`)

**`extract_frames(video_path, out_dir, frame_rate=None, start_time=0.0, duration=None, reporter=None)`**

Extracts individual frames from video files as PNG sequences.

**Algorithm:**
1. Open video with `cv2.VideoCapture`
2. Seek to `start_frame = start_time * fps`
3. Compute frame step: `max(1, round(video_fps / target_fps))`
4. Iterate over frames:
   - Target frames: `vidcap.read()` (decode + return)
   - Skip frames: `vidcap.grab()` (advance without decoding — 2-3× faster)
5. Write PNG files named `0.png`, `1.png`, ... (zero-padded to match total count)

**Key optimization:** `grab()` skips pixel decoding for non-target frames, providing ~2-3× speedup for high-FPS source video at low target frame rates.

**Supported formats:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` (whatever OpenCV + FFmpeg backend supports)

### 6.5.2 Degradation Pipeline (`degrade.py`)

**`batch_degrade(hr_paths, lr_dir, scale, config, reporter=None)`**

Orchestrates parallel HR→LR degradation using `ProcessPoolExecutor`.

**Worker initialization:**
```python
def _init_worker():
    cv2.setNumThreads(1)  # Prevent CPU oversubscription
    seed = os.getpid() + int.from_bytes(os.urandom(4), "little")
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
```

Each worker process gets a unique random seed derived from PID + OS randomness, ensuring varied degradation across frames.

**Per-frame degradation order (`_degrade_image`):**

```
HR image (BGR uint8)
  │
  ├── 1. Crop to scale-multiple dimensions
  │     height -= height % scale
  │     width -= width % scale
  │
  ├── 2. Color Jitter (if enabled AND random() < prob)
  │     BGR → HSV → shift H/S/V → HSV → BGR
  │
  ├── 3. Blur (if enabled, mutually exclusive sub-types)
  │     ├── Gaussian blur (if triggered)
  │     └── Motion blur (if triggered) — only one applied
  │
  ├── 4. Antialias pre-filter (if enabled AND method != "area")
  │     GaussianBlur(sigma=0.5)
  │
  ├── 5. Downsample (ALWAYS applied)
  │     cv2.resize(lr_size, interpolation=method)
  │
  ├── 6. Noise (if enabled, mutually exclusive sub-types)
  │     ├── Gaussian noise
  │     ├── Poisson noise
  │     └── Salt & Pepper — only one applied
  │
  ├── 7. JPEG compression (if enabled AND random() < prob)
  │     cv2.imencode('.jpg') → cv2.imdecode()
  │
  ├── 8. JPEG2000 compression (if enabled AND random() < prob)
  │     cv2.imencode('.jp2') → cv2.imdecode()
  │
  └── Return LR image (BGR uint8, lr_size)
```

**Degradation stage details:**

| Stage | Function | Parameters | Effect |
|-------|----------|------------|--------|
| Color Jitter | `_apply_color_jitter()` | hue_range, saturation_range, value_range | HSV shift simulating camera differences |
| Gaussian Blur | `_apply_gaussian_blur()` | kernel_size, sigma | Isotropic lens defocus blur |
| Motion Blur | `_apply_motion_blur()` | max_kernel_size | Linear directional motion blur |
| Downsample | (inline) | method (area/bicubic/etc.), antialias | Spatial resolution reduction |
| Gaussian Noise | `_add_gaussian_noise()` | sigma_range | Additive white Gaussian noise |
| Poisson Noise | `_add_poisson_noise()` | scale_range | Photon counting noise (signal-dependent) |
| Salt & Pepper | `_add_salt_pepper_noise()` | amount, salt_vs_pepper | Random black/white pixels |
| JPEG | `_apply_jpeg_compression()` | quality_range | DCT-based lossy compression |
| JPEG2000 | `_apply_jpeg2000_compression()` | quality_range | Wavelet-based lossy compression |

### 6.5.3 Dataset Builder (`dataset_builder.py`)

**`build_from_video(video_path, output_dir, config, reporter=None)`**

Full pipeline: extract frames → degrade → write manifest → validate.

```python
def build_from_video(video_path, output_dir, config, reporter=None):
    hr_dir = output_dir / "HR"
    lr_dir = output_dir / "LR"
    hr_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract frames
    hr_paths = extract_frames(video_path, hr_dir, ...)

    # Step 2: Degrade frames
    pairs = batch_degrade(hr_paths, lr_dir, scale, config, reporter)

    # Step 3: Write manifest
    manifest = {
        "config": {"scale": scale, "frame_rate": frame_rate, "video_source": video_path.name},
        "pairs": [{"hr": str(h.relative_to(output_dir)), "lr": str(l.relative_to(output_dir))}
                  for h, l in pairs]
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Step 4: Validate
    report = validate(output_dir)
    if not report.ok:
        (output_dir / "manifest.json").unlink()  # Self-cleaning
        raise RuntimeError(f"Dataset validation failed: {report.problems}")

    return output_dir
```

**`build_from_preprocessed(input_dir)`**

Validates and creates a manifest from existing HR/LR directories. Used when HR and LR images already exist (e.g., from a different source).

### 6.5.4 Dataset Validator (`dataset_validator.py`)

**`validate(dataset_dir) → ValidationReport`**

Checks performed:
1. `HR/`, `LR/` directories and `manifest.json` exist
2. Manifest is valid JSON with expected structure
3. Every file in manifest exists on disk
4. Every file is a readable image (`cv2.imread` succeeds)
5. Dimension ratio: `HR_dim / LR_dim == scale` (exact integer match)
6. No orphan files (files on disk not in manifest)

```python
@dataclass
class ValidationReport:
    ok: bool
    num_pairs: int = 0
    problems: list[str] = field(default_factory=list)
```

### 6.5.5 Dataset Health Checker (`dataset_health.py`)

**`check_dataset_health(dataset_dir) → dict`**

Profiles every HR frame for:
- Resolution distribution
- Aspect ratio distribution
- Color channel distribution (RGB, grayscale detection)
- Black frame detection via adaptive thresholding

**Adaptive black frame threshold:**
```
1. Sort frame mean brightness values
2. Analyze lowest 15th percentile
3. Find largest gap between consecutive values in this percentile
4. If gap > 1.5: threshold = midpoint of gap (clamped to maximum 25.0)
5. Otherwise: fallback based on dynamic range
   - If 15th percentile < 10.0 → 3.5 (full range, 0-255)
   - Otherwise → 18.5 (limited range, 16-235)
```

**`prune_black_frames(dataset_dir, threshold=None, yes=False) → list[str]`**

Deletes black frames (HR + LR) and removes them from manifest. Interactive confirmation unless `--yes` flag is set.

### 6.5.6 PyTorch Dataset (`datasets.py`)

**`PairedImageFolderDataset(root_dir, split="train", scale=4, transforms=None)`**

Reads HR/LR pairs from `manifest.json` (or directory scan fallback). Returns `(lr_tensor, hr_tensor)` with shape `(C, H, W)` in range `[0, 1]`.

```python
def __getitem__(self, index) -> tuple[Tensor, Tensor]:
    hr_path, lr_path = self.pairs[index]
    hr = self._load_tensor(hr_path)
    lr = self._load_tensor(lr_path)
    if self.transform:
        lr, hr = self.transform(lr, hr)
    return lr, hr

def _load_tensor(self, path) -> Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR uint8
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
    return tensor.permute(2, 0, 1).contiguous()  # HWC → CHW
```

### 6.5.7 Transforms (`transforms.py`)

All transforms operate on `(lr, hr)` tuples simultaneously, ensuring aligned augmentation.

| Transform | Operation | Use case |
|-----------|-----------|----------|
| `RandomCrop(patch_size, scale)` | Crop same spatial region from LR and HR | Training — creates random patches |
| `CenterCrop(patch_size, scale)` | Center crop | Validation — deterministic |
| `RandomFlip(direction)` | Horizontal or vertical flip (50%) | Data augmentation |
| `RandomRotate(angles)` | 90/180/270° rotation | Data augmentation |
| `Compose(transforms)` | Chain multiple transforms | Standard pipeline |

```python
class Compose:
    def __call__(self, lr, hr):
        for t in self.transforms:
            lr, hr = t(lr, hr)
        return lr, hr
```

### Failure Modes

| Symptom | Cause | Resolution |
|---------|-------|------------|
| `FileNotFoundError: Could not open video` | Unsupported codec or corrupt file | Check codec support with `ffprobe` |
| Zero frames extracted | `start_time` > video duration | Adjust `--start-time` |
| `RuntimeError: Dataset validation failed` | Dimension mismatch or corrupt HR/LR | Run validation with verbose output to see specific problems |
| Black frame false positives | Very dark valid content (night footage) | Tune health threshold manually |
| Very slow dataset build | Large video, many frames | Use higher `--frame-rate` skip value, or split video |
| `OSError: No space left on device` | Disk full | LR images are 1/scale² the size of HR; estimate disk needs |

---

## 6.6 Model Registry and Architectures

**Location:** `src/sr_engine/models/registry.py`, `models/archs/rrdbnet.py`, `models/archs/swinir.py`

### Model Registry

The registry uses a **decorator-based pattern** for automatic registration:

```python
_registry: dict[str, type[nn.Module]] = {}

def register(name: str):
    def wrapper(cls):
        _registry[name] = cls
        return cls
    return wrapper

def build_model(name: str, config: dict) -> nn.Module:
    cls = _registry.get(name)
    if cls is None:
        raise ValueError(f"Unknown model: {name}")
    return cls(**config)
```

Models self-register at import time via the `@register` decorator. To add a new model, create a class in `models/archs/`, decorate it with `@register("name")`, and import it in `models/__init__.py`.

### RRDB (Residual-in-Residual Dense Block) — `rrdbnet.py`

**Registered as:** `"rrdb_esrgan"`

**Architecture:**
```
Input (B, 3, H, W)
  │
  └── Conv 3×3, num_feat
       │
       └── [RRDB Block × num_block]  ──┬──
            │                          │
            │ Each RRDB:               │
            │   dense_conv1 ──► cat    │
            │   dense_conv2 ──► cat    │
            │   dense_conv3 ──► + *0.2 │
            │                          │
            └──────────────────────────┘ (residual)
       │
       └── Conv 3×3 (post-body, residual add)
            │
            └── Upsample (nearest + 3×3 Conv) × 1 (for 4× scale)
                 │
                 └── Output (B, 3, H*scale, W*scale)
```

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_feat` | 64 | Base feature channels |
| `num_block` | 23 | Number of RRDB blocks |
| `num_grow_ch` | 32 | Growth channels per dense layer |
| `scale` | 4 | Upscaling factor |

**RRDB block internals:**
```python
class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        self.conv1 = Conv2d(nf, gc, 3)        # nf → gc
        self.conv2 = Conv2d(nf + gc, gc, 3)   # nf+gc → gc
        self.conv3 = Conv2d(nf + 2*gc, nf, 3) # nf+2*gc → nf

    def forward(self, x):
        x1 = lrelu(conv1(x))
        x2 = lrelu(conv2(cat(x, x1)))
        x3 = conv3(cat(x, x1, x2))
        return x3 * 0.2 + x  # Residual scaling
```

Residual scaling (0.2) stabilizes training by reducing the variance of the residual branch — a well-known trick from ESRGAN.

### SwinIR — `swinir.py`

**Registered as:** `"swinir"`

**Architecture:**
```
Input (B, 3, H, W)
  │
  └── Conv 3×3, embed_dim (shallow feature extraction)
       │
       └── [RSTB × 6 stages]
            │ Each RSTB:
            │   [SwinTransformerLayer × depth]
            │     ├── W-MSA (no shift)
            │     ├── SW-MSA (with cyclic shift)
            │     └── MLP (GELU, 2× expansion)
            │   Conv 3×3 residual connection
       │
       └── Conv 3×3 (post-body, residual add)
            │
            └── Conv 3×3 + LeakyReLU (pre-upsample)
                 │
                 └── PixelShuffle upsampler (Conv → PixelShuffle → Conv)
                      │
                      └── Output (B, 3, H*scale, W*scale)
```

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 180 | Embedding dimension after shallow conv |
| `depths` | [6,6,6,6,6,6] | SwinTransformerLayers per RSTB |
| `num_heads` | [6,6,6,6,6,6] | Attention heads per RSTB |
| `window_size` | 8 | Local window size for W-MSA/SW-MSA |
| `scale` | 4 | Upscaling factor |

**Window attention mechanism:**
```
window_partition(x, window_size)
  │
  └── Split (B, H, W, C) → (num_windows * B, window_size², C)
       │
       └── WindowAttention:
             ├── QKV projection (Linear, 3× expansion)
             ├── Scaled dot-product with relative position bias
             ├── Optional attention mask (for SW-MSA)
             └── Output projection (Linear)
       │
       └── window_reverse → (B, H, W, C)
```

**Relative position bias:** Learned parameters of shape `(2*window_size-1)², num_heads`. Computed once from relative coordinates and applied as a bias to attention scores. Provides translation-invariant position encoding within each window.

**Cyclic shift (SW-MSA):** Alternating layers shift the feature map by `(window_size//2, window_size//2)` pixels before window partitioning. This creates cross-window connections without increasing computation. An attention mask prevents undesired connections between shifted regions.

### Comparison

| Property | RRDB | SwinIR |
|----------|------|--------|
| Core operation | Convolution (3×3) | Self-attention (windowed) |
| Receptive field | Determined by depth (large) | Local window + cross-window via shifting |
| Memory | O(C*H*W) — linear | O(C*H*W + W²*H*W) — window attention dominates |
| Upsampler | Nearest + Conv | PixelShuffle or Nearest + Conv |
| Best for | Throughput, production | Quality, research, large-scale |

---

## 6.7 Training Engine

**Location:** `src/sr_engine/engine/trainer.py`

### Purpose

Orchestrates the complete training lifecycle: dataset loading, model instantiation, optimizer setup, epoch loop, validation, checkpointing, and callback dispatch.

### Responsibilities

- Build model from config via registry
- Create optimizer (Adam) and learning rate scheduler (cosine with linear warmup)
- Split dataset into train/validation sets
- Run epoch-based training loop with forward/backward/step
- Run validation at configurable frequency (PSNR, SSIM)
- Save checkpoints at configurable frequency
- Dispatch events to all attached callbacks
- Support training resume from checkpoint

### Trainer Constructor

```python
class Trainer:
    def __init__(
        self,
        model_cfg: dict,          # Model architecture config
        train_cfg: dict,          # Training hyperparameters
        dataset_dir: Path,         # Path to HR/LR dataset
        resume_from: Path | None,  # Resume checkpoint
        device: str,               # cuda/cpu/auto
        validation_enabled: bool,  # Enable validation split
        validation_split: float,   # Fraction for validation
        metrics_stream: MetricsStream | None,
        metrics_frequency: int,    # Steps between log events
        progress_reporter: ProgressReporter | None,
        callbacks: list[TrainerCallback] | None,
        cancel_check: Callable | None,
    )
```

### Training Loop

```python
def train(self) -> None:
    self._emit_phase("training")

    for epoch in range(self.current_epoch, self.max_epochs):
        # --- Training ---
        self.model.train()
        for batch, (lr, hr) in enumerate(self.train_loader):
            lr, hr = lr.to(self.device), hr.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type,
                                    dtype=autocast_dtype(self.device)):
                sr = self.model(lr)
                loss_pixel = self.pixel_loss(sr, hr)
                loss = loss_pixel

                if self.perceptual_weight > 0:
                    loss_perceptual = self.perceptual_loss(sr, hr)
                    loss = loss + self.perceptual_weight * loss_perceptual

                if self.gan_weight > 0:
                    # GAN loss requires discriminator — additional complexity
                    ...

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self._emit_step(epoch + 1, batch + 1, total_batches, loss, lr)

        self.scheduler.step()

        # --- Validation (every save_per_epoch) ---
        if (epoch + 1) % self.save_per_epoch == 0:
            self._validate(epoch + 1)
            self._save_checkpoint(epoch + 1)

    self._emit_phase("complete")
    self._emit_done(elapsed)
```

### Validation

```python
def _validate(self, epoch: int) -> None:
    self.model.eval()
    total_psnr, total_ssim = 0.0, 0.0
    num_batches = 0

    with torch.no_grad():
        for lr, hr in self.val_loader:
            lr, hr = lr.to(self.device), hr.to(self.device)
            sr = self.model(lr)
            total_psnr += psnr(sr, hr)
            total_ssim += ssim(sr, hr)
            num_batches += 1

    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    self._emit_validate(epoch, avg_psnr, avg_ssim)
```

### LR Schedule

Linear warmup for `warmup_steps` steps, then cosine annealing from `learning_rate` to `min_lr`:

```python
warmup_steps = int(self.train_cfg.get("warmup_steps", 1000))

class CosineLRWithWarmup:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr):
        ...

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.base_lr * step / self.warmup_steps
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + cos(π * progress))
```

### Callback System

```python
class TrainerCallback:
    def on_phase(self, phase: str, **data): ...     # training, saving, complete, cancelled
    def on_step(self, epoch, batch, total_batches, **losses): ...  # Every N batches
    def on_validate(self, epoch, **metrics): ...     # After each validation
    def on_done(self, elapsed_seconds): ...           # Training complete
```

Built-in callbacks:
- `_MetricsStreamCallback`: Writes JSONL metrics file
- `SocketCallback` (gui_bridge): Sends events over TCP to GUI
- CLI-injected callbacks from `resolve_callbacks()`

### Mixed Precision

Uses `torch.amp.autocast` with `GradScaler`:

```python
self.scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None

# Forward pass
with torch.amp.autocast(device_type=self.device.type, dtype=autocast_dtype(self.device)):
    sr = self.model(lr)
    loss = self.pixel_loss(sr, hr)

# Backward + step
self.scaler.scale(loss).backward()
self.scaler.step(self.optimizer)
self.scaler.update()
```

### Resume from Checkpoint

```python
if resume_from:
    checkpoint = load_checkpoint(resume_from, map_location=self.device)
    self.model.load_state_dict(checkpoint["state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state"])
    self.current_epoch = checkpoint["step"]  # or checkpoint["epoch"]
```

### Failure Modes

| Symptom | Cause | Resolution |
|---------|-------|------------|
| Loss = NaN | Learning rate too high, fp16 underflow, model instability | Reduce LR, enable gradient clipping, check fp16 loss scaling |
| Validation PSNR not improving | Overfitting, wrong degradation config, LR too low | Adjust LR, check degradation quality, increase dataset size |
| `CUDA out of memory` | Batch size + patch size too large for VRAM | Reduce batch size, reduce patch size, enable gradient accumulation |
| Resume checkpoint fails | Architecture mismatch, missing keys | Train with same model config as original |

---

## 6.8 Inference Engine

**Location:** `src/sr_engine/engine/inference.py`

### Purpose

Applies a trained model to images or videos, producing super-resolved output. Supports tiled inference for VRAM-limited GPUs.

### Image Inference

```python
def infer_image(model, image_path, output_path, device="cuda", tile=0, overlap=64):
    # 1. Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)  # BGR uint8
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # RGB

    # 2. Preprocess
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)

    # 3. Inference (with optional tiling)
    if tile > 0:
        sr = _tiled_inference(model, tensor, tile, overlap)
    else:
        with torch.no_grad():
            sr = model(tensor)

    # 4. Postprocess
    sr_np = sr.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    sr_np = np.clip(sr_np * 255.0, 0, 255).astype(np.uint8)
    sr_np = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)

    # 5. Save
    cv2.imwrite(str(output_path), sr_np)
```

### Video Inference

```python
def infer_video(model, video_path, output_path, device="cuda", tile=0, overlap=64):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * scale
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        sr_frame = infer_image(model, frame, ...)  # In-memory path
        writer.write(sr_frame)

    cap.release()
    writer.release()
```

### Input Type Detection

File extension determines inference mode:
- Images: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif` — single-frame SR
- Videos: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` — frame-by-frame upscaling

### Failure Modes

| Symptom | Cause | Resolution |
|---------|-------|------------|
| `FileNotFoundError: Unknown format` | Unsupported file extension | Use one of the supported image/video formats |
| Video output has wrong FPS | FPS metadata incorrectly read | Check source video properties |
| Tiling artifacts visible at seams | Overlap too small | Increase `--overlap` (default 64, try 128) |
| Very slow video inference | Frame-by-frame processing | Use GPU with higher VRAM, reduce tile overlap, batch frames if model supports it |

---

## 6.9 Loss Functions

**Location:** `src/sr_engine/models/losses.py`

### L1Loss (Charbonnier)

```python
class L1Loss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()
```

A differentiable approximation of L1 loss. Behaves like L1 for large errors but is smooth (differentiable) near zero. The `eps` parameter controls the transition from L1-like to L2-like behavior:
- Smaller `eps` → closer to true L1 (sharper minimum)
- Larger `eps` → smoother (more stable training)

### PerceptualLoss (VGG19-based)

```python
class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids: list[str] | None = None):
        # Default: ["relu5_4"]
        # VGG19 feature extractor, frozen, ImageNet-normalized

    def forward(self, pred, target):
        # Extract VGG19 features at specified layers
        # Compute L1 distance between pred and target features
        # Return weighted sum
```

Uses a pretrained VGG19 network as a fixed feature extractor. Loss is computed in feature space rather than pixel space, encouraging perceptually similar outputs.

**VGG19 layer indices:**
```python
_VGG19_LAYER_INDEX = {
    "relu1_1": 1, "relu1_2": 3,
    "relu2_1": 6, "relu2_2": 8,
    "relu3_1": 11, "relu3_2": 13, "relu3_3": 15, "relu3_4": 17,
    "relu4_1": 20, "relu4_2": 22, "relu4_3": 24, "relu4_4": 26,
    "relu5_1": 29, "relu5_2": 31, "relu5_3": 33, "relu5_4": 35,
}
```

The VGG backbone is truncated at the deepest requested layer to reduce computation. Input images in `[0, 1]` range are normalized with ImageNet statistics internally.

### GANLoss (Adversarial)

```python
class GANLoss(nn.Module):
    def __init__(self, gan_type: str = "vanilla"):
        # vanilla → BCEWithLogitsLoss
        # lsgan  → MSELoss

    def forward(self, pred, target_is_real: bool):
        target = 1.0 if target_is_real else 0.0
        return self.loss_fn(pred, target)
```

**Types:**
- `"vanilla"`: Standard GAN loss using `BCEWithLogitsLoss`. Discriminator outputs raw logits.
- `"lsgan"`: Least Squares GAN using `MSELoss`. Targets are 1.0 (real) / 0.0 (fake).

### Loss Combination in Training

```python
loss_pixel = self.pixel_loss(sr, hr) * pixel_weight
loss_perceptual = self.perceptual_loss(sr, hr) * perceptual_weight
loss_gan = self.gan_loss(discriminator(sr), True) * gan_weight

loss = loss_pixel + loss_perceptual + loss_gan
```

Typical weight ratios: `pixel:perceptual:gan = 1.0 : 0.1 : 0.005`

---

## 6.10 Metrics System

**Location:** `src/sr_engine/engine/metrics.py`, `engine/metrics_stream.py`

### PSNR

```python
def psnr(img1, img2, data_range=1.0) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    mse = F.mse_loss(img1, img2)
    return 10 * log10(data_range² / mse)
```

- Operating on Y channel (luminance) in YCbCr space by default
- MSE clamped to a minimum of `1e-10` to avoid division by zero
- Higher is better. Typical range: 20–40 dB for SR

### SSIM

```python
def ssim(img1, img2, data_range=1.0) -> float:
    """Structural Similarity Index."""
    # Gaussian-weighted statistics (mean, variance, covariance)
    # SSIM = (2*μ_x*μ_y + C1)(2*σ_xy + C2) / (μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)
```

- 1.0 = identical images
- Typical range for SR: 0.6–0.99

### LPIPS

```python
def lpips(img1, img2, net='alex') -> float:
    """Learned Perceptual Image Patch Similarity."""
    # Requires `lpips` package (lazily imported)
    # Uses pretrained AlexNet or VGG feature extractors
```

- Lower = more similar (perceptually)
- Typical range: 0.0–0.5
- Requires separate `pip install lpips`

### MetricsStream (JSONL Writer)

```python
class MetricsStream:
    def __init__(self, path: Path):
        self.file = open(path, 'w')

    def write(self, data: dict):
        # First line: JSON metadata comment
        # Subsequent lines: one JSON object per line
        json_line = json.dumps(data) + "\n"
        self.file.write(json_line)
        self.file.flush()

    def close(self):
        self.file.close()
```

File format:
```json
# experiment: run_001, model: swinir, seed: 42
{"type":"step","epoch":1,"batch":10,"total_batches":100,"pixel":0.05,"total":0.05,"lr":0.0001}
{"type":"phase","phase":"training","max_epochs":20}
{"type":"validate","epoch":1,"psnr":30.2,"ssim":0.89}
{"type":"done","elapsed_seconds":3600,"total_epochs":20}
```

---

## 6.11 Checkpointing and Export

**Location:** `src/sr_engine/models/checkpoint.py`

### Save Checkpoint

```python
def save_checkpoint(path, state_dict, ema_state_dict=None, optimizer_state=None,
                    step=0, config=None, backend_info=None):
```

Writes a `.pt` file containing:
- `state_dict` — model weights
- `ema_state_dict` — optional EMA weights
- `optimizer_state` — optimizer state (for resume)
- `step` — global training step
- `config` — complete model config (must include `"name"` key for rebuild)
- `backend_info` — device metadata

**Atomic write:** Writes to a `.tmp` path first, then renames. Prevents corrupted checkpoints on crash.

### Load Checkpoint

```python
def load_checkpoint(path, map_location=None, load_ema=False) -> dict:
```

Returns dict with `state_dict`, `step`, `config`, etc.
- Uses `weights_only=True` by default (safe from pickle exploits)
- Falls back to `weights_only=False` on older PyTorch versions
- If `load_ema=True`, replaces `state_dict` with `ema_state_dict`

### Export Formats

| Format | Function | Use Case |
|--------|----------|----------|
| **ONNX** | `export_to_onnx(ckpt_path, out_path, input_shape)` | Cross-platform inference, ONNX Runtime |
| **SafeTensors** | `export_to_safetensors(ckpt_path, out_path, load_ema)` | Safe weight distribution, HuggingFace |
| **TorchScript** | `export_to_torchscript(ckpt_path, out_path)` | C++ `libtorch` deployment |

**Export flow:**
```python
checkpoint = load_checkpoint(ckpt_path, map_location="cpu")
model = _build_model_from_checkpoint(checkpoint)  # Rebuild from saved config
model.eval()

if format == "onnx":
    torch.onnx.export(model, dummy_input, out_path, ...)
elif format == "safetensors":
    save_file(state_dict, out_path)  # No model rebuild needed
elif format == "torchscript":
    traced = torch.jit.trace(model, dummy_input)
    traced.save(out_path)
```

---

## 6.12 GUI Bridge

**Location:** `src/sr_engine/gui_bridge/`

### Purpose

Provides a TCP/NDJSON protocol for integrating sr-engine with a Godot (C#) GUI client. Supports synchronous queries (workspace info, dataset validation) and asynchronous jobs (training, inference, dataset building) with real-time progress streaming.

### Architecture

```
┌──────────────────────┐     TCP / NDJSON      ┌──────────────────────────────┐
│   Godot GUI (C#)     │◄──────────────────────►│   sr-engine Server           │
│                      │    persistent conn     │                              │
│  SrEngineClient      │    port 8765 (default) │  Server                      │
│    ↕ event-driven    │                        │    ├─ gui_listener :8765     │
│    ↕ _Process queue  │                        │    ├─ job_listener :random   │
│                      │                        │    ├─ ClientHandler(s)       │
│                      │                        │    ├─ ControlHandler(s)      │
│                      │                        │    └─ JobManager             │
└──────────────────────┘                        └──────────┬───────────────────┘
                                                           │
                                  TCP / NDJSON             │
                                  control socket           │
                                                           ▼
                                                  ┌──────────────────┐
                                                  │  Subprocess      │
                                                  │  (srengine       │
                                                  │   train/infer/   │
                                                  │   dataset.build) │
                                                  │                  │
                                                  │  SocketReporter  │
                                                  │  SocketCallback  │
                                                  └──────────────────┘
```

### Server (`server.py`)

```python
class Server:
    def __init__(self, workspace, host="127.0.0.1", gui_port=8765):
        self.workspace = workspace
        self.host = host
        self.gui_port = gui_port
        self.job_manager = JobManager(workspace)
        self._gui_clients = []
        self._control_clients = {}

    def start(self):
        # Accept GUI connections on gui_port
        # Accept control connections on random port
        # Dispatch handlers
```

### JobManager (`jobs.py`)

```python
class JobManager:
    def start_job(self, job_type: str, params: dict) -> str:
        job_id = f"{job_type}_{timestamp}_{random_suffix}"
        manifest = JobManifest(job_id=job_id, job_type=job_type, ...)
        self._save_manifest(manifest)

        # Build CLI args from params
        cli_args = cli_args_for(job_type, params)
        env = os.environ.copy()
        env["SRENGINE_GUI_SOCKET"] = json.dumps({
            "job_id": job_id,
            "token": token,
            "control_host": self.host,
            "control_port": self._control_port,
        })

        process = subprocess.Popen(cli_args, env=env, ...)
        return job_id

    def cancel_job(self, job_id: str):
        process = self._processes.get(job_id)
        if process:
            process.terminate()  # SIGTERM
```

### Wire Protocol

**Transport:** Raw TCP, NDJSON framing.

**Request:** `{"id": "<string>", "command": "<string>", "params": {...}}`

**Response types:**
- `{"id": "...", "type": "result", "data": {...}}` — synchronous command result
- `{"id": "...", "type": "accepted", "data": {"job_id": "..."}}` — async command accepted
- `{"id": "...", "type": "error", "message": "...", "error_type": "..."}` — error

**Unsolicited events (broadcast to all GUI clients):**
- `{"type": "progress_start", "total": N, "desc": "...", "job_id": "..."}`
- `{"type": "progress_update", "n": 1, "job_id": "..."}`
- `{"type": "progress_end", "job_id": "..."}`
- `{"type": "postfix", "desc": "...", "loss": 0.05, "job_id": "..."}`
- `{"type": "phase", "phase": "training", "max_epochs": 100, "job_id": "..."}`
- `{"type": "step", "epoch": 1, "batch": 10, "total": 0.05, "lr": 0.0001, "job_id": "..."}`
- `{"type": "validate", "epoch": 1, "psnr": 30.2, "ssim": 0.89, "job_id": "..."}`
- `{"type": "done", "exit_code": 0, "job_id": "..."}`
- `{"type": "log", "level": "info", "message": "...", "job_id": "..."}`

### Commands

**Synchronous:**
| Command | Purpose |
|---------|---------|
| `hello` | Handshake — returns schema_version, server_version |
| `workspace.info` | Returns resolved workspace path |
| `workspace.check` | Checks workspace exists |
| `project.list` | Lists workspace projects |
| `project.create` | Creates a new project |
| `dataset.validate` | Deep validation of dataset |
| `dataset.health` | Dataset profiling |
| `model.info` | Model config information |
| `job.cancel` | Cancel a running job |
| `job.list` | List all completed jobs |
| `job.status` | Get a specific job's status |

**Asynchronous (returns job_id immediately):**
| Command | Purpose |
|---------|---------|
| `train.start` | Start model training |
| `infer.start` | Start inference on image/video |
| `dataset.build` | Build dataset from video |

### Job Lifecycle

```
accept ─► pending ─► running ─► completed (exit 0)
                                ├── cancelled (exit 130)
                                └── failed (exit 1+)
```

- **pending**: Subprocess spawned, waiting for control socket connection (10s timeout)
- **running**: Control socket connected, events flowing
- **completed/cancelled/failed**: Exit code mapped to status, manifest persisted to `<workspace>/jobs/<job_id>.json`

### Subprocess Integration

When a subprocess starts, it:
1. Reads `SRENGINE_GUI_SOCKET` env var for connection info
2. Connects control socket, sends `hello` with job_id + token
3. Uses `SocketReporter` (instead of `TqdmReporter`) for progress
4. Uses `SocketCallback` (instead of no-op) for training events
5. Checks `was_cancelled()` periodically for early termination

---

## 6.13 Progress Reporting

**Location:** `src/sr_engine/utils/progress.py`

### Purpose

Abstract interface for reporting operation progress, with different implementations for terminal and GUI environments.

### ProgressReporter (Base)

```python
class ProgressReporter:
    def start(self, total: int | None, desc: str = ""): ...
    def update(self, n: int = 1): ...
    def finish(self): ...
    def set_description(self, desc: str): ...
    def set_postfix(self, **kwargs): ...
```

All methods are no-ops by default — subclasses override what they need.

### TqdmReporter

```python
class TqdmReporter(ProgressReporter):
    def start(self, total, desc=""):
        self._pbar = tqdm(total=total, desc=desc)

    def update(self, n=1):
        self._pbar.update(n)

    def finish(self):
        self._pbar.close()

    def set_description(self, desc):
        self._pbar.set_description(desc)

    def set_postfix(self, **kwargs):
        self._pbar.set_postfix(kwargs)
```

### SocketReporter

```python
class SocketReporter(ProgressReporter):
    def __init__(self, send_fn):
        self._send = send_fn

    def start(self, total, desc=""):
        self._send({"type": "progress_start", "total": total, "desc": desc})

    def update(self, n=1):
        self._send({"type": "progress_update", "n": n})

    def finish(self):
        self._send({"type": "progress_end"})

    def set_postfix(self, **kwargs):
        desc = kwargs.pop("desc", "")
        self._send({"type": "postfix", "desc": desc, **kwargs})
```

---

## 6.14 Tiling System

**Location:** `src/sr_engine/engine/tiling.py`

### Purpose

Enables handling of arbitrarily large images on VRAM-constrained GPUs by splitting the input into overlapping tiles, running inference on each tile independently, and blending the results.

### tile_image()

```python
def tile_image(image: Tensor, tile_size: int = 512, overlap: int = 64) -> list[tuple[Tensor, int, int]]:
```

Splits `(C, H, W)` image into overlapping tiles:
```
num_tiles_x = ceil((W - overlap) / (tile_size - overlap))
num_tiles_y = ceil((H - overlap) / (tile_size - overlap))
```

Edge tiles are padded with reflection padding to reach `tile_size × tile_size`.

### stitch_tiles()

```python
def stitch_tiles(tiles: list[tuple[Tensor, int, int]], output_size: tuple, overlap: int = 64) -> Tensor:
```

- Places each tile at its (scaled) position
- Creates linear ramp weight map: 0→1 in overlap regions
- Blends overlapping areas using normalized weighted average
- Crops any padding added during tiling

**Weight map formula (1D):**
```
In overlap region of width `overlap`:
  weight(distance) = distance / overlap  (in → out)
```

The 2D weight is the product of x and 1D weights, normalized by the sum across all overlapping tiles.

### VRAM Savings

| GPU VRAM | Max full-frame (SwinIR 4×) | Tiled (512px, 64 overlap) |
|----------|---------------------------|---------------------------|
| 8 GB | ~1280×720 | 4K+ inputs |
| 16 GB | ~1920×1080 | 8K+ inputs |
| 24 GB | ~3840×2160 | Any resolution |

### Tile count examples (1920×1080, tile=512, overlap=64):
```
num_tiles_x = ceil((1920 - 64) / (512 - 64)) = 5
num_tiles_y = ceil((1080 - 64) / (512 - 64)) = 3
total_tiles = 15
```

---

# Internal Mechanics

## 7.1 Training Loop Lifecycle

### Full State Machine

```
INIT ──► SETUP ──► TRAINING ──► COMPLETE
                       │              │
                       ├──► SAVING ───┤
                       │       │      │
                       │       ◄──────┘
                       │
                       └──► CANCELLED ──► CLEANUP
```

### Per-Epoch Breakdown

```
for epoch in range(start_epoch, max_epochs):
    1. model.train()
    2. Shuffle train_loader (DataLoader shuffle=True)
    ┌──────────────────────────────────────┐
    │ for batch, (lr, hr) in train_loader: │
    │   lr, hr = lr.to(device), hr.to(device)
    │   optimizer.zero_grad()
    │   with autocast:
    │     sr = model(lr)
    │     loss = pixel_loss(sr, hr)
    │     if perceptual: loss += perceptual_loss(sr, hr) * weight
    │     if gan: loss += gan_loss(discriminator(sr), True) * weight
    │   scaler.scale(loss).backward()
    │   scaler.step(optimizer)
    │   scaler.update()
    │   if batch % metrics_frequency == 0:
    │     CALLBACK: on_step(epoch, batch, loss, lr)
    │   if cancel_check(): raise CancelledError
    └──────────────────────────────────────┘
    3. scheduler.step()
    4. if epoch % save_per_epoch == 0:
         CALLBACK: on_phase("saving")
         run_validation()
         save_checkpoint()
    └──────────────────────────────────────┘
```

### Data Flow Per Batch

```
DataLoader ──► (lr_tensor, hr_tensor) ──► to(device)
                     │
                     ▼
         model(lr) ──► sr_tensor
                     │
                     ▼
    pixel_loss(sr, hr) ──► loss_pixel
    perceptual_loss(sr, hr) ──► loss_perceptual (optional)
                     │
                     ▼
            loss = weighted_sum
                     │
                     ▼
          scaler.scale(loss).backward()
                     │
                     ▼
          scaler.step(optimizer)
                     │
                     ▼
          CALLBACK: on_step(epoch, batch, losses)
```

### Gradient Flow

```
loss.backward()
  │
  ├── d(loss_pixel)/d(sr) · d(sr)/d(θ_model)
  ├── d(loss_perceptual)/d(sr) · d(sr)/d(θ_model)  (if enabled)
  └── d(loss_gan)/d(sr) · d(sr)/d(θ_model)          (if enabled)

optimizer.step()
  └── θ_model ← θ_model - lr · ∇θ_model
```

## 7.2 Degradation Pipeline Execution

### Per-Worker Execution

```
Worker Process (one per CPU core)
  │
  ├── _init_worker()  ───► set OpenCV threads=1, seed RNG
  │
  │   (for each frame assigned by pool)
  │
  ├── hr_img = cv2.imread(hr_path)   # BGR uint8
  │
  ├── _degrade_image(hr_img, scale, **stage_kwargs)
  │     ├── crop to scale-multiple (in-place)
  │     ├── color_jitter? → HSV shift
  │     ├── blur? → Gaussian or Motion (exclusive)
  │     ├── antialias? → GaussianBlur(sigma=0.5)
  │     ├── downsample → cv2.resize(lr_size)
  │     ├── noise? → Gaussian, Poisson, or S&P (exclusive)
  │     ├── jpeg? → imencode/imdecode .jpg
  │     └── jpeg2000? → imencode/imdecode .jp2
  │
  ├── cv2.imwrite(lr_path, lr_img)
  │
  └── return (hr_path, lr_path)
```

### Process Pool Coordination

```
Main Process
  │
  ├── Create ProcessPoolExecutor(max_workers=N)
  │     N = min(len(hr_paths), os.cpu_count() or 4)
  │
  ├── Submit all tasks
  │     futures = [executor.submit(worker, path) for path in hr_paths]
  │
  ├── Collect as completed (for progress reporting)
  │     for future in as_completed(futures):
  │         hr_path, lr_path = future.result()
  │         reporter.update()
  │
  └── Shutdown pool
        executor.shutdown()
```

### Seed Management for Reproducibility

```python
def _init_worker():
    cv2.setNumThreads(1)  # Avoid N_cpu × N_OpenCV_thread oversubscription
    pid = os.getpid()
    rand_add = int.from_bytes(os.urandom(4), "little")
    seed = pid + rand_add
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
```

Each worker's RNG state is deterministic given the PID + random additive. The additive ensures different runs get different degradation, while the PID component ensures processes don't share seeds. For fully deterministic degradation, seed `random` and `np.random` with a fixed value per frame ID instead.

## 7.3 Config Merge Resolution

### Full Resolution Order

```
1. DefaultConfigs.__init__()
     │
     ├── Load all YAMLs from utils/configs/ recursively
      │     utils/configs/train/base.yaml
     │     utils/configs/datasets/video_pairs.yaml
     │     utils/configs/models/swinir.yaml
     │     utils/configs/models/rrdb_esrgan.yaml
     │
     └── Store in self.builtins dict (filename → parsed dict)

2. get_train_config(config_path=None)
     │
     ├── Level 1: cfg = deepcopy(self.builtins["train/base.yaml"])
     │
     ├── Level 2: if workspace:
     │     ws_cfg = workspace.path / "configs/train/base.yaml"
     │     if ws_cfg.exists():
     │         cfg = merge_overrides(cfg, load_yaml(ws_cfg))
     │
     ├── Level 3: if config_path:
     │     cfg = merge_overrides(cfg, load_yaml(config_path))
     │
     └── Return cfg

3. apply_cli_overrides(cfg, cli_kwargs)
     │
     ├── Level 4: cfg = merge_overrides(cfg, cli_kwargs)
     │
     └── Return final cfg
```

### merge_overrides Algorithm

```python
def merge_overrides(base: dict, overrides: dict) -> dict:
    result = base.copy()
    for key, value in overrides.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_overrides(result[key], value)
            else:
                result[key] = value
        else:
            result[key] = value
    return result
```

**Key behavior:**
- Dict values recurse (nested overrides)
- Scalar values replace entirely
- New keys are added (not ignored)
- Lists are replaced (not appended)

### Config Resolution for Training

```
CLI: srengine train run --dataset my_set --model swinir --batch-size 4
  │
  ├── DefaultConfigs(workspace=discovered_ws)
  │
  ├── cfg_train = get_train_config()  ← Levels 1-3
  │     ├── builtins/train/base.yaml
  │     ├── ws/configs/train/base.yaml (if exists)
  │     └── (no --config given, skip Level 3)
  │
  ├── cfg_model = get_model_config("swinir")
  │     ├── builtins/models/swinir.yaml
  │     └── ws/configs/models/swinir.yaml (if exists)
  │
  ├── cfg_dataset = get_dataset_config()
  │     (same pattern)
  │
  ├── cfg_full = {**cfg_train, **cfg_model, **cfg_dataset, "name": "swinir"}
  │
  ├── apply_cli_overrides(cfg_full, {"batch_size": 4})
  │     └── cfg_full["batch_size"] = 4
  │
  └── validate_config(cfg_full)
```

## 7.4 Device Detection Flow

```
get_device("auto")
  │
  ├── torch.cuda.is_available()?
  │     │
  │     ├── Yes:
  │     │   ├── is_rocm()? → hasattr(torch.version, 'hip')
  │     │   │     ├── True:  ROCm detected
  │     │   │     └── False: CUDA detected
  │     │   │
  │     │   └── return torch.device("cuda:0")
  │     │
  │     └── No:
  │         └── return torch.device("cpu")
  │
  └── (explicit request: validate and return or raise)
```

### Autocast Dtype Selection

```
autocast_dtype(device)
  │
  ├── device.type != "cuda" → return torch.float32
  │
  └── device.type == "cuda":
        ├── torch.cuda.is_bf16_supported()?
        │     ├── True → return torch.bfloat16
        │     └── False → return torch.float16
        └── (ROCm path also checks torch.version.hip)
```

## 7.5 GUI Bridge Request Lifecycle

### Synchronous Command

```
Client                          Server
  │                               │
  ├── connect() ─────TCP──────►   │
  │                               ├── accept() → ClientHandler
  │                               │
  ├── send("hello") ──────────►   │
  │                               ├── _handle_hello()
  │                               ├── returns {"schema_version": 1, ...}
  │   ◄──── result ────────────   │
  │                               │
  ├── send("project.list") ───►   │
  │                               ├── _handle_project_list()
  │                               ├── workspace.list_projects()
  │                               ├── returns {"projects": [...]}
  │   ◄──── result ────────────   │
  │                               │
```

### Asynchronous Command (e.g., train.start)

```
Client            Server                         Subprocess
  │                 │                                │
  ├── train.start ─►│                                │
  │                 ├── JobManager.start_job()       │
  │                 │   ├── Generate job_id + token  │
  │                 │   ├── Persist manifest         │
  │                 │   ├── Build CLI args           │
  │                 │   ├── Set SRENGINE_GUI_SOCKET  │
  │                 │   └── subprocess.Popen(...) ──►│
  │                 │                                │
  │   ◄── accepted ─┤                                │
  │                 │                                │
  │                 │   (10s timeout)                │
  │                 │   ◄── connect control socket ──┤
  │                 │         hello with job_id+tok  │
  │                 │         status=ok ────────────►│
  │                 │                                │
  │                 │   ◄── progress events ─────────┤
  │   ◄── broadcast ─┤    (SocketReporter)            │
  │                 │                                │
  │                 │   ◄── training events ─────────┤
  │   ◄── broadcast ─┤    (SocketCallback)            │
  │                 │                                │
  │                 │   ◄── exit ────────────────────┤
  │                 ├── Map exit code → status       │
  │                 ├── Persist final manifest       │
  │   ◄── broadcast ─┤    done event                  │
  │                 │                                │
```

### Message Format

```
Request:  {"id":"req_1","command":"train.start","params":{"model_name":"swinir",...}}\n
          │     │        │                  │                              │
          └─────┴────────┴──────────────────┴──────────────────────────────┘
          Newline-delimited JSON over raw TCP

Response: {"id":"req_1","type":"accepted","data":{"job_id":"train_1747000000_a1b2"}}\n

Event:    {"type":"step","epoch":1,"batch":10,"total":0.05,"lr":0.0001,"job_id":"train_..."}\n
```

## 7.6 Tiled Inference Flow

```
Input image (C, H, W)
  │
  ├── Determine tile grid:
  │     tiles_x = ceil((W - overlap) / (tile_size - overlap))
  │     tiles_y = ceil((H - overlap) / (tile_size - overlap))
  │
  ├── For each tile (ix, iy):
  │     │
  │     ├── Calculate crop region:
  │     │     x_start = ix * (tile_size - overlap)
  │     │     y_start = iy * (tile_size - overlap)
  │     │     x_end = min(x_start + tile_size, W)
  │     │     y_end = min(y_start + tile_size, H)
  │     │
  │     ├── Extract tile: image[:, y_start:y_end, x_start:x_end]
  │     │
  │     ├── Pad to tile_size×tile_size:
  │     │     pad_right = tile_size - (x_end - x_start)
  │     │     pad_bottom = tile_size - (y_end - y_start)
  │     │     tile = F.pad(tile, (0, pad_right, 0, pad_bottom), mode='reflect')
  │     │
  │     ├── Run inference:
  │     │     sr_tile = model(tile.unsqueeze(0).to(device))
  │     │     sr_tile = sr_tile.squeeze(0).cpu()
  │     │
  │     └── Store (sr_tile, x_start * scale, y_start * scale)
  │
  ├── Create output canvas (C, H*scale, W*scale), zero-filled
  ├── Create weight canvas (1, H*scale, W*scale), zero-filled
  │
  ├── For each tile:
  │     │
  │     ├── Create per-tile weight map:
  │     │     weight = ones(tile_size*scale, tile_size*scale)
  │     │     Apply linear ramp in overlap regions:
  │     │       left overlap:   weight[:, :overlap*scale] *= x / (overlap*scale)
  │     │       right overlap:  weight[:, -overlap*scale:] *= (max-x) / (overlap*scale)
  │     │       (same for top/bottom)
  │     │
  │     ├── Add weighted tile to output: output += sr_tile * weight
  │     └── Add weight to total: weight_sum += weight
  │
  ├── Normalize: output /= weight_sum (avoid division by zero with epsilon)
  │
  └── Crop to original output dimensions: output[:, :H*scale, :W*scale]
```

### Weight Map Detail (1D cross-section)

```
    weight = 1.0         weight = 1.0
    ┌─────────┐          ┌─────────┐
    │         │          │         │
    │  Tile A │  linearly│  Tile B │
    │         │  ramping │         │
    └─────────┘    to    └─────────┘
    ↑         ↑ 1.0→0.0  ↑         ↑
    |←─ 64 ─→|    +      |←─ 64 ─→|
              0.0→1.0

    Combined weight (sum of A + B weights):
    ┌────────────────────────────────────┐
    │         ┌──────────────────┐        │
    │  Tile A │   1.0 + 0.0→1.0 │ Tile B │
    │         └──────────────────┘        │
    └────────────────────────────────────┘
    Result after normalization:
    ┌────────────────────────────────────┐
    │         ┌──────────────────┐        │
    │  Tile A │  (Tile_A + Tile_B│ Tile B │
    │         │     weighted avg)│        │
    └────────────────────────────────────┘
```

---

# Data Model

## Entities and Schemas

### Dataset Structure

```
<dataset_dir>/
├── HR/
│   ├── 000001.png        # Original extracted frame (H×W)
│   ├── 000002.png
│   └── ...
├── LR/
│   ├── 000001.png        # Degraded counterpart (H/scale × W/scale)
│   ├── 000002.png
│   └── ...
└── manifest.json
```

### manifest.json Schema

```json
{
  "config": {
    "scale": 4,
    "frame_rate": 24,
    "video_source": "input_video.mp4"
  },
  "pairs": [
    {"hr": "HR/000001.png", "lr": "LR/000001.png"},
    {"hr": "HR/000002.png", "lr": "LR/000002.png"}
  ]
}
```

### Workspace Structure

```
<workspace>/
├── .sr_workspace                        # JSON: {"version":1,"created":"/path"}
├── configs/
│   ├── train/base.yaml                  # User overrides for train config
│   ├── datasets/video_pairs.yaml        # User overrides for dataset config
│   └── models/{swinir,rrdb_esrgan}.yaml # User overrides for model config
├── datasets/<name>/
│   ├── HR/
│   ├── LR/
│   └── manifest.json
├── models/<instance>/
│   ├── config.yaml
│   ├── versions/
│   ├── checkpoints/epoch_*.pt
│   └── runs/run_<ts>/
└── jobs/<job_id>.json
```

### Checkpoint (.pt) Schema

```python
checkpoint = {
    "state_dict": OrderedDict,          # Model weights (str → Tensor)
    "ema_state_dict": OrderedDict,      # Optional EMA weights
    "optimizer_state": dict,            # Optimizer state for resume
    "step": int,                        # Global training step
    "config": {                         # Model architecture config
        "name": str,                    # Registry name (e.g., "swinir")
        ...                             # Architecture-specific params
    },
    "backend_info": str,                # Device metadata
}
```

### Job Manifest Schema

```python
@dataclass
class JobManifest:
    job_id: str
    job_type: str                   # "train", "infer", "dataset.build"
    status: str                     # "pending", "running", "completed", "cancelled", "failed"
    pid: int | None
    started_at: str | None          # ISO datetime
    finished_at: str | None
    exit_code: int | None
    project: str | None
    log_path: str | None
    error_message: str | None
```

### MetricsStream File Format (JSONL)

Line 1 (comment header):
```
# experiment: run_001, model: swinir, seed: 42, started_at: 2025-05-16T12:00:00
```

Subsequent lines (one JSON object per line):
```json
{"type":"step","epoch":1,"batch":10,"total_batches":100,"pixel":0.05,"perceptual":0.01,"total":0.06,"lr":0.0001}
{"type":"validate","epoch":1,"psnr":30.2,"ssim":0.89}
{"type":"phase","phase":"training","max_epochs":100}
{"type":"phase","phase":"complete"}
{"type":"done","elapsed_seconds":3600.5,"total_epochs":100}
```

## Relationships

```
Workspace 1──N Dataset
Workspace 1──N Project
Project 1──N ModelInstance
ModelInstance 1──N Checkpoint
ModelInstance 1──N Run
Run 1──1 MetricsStream (.jsonl)

CLI_Command 1──1 Config_Merge (4 levels → 1 dict)
Config_Merge 1──1 Model (via registry.build_model)
Config_Merge 1──1 Trainer (via Trainer.__init__)
Config_Merge 1──1 Degradation (via batch_degrade config param)

GUI_Server 1──N GUI_Client
GUI_Server 1──N Job
Job 1──1 Subprocess
Job 1──N GUI_Event (broadcast)
```

## Storage Patterns

| Data | Format | Location | Access Pattern |
|------|--------|----------|---------------|
| Training images | PNG (lossless) | `<dataset>/HR/`, `<dataset>/LR/` | Sequential read (DataLoader) |
| Dataset index | JSON | `<dataset>/manifest.json` | Read once at Dataset init |
| Model weights | `.pt` (pickle) | `<project>/checkpoints/` | Write N times, load once |
| Training metrics | JSONL | `<project>/metrics/` | Append-only write |
| Configs | YAML | `utils/configs/`, `<ws>/configs/` | Read once at startup |
| Job state | JSON | `<ws>/jobs/<id>.json` | Write twice (create + finalize) |
| Model instances | YAML + `.pt` | `<ws>/models/<m>/` | CRUD via CLI |

## Consistency Models

- **Dataset building**: Process pool workers write independent LR files. Main process writes manifest after all workers complete. Manifest is the source of truth.
- **Checkpoint saving**: Atomic via `.tmp` + rename. If crash during write, `.tmp` is discarded and no checkpoint corruption occurs.
- **Metrics streaming**: Append-only JSONL. Each write is flushed to disk. Partial writes (on crash) may lose the last line but never corrupt prior data.
- **Job manifests**: Written atomically (create + final update). Persisted across server restarts.

---

# APIs / Interfaces

## Python API

### Internal Public API Surface

The following classes and functions constitute the public API. Internal functions (prefixed with `_`) should not be imported from outside their module.

**Workspace:**
```python
Workspace(path)                        # Create workspace handle
Workspace.discover()                   # Auto-discover workspace from CWD
workspace.init(reset_configs=False)    # Create workspace structure
workspace.check() → dict              # Health check
workspace.create_project(name) → Project
workspace.list_projects() → list[Project]
workspace.resolve_dataset(path) → Path
workspace.create_model_instance(project, name, config) → ModelInstance
```

**Config:**
```python
DefaultConfigs(workspace=None)
configs.get_train_config(config_path=None) → dict
configs.get_model_config(name, config_path=None) → dict
configs.get_dataset_config(config_path=None) → dict
merge_overrides(base, overrides) → dict
validate_config(config)
```

**Models:**
```python
register(name) → decorator
build_model(name, config) → nn.Module
save_checkpoint(path, state_dict, ...)
load_checkpoint(path, map_location=None, load_ema=False) → dict
export_to_onnx(ckpt_path, out_path, input_shape)
export_to_safetensors(ckpt_path, out_path, load_ema=False)
export_to_torchscript(ckpt_path, out_path)
L1Loss(eps=1e-6)
PerceptualLoss(layer_ids=["relu5_4"])
GANLoss(gan_type="vanilla")
```

**Data:**
```python
extract_frames(video_path, out_dir, frame_rate, ...) → list[Path]
batch_degrade(hr_paths, lr_dir, scale, config, reporter) → list[tuple]
build_from_video(video_path, output_dir, config, reporter) → Path
build_from_preprocessed(input_dir) → Path
validate(dataset_dir) → ValidationReport
check_dataset_health(dataset_dir) → dict
prune_black_frames(dataset_dir, threshold, yes) → list[str]
PairedImageFolderDataset(root_dir, split, scale, transforms)
```

**Engine:**
```python
Trainer(model_cfg, train_cfg, dataset_dir, ...)
trainer.add_callback(callback)
trainer.train()
infer_image(model, image_path, output_path, device, tile, overlap)
infer_video(model, video_path, output_path, device, tile, overlap)
tile_image(image, tile_size, overlap) → list[tuple]
stitch_tiles(tiles, output_size, overlap) → Tensor
psnr(img1, img2) → float
ssim(img1, img2) → float
lpips(img1, img2) → float
MetricsStream(path)
```

**Device:**
```python
get_device(preferred="auto") → torch.device
is_rocm() → bool
get_device_name() → str
get_vram() → int
get_vram_used() → int
autocast_dtype(device) → torch.dtype
supports_flash_attn(device) → bool
scaled_dot_product_attention(q, k, v, ...) → Tensor
get_conv2d(in_ch, out_ch, kernel_size, ...) → nn.Conv2d
```

**Progress:**
```python
ProgressReporter()                     # Base (no-op)
TqdmReporter()                         # Terminal progress bar
SocketReporter(send_fn)                # TCP progress events
```

**GUI Bridge:**
```python
Server(workspace, host, gui_port)
server.start()
JobManager(workspace)
job_manager.start_job(job_type, params) → str
job_manager.cancel_job(job_id)
SocketReporter(send_fn)
SocketCallback(send_fn)
connect_control_socket() → (job_id, send_fn, close_fn)
make_json_sender(writer) → callable
parse_message(data) → dict
```

## CLI Interface

All CLI commands and their flags are documented in [docs/cli-reference.md](cli-reference.md). Key commands:

```bash
srengine workspace init           # Initialize a workspace
srengine project create <name>    # Create a project
srengine dataset build -i <video> # Build dataset from video
srengine train run -d <dataset> -m <model>  # Train a model
srengine infer run -m <ckpt> -i <input> -o <output>  # Inference
srengine model export -m <name> -c <ckpt> -f <fmt> -o <out>  # Export
srengine env check                # Hardware diagnostics
srengine serve start --port 8765  # Start GUI bridge server
```

## GUI Bridge Protocol (TCP/NDJSON)

**Wire format:** Newline-delimited JSON over raw TCP.

**Request:**
```json
{"id": "req_1", "command": "hello"}
{"id": "req_2", "command": "train.start", "params": {"model_name": "swinir", "dataset": "/data/my_set"}}
```

**Response:**
```json
{"id": "req_1", "type": "result", "data": {"schema_version": 1, "server_version": "0.1.0"}}
{"id": "req_2", "type": "accepted", "data": {"status": "accepted", "job_id": "train_1747000000_a1b2"}}
```

**Error:**
```json
{"id": "req_1", "type": "error", "message": "Unknown command: foo", "error_type": "KeyError"}
```

**Events (broadcast):**
```json
{"type": "phase", "phase": "training", "max_epochs": 100, "job_id": "train_..."}
{"type": "step", "epoch": 1, "batch": 10, "total_batches": 100, "total": 0.05, "lr": 0.0001, "job_id": "train_..."}
{"type": "validate", "epoch": 1, "psnr": 30.2, "ssim": 0.89, "job_id": "train_..."}
{"type": "done", "exit_code": 0, "elapsed_seconds": 3600.0, "job_id": "train_..."}
{"type": "log", "level": "info", "message": "Epoch 1/100 started", "job_id": "train_..."}
```

### C# Client (Godot 4.x)

A complete C# client implementation is provided in [docs/gui_bridge.md](gui_bridge.md). Key patterns:

```csharp
public async Task<string> StartTrainAsync(Dictionary<string, JsonElement> paramsDict)
{
    var result = await SendCommandAsync("train.start", paramsDict);
    return result["job_id"].GetString();
}

public override void _Process(double delta)
{
    while (_incoming.TryDequeue(out var msg))
        DispatchMessage(msg);
}
```

---

# Installation and Setup

## Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11–3.13 | Strictly tested on 3.11 |
| uv | ≥0.4 | Package manager |
| Linux | Any | ROCm requires Linux |
| macOS/Windows | Any | CPU-only or CUDA only |
| GPU (optional) | NVIDIA CUDA or AMD ROCm | See backend table |

## Environment Setup

### Quick Install

```bash
# CPU-only (no GPU, works everywhere)
./envs/build.sh --backend cpu

# NVIDIA CUDA
./envs/build.sh --backend cuda

# AMD ROCm (Linux only)
./envs/build.sh --backend rocm
```

The build script:
1. Creates `.venv` with `uv venv`
2. Runs `uv sync` to install runtime dependencies
3. Installs PyTorch with the correct index URL for the backend
4. Runs `envs/verify_env.py` to confirm everything works

### Manual Install

```bash
uv venv
uv sync
# Install PyTorch for your backend:
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
# or: uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
# or: uv pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchvision
```

### Dev Install

```bash
uv sync --group dev
# Installs: pytest, pytest-cov, pytest-mock, ruff
```

### PyTorch Backend Selection

| `--backend` | PyTorch Index | CUDA/ROCm Version |
|-------------|---------------|-------------------|
| `cpu` | `https://download.pytorch.org/whl/cpu` | None |
| `cuda` | `https://download.pytorch.org/whl/cu124` | CUDA 12.4 |
| `rocm` | `https://download.pytorch.org/whl/rocm6.2` | ROCm 6.2 |

PyTorch is **explicitly excluded** from `pyproject.toml` dependencies. It is installed separately to avoid:
- Downloading CUDA libraries on ROCm systems
- Downloading CUDA libraries on CPU-only systems
- Version conflicts between PyTorch and system CUDA/ROCm drivers

## Workspace Initialization

```bash
# Initialize a workspace in the current directory
srengine workspace init

# Initialize in a specific directory
srengine workspace init --path /data/my_workspace

# Verify
srengine workspace check
```

This creates:
```
<path>/
├── .sr_workspace
├── datasets/
├── models/
├── experiments/
├── jobs/
└── configs/
    ├── train/
    ├── datasets/
    └── models/
```

## Verification

```bash
# Check hardware capabilities
srengine env check

# Benchmark model throughput
srengine env bench --model swinir --batch-size 4 --iterations 20

# Initialize workspace
srengine workspace init

# Create a project
srengine project create my_experiment

# Create a dataset
srengine dataset build --input video.mp4

# Train (quick test)
srengine train run --dataset my_set --model rrdb_esrgan --max-epochs 2
```

---

# Configuration Reference

## ~~Global Defaults (`default.yaml`)~~ *(deleted)*

The `default.yaml` file was removed — its values were distributed to the specific
config files they belong to (see `train/base.yaml`, model configs, and CLI defaults below).

## Training Config (`train/base.yaml`)

```yaml
seed: 42
batch_size: 32
num_workers: 4
patch_size: 128
learning_rate: 2e-4
weight_decay: 0.0
betas: [0.9, 0.99]
max_epochs: 10
save_per_epoch: 5
checkpoint_dir: "experiments/checkpoints"
validation:
  enabled: true
  split: 0.1
lr_scheduler: cosine
warmup_steps: 2000
min_lr: 1e-7
dtype: float32
losses:
  perceptual_weight: 0.1
```

## Dataset Config (`datasets/video_pairs.yaml`)

```yaml
scale: 4
frame_rate: 10
frame_format: png
start_time: 0.0
duration: null
degradation:
  color_jitter:
    enabled: false
    hue_range: [-0.05, 0.05]
    saturation_range: [-0.3, 0.3]
    value_range: [-0.3, 0.3]
    prob: 0.8
  blur:
    enabled: true
    gaussian:
      kernel_size: 21
      sigma: [0.1, 3.0]
      prob: 1.0
    motion:
      max_kernel_size: 31
      prob: 0.5
  resize:
    method: area
    antialias: true
  noise:
    enabled: false
    gaussian:
      sigma_range: [1, 30]
      prob: 0.5
    poisson:
      scale_range: [0.05, 3.0]
      prob: 0.5
    salt_pepper:
      amount: 0.01
      salt_vs_pepper: 0.5
      prob: 0.3
  jpeg:
    enabled: true
    quality_range: [30, 95]
    prob: 1.0
  jpeg2000:
    enabled: false
    quality_range: [30, 95]
    prob: 0.5
```

## Model Configs

### SwinIR (`models/swinir.yaml`)
```yaml
name: swinir
num_in_ch: 3
num_out_ch: 3
embed_dim: 180
depths: [6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6]
window_size: 8
mlp_ratio: 2.0
img_range: 1.0
upsampler: pixelshuffle
scale: 4
```

### RRDB ESRGAN (`models/rrdb_esrgan.yaml`)
```yaml
name: rrdb_esrgan
num_in_ch: 3
num_out_ch: 3
num_feat: 64
num_block: 23
num_grow_ch: 32
scale: 4
```

## CLI Flag Overrides

Any config key can be overridden via CLI flag. The flag name derives from the config key by replacing `_` with `-`:

| Config Key | CLI Flag | Example |
|------------|----------|---------|
| `batch_size` | `--batch-size` | `--batch-size 4` |
| `learning_rate` | `--learning-rate` | `--learning-rate 1e-4` |
| `max_epochs` | `--max-epochs` | `--max-epochs 50` |
| `patch_size` | `--patch-size` | `--patch-size 64` |
| `num_workers` | `--num-workers` | `--num-workers 8` |
| `validation.split` | `--validation-split` | `--validation-split 0.15` |
| `losses.perceptual_weight` | `--perceptual-weight` | `--perceptual-weight 0.05` |

---

# Usage Guide

## Beginner Examples

### 1. Environment Check

```bash
srengine env check
# PyTorch version:  2.5.0+cu124
# Detected device:  cuda:0
# Device name:      NVIDIA GeForce RTX 4090
# VRAM total:       24564 MB
# BF16 support:     True
# Autocast dtype:   torch.bfloat16
```

### 2. Initialize a Workspace

```bash
cd /data
srengine workspace init
srengine workspace info
```

### 3. Build a Dataset from Video

```bash
srengine dataset build --input my_video.mp4
# Dataset created at: /data/datasets/my_video/
```

### 4. Train a Model (Quick Test)

```bash
srengine train run \
  --dataset my_video \
  --model rrdb_esrgan \
  --max-epochs 2 \
  --batch-size 2
```

### 5. Run Inference

```bash
srengine infer run \
  --model models/default/checkpoints/epoch_002.pt \
  --input test_image.png \
  --output sr_image.png
```

## Intermediate Examples

### Custom Dataset Degradation

```bash
# JPEG only — simulate old photo compression
srengine dataset build \
  --input video.mp4 \
  --degradations jpeg \
  --output ./datasets/jpeg_only

# Blur + noise + JPEG with Lanczos downsampling
srengine dataset build \
  --input video.mp4 \
  --degradations blur,noise,jpeg \
  --resize-method lanczos \
  --output ./datasets/balanced

# Color jitter + JPEG2000 (unusual but useful for research)
srengine dataset build \
  --input video.mp4 \
  --degradations color-jitter,jpeg2000
```

### Training with Custom Config

```bash
# Create a custom train config
cat > my_train.yaml << 'EOF'
max_epochs: 200
learning_rate: 1e-4
batch_size: 8
patch_size: 96
losses:
  pixel_weight: 1.0
  perceptual_weight: 0.01
  perceptual_layers:
    - relu5_4
    - relu4_4
validation:
  split: 0.05
EOF

# Train with it
srengine train run \
  --config my_train.yaml \
  --dataset my_set \
  --model swinir \
  --instance ablation_1

# See what config was resolved
srengine train run \
  --config my_train.yaml \
  --dataset my_set \
  --model swinir \
  --dump-config
```

### Using a Project

```bash
# Create project
srengine project create experiment_1

# Train with project — all outputs go to project directory
srengine train run \
  --instance experiment_1 \
  --dataset my_set \
  --model swinir

# List checkpoints
ls models/experiment_1/checkpoints/

# View metrics
cat models/experiment_1/metrics/*.jsonl
```

### Machine Mode (JSONL metrics)

```bash
srengine train run \
  --instance experiment_1 \
  --dataset my_set \
  --model swinir \
  --machine \
  --metrics-frequency 10
```

This writes metrics to `<project>/metrics/<experiment_id>.jsonl` in JSONL format for programmatic consumption.

## Advanced Examples

### Resuming Training

```bash
srengine train run \
  --dataset my_set \
  --model swinir \
  --resume models/experiment_1/checkpoints/epoch_050.pt \
  --max-epochs 200
```

The trainer loads the checkpoint's model weights, optimizer state, and starting epoch. Training continues from epoch 51 to 200.

### Tiled Inference for Large Images

```bash
# 4K input → 16K output with 8GB GPU
srengine infer run \
  --model model.pth \
  --input 4k_image.png \
  --output 16k_image.png \
  --tile 512 \
  --overlap 128

# Large video with tiling
srengine infer run \
  --model model.pth \
  --input video_1080p.mp4 \
  --output video_4k.mp4 \
  --tile 384 \
  --overlap 64
```

### Model Export for Deployment

```bash
# ONNX for ONNX Runtime
srengine model export \
  --model-name swinir \
  --ckpt model.pth \
  --format onnx \
  --out model.onnx

# SafeTensors for HuggingFace
srengine model export \
  --model-name rrdb_esrgan \
  --ckpt model.pth \
  --format safetensors \
  --out model.safetensors

# TorchScript for C++ inference
srengine model export \
  --model-name swinir \
  --ckpt model.pth \
  --format torchscript \
  --out model.pt
```

### Model Instance Workflow

```bash
# Create a named model instance
srengine model create-instance \
  --instance experiment_1 \
  --name my_swinir \
  --model swinir

# List instances
srengine model list-instances --instance experiment_1

# Train with the instance — checkpoints go to instance directory
srengine train run \
  --instance experiment_1 \
  --instance my_swinir \
  --dataset my_set \
  --model swinir

# List training runs
srengine model list-runs --instance experiment_1/my_swinir

# List checkpoints
srengine model list-instances --instance experiment_1  # shows paths
ls models/my_swinir/checkpoints/
```

## Production Examples

### Full Production Workflow

```bash
# 1. Initialize workspace
cd /data
srengine workspace init

# 2. Build high-quality dataset
srengine dataset build \
  --input source_footage.mp4 \
  --output ./datasets/production_v1 \
  --degradations blur,noise,jpeg \
  --resize-method lanczos

# 3. Validate dataset
srengine dataset validate --path ./datasets/production_v1
srengine dataset health --path ./datasets/production_v1

# 4. Create project
srengine project create production_model

# 5. Train
srengine train run \
  --instance production_model \
  --instance production_swinir \
  --dataset production_v1 \
  --model swinir \
  --batch-size 8 \
  --max-epochs 500 \
  --learning-rate 1e-4 \
  --validation-split 0.1 \
  --machine \
  --metrics-frequency 10

# 6. Monitor training (in another terminal)
tail -f models/production_model/metrics/*.jsonl

# 7. Select best checkpoint
ls -lt models/production_swinir/checkpoints/

# 8. Export for deployment
srengine model export \
  --model-name swinir \
  --ckpt models/production_swinir/checkpoints/epoch_450.pt \
  --format onnx \
  --out production_swinir.onnx

# 9. Run inference on batch
for img in inputs/*.png; do
  srengine infer run \
    --model production_swinir.onnx \
    --input "$img" \
    --output "outputs/$(basename $img)" \
    --tile 512
done
```

### GUI Server Setup

```bash
# Terminal 1: Start server
srengine serve start --port 8765

# Terminal 2: Godot client connects
# (see docs/gui_bridge.md for C# implementation)

# Terminal 3: Test with netcat
echo '{"id":"1","command":"hello"}' | nc 127.0.0.1 8765
# → {"id":"1","type":"result","data":{"schema_version":1,"server_version":"0.1.0"}}
```

---

# Implementation Guide

## Implementation Strategy

### Phase 1: Workspace Setup
1. Initialize workspace with `srengine workspace init`
2. Copy built-in configs to workspace for customization
3. Create projects for different experiments

### Phase 2: Dataset Building
1. Acquire high-resolution source video (preferably 4K+, high bitrate, minimal compression)
2. Configure degradation pipeline for target use case
3. Build dataset and validate thoroughly
4. Run health check and prune any problematic frames

### Phase 3: Model Selection and Training
1. Select architecture (RRDB for throughput, SwinIR for quality)
2. Configure hyperparameters (batch size from VRAM, patch size from image content)
3. Train with validation monitoring
4. Resume from best checkpoint if needed

### Phase 4: Evaluation and Export
1. Run inference on test images/videos
2. Compare PSNR, SSIM, LPIPS against baselines
3. Export to deployment format
4. Deploy in target environment

## Design Choices

### Why ProcessPoolExecutor over ThreadPoolExecutor

Degradation is CPU-bound (image convolution, compression, resize). Python's GIL prevents true parallelism with threads. `ProcessPoolExecutor` spawns separate processes, each with its own GIL and memory space, achieving true multi-core parallelism.

**Cost:** Each process has high memory overhead (~50-100 MB for Python + OpenCV + NumPy). With 8 processes, that's ~400-800 MB overhead. Acceptable for dataset building where per-frame processing dominates.

### Why OpenCV over PIL/Pillow

- OpenCV provides consistent, well-optimized implementations of the entire degradation pipeline (resize, blur, morphology, color conversion, JPEG encode/decode)
- OpenCV's `cv2.imencode`/`cv2.imdecode` enables in-memory compression without temporary files
- Same library handles video I/O, image I/O, and all processing — fewer dependencies

### Why Decorator-Based Registry over Explicit Registry

```python
# Decorator (sr-engine approach):
@register("swinir")
class SwinIR(nn.Module): ...

# Explicit registry (alternative):
_registry["swinir"] = SwinIR
```

The decorator approach:
- Automatically registers at import time — no manual registry updates
- Model stays co-located with its registration — easier to see available names
- Adding a new model requires only: create file + add import in `__init__.py`
- No central registry file to edit

### Why 4-Level Config over Single Config File

The 4-level merge system enables:
1. **Sane defaults** that work out of the box
2. **Team-wide overrides** via workspace configs (shared in git)
3. **Experiment-specific configs** via `--config` files
4. **Quick iteration** via CLI flags — no file editing needed for one-off changes

This eliminates the common friction point of "I need to change one parameter but I have to find and edit a YAML file."

## Common Patterns

### Adding Command-Line Override for a New Config Key

```python
# In cmd_train.py or cmd_dataset.py:
@click.option("--my-new-param", type=float, help="Description")
def run(..., my_new_param):
    # The helper applies CLI overrides via merge_overrides
    config = helpers.parse_config_overrides(config, {"my_new_param": my_new_param})
```

### Adding a New Degradation Stage

1. Add the function in `data/degrade.py`:
```python
def _apply_my_custom_degradation(image, **params):
    # Process image
    return processed_image
```

2. Add it to `_degrade_image()`:
```python
if degrade_config.get("my_custom", {}).get("enabled", False):
    prob = degrade_config["my_custom"].get("prob", 1.0)
    if random.random() < prob:
        img = _apply_my_custom_degradation(img, **degrade_config["my_custom"])
```

3. Add config to `utils/configs/datasets/video_pairs.yaml`

### Testing a New Model Architecture

```bash
# Quick overfit test: train on a single patch for 10 epochs
srengine train run \
  --dataset tiny_set \
  --model my_new_model \
  --max-epochs 10 \
  --batch-size 4 \
  --patch-size 64 \
  --validation-enabled false
```

If training loss decreases and PSNR increases, the architecture learns. If not, debug the model forward pass, gradient flow, and loss function.

## Anti-Patterns

| Anti-Pattern | Why It's Wrong | Correct Approach |
|-------------|----------------|------------------|
| Editing built-in configs in `utils/configs/` | Changes are lost on package reinstall | Copy to workspace configs and edit there |
| Mixing LR and HR lists independently | Frame failures cause silent misalignment | Always use pairs from `batch_degrade()` return |
| Using `zip(hr_files, lr_files)` | Works only if both lists are sorted identically and have no failures | Use the pairs list from dataset builder |
| Training without validation split | No signal for overfitting or convergence | Use `--validation-split 0.1` |
| Running inference without tiling on large images | OOM on VRAM-limited GPUs | Use `--tile 512` |
| Manually modifying `.pt` files | Corrupts checkpoint format | Use the provided export functions |
| Running GUI bridge without a workspace | Server refuses to start | Initialize workspace first |

---

# Security Analysis

## Threat Model

```
┌────────────────────────────────────────────────┐
│             Trust Boundary                       │
│                                                  │
│  User (CLI) ───► sr-engine ──► Filesystem        │
│                      │                           │
│  GUI Client ───► TCP Server ──► Subprocess       │
│                                                  │
│  Attack surfaces:                                │
│  1. Malicious YAML config files                  │
│  2. Malicious checkpoint files (.pt)             │
│  3. GUI bridge TCP connection (localhost only)   │
│  4. Video/image files from untrusted sources    │
│  5. Environment variable injection               │
└────────────────────────────────────────────────┘
```

## Attack Surface

| Surface | Vector | Risk Level | Mitigation |
|---------|--------|------------|------------|
| YAML config loading | Arbitrary YAML from `--config` | Low | PyYAML `safe_load` — no arbitrary code execution |
| Checkpoint loading | Pickle-based `.pt` files | **High** | `weights_only=True` by default; fallback warning |
| GUI bridge TCP | Network access to port 8765 | Low | Binds to `127.0.0.1` only; no TLS but local-only |
| Video/image files | Malformed media triggering buffer overflows | Low | OpenCV handles parsing; CVEs are rare and patched |
| Subprocess arguments | CLI injection via job params | Low | `subprocess.Popen` with list args (no shell=True) |
| Environment variables | `SRENGINE_GUI_SOCKET` content | Low | JSON parsed, validated before use |

## Authentication

- **CLI**: No authentication — assumed to run in a trusted user session
- **GUI bridge**: Token-based authentication for subprocess control socket handshake. The token is a 32-byte random hex string generated per job, passed via environment variable, verified by the server on control socket connection.
- **No multi-user support**: The system assumes a single user operating in a trusted environment

## Authorization

- **Filesystem-based**: Relies on OS file permissions. No internal authorization layer.
- **GUI bridge**: Any connected GUI client can issue any command. No per-client authorization. The server binds to localhost only, limiting network exposure.

## Encryption

- **At rest**: No built-in encryption for checkpoint files, configs, or datasets. Use filesystem-level encryption (LUKS, eCryptfs) if needed.
- **In transit**: No TLS for GUI bridge. The server binds to `127.0.0.1` by default, so traffic never leaves the host. For remote access, tunnel through SSH or a VPN.

## Secrets Management

| Secret | Location | Risk | Recommendation |
|--------|----------|------|---------------|
| GUI bridge tokens | Environment variable (`SRENGINE_GUI_SOCKET`) | Low (ephemeral per job) | No action needed — tokens are random, per-job, short-lived |
| Model weights | `.pt` files on disk | Low (no inherent secrets) | None |
| Dataset paths | Config YAML files | Low (no sensitive data) | None |

## Vulnerabilities

### Known: Pickle Deserialization in Checkpoint Loading

**Risk:** `torch.save` uses Python's `pickle` protocol, which can execute arbitrary code during deserialization.

**Mitigation in sr-engine:**
```python
try:
    checkpoint = torch.load(path, map_location=..., weights_only=True)
except Exception:
    warnings.warn("Loading with weights_only=False. Unsafe for untrusted sources.")
    checkpoint = torch.load(path, map_location=..., weights_only=False)
```

`weights_only=True` (default) restricts deserialization to tensors and basic Python types, preventing arbitrary code execution.

**Recommendation:** Only load checkpoints from trusted sources. Use SafeTensors format for distribution (no pickle vulnerability).

### Potential: YAML Deserialization

Not a vulnerability in sr-engine because PyYAML's `safe_load` is used (not `load`). If switching to a different YAML library, ensure it uses safe loading.

## Security Best Practices

1. **Never load checkpoints from untrusted sources** without verifying with SafeTensors format first
2. **Run GUI bridge on localhost only** (default and recommended)
3. **Use OS file permissions** to restrict access to workspace directories
4. **Use SSH tunnels** if remote access to GUI bridge is needed
5. **Regularly update OpenCV and PyTorch** for security patches
6. **Validate input video files** before dataset building (corrupted media can cause crashes)
7. **Do not run as root** — the system does not need elevated privileges

---

# Performance Analysis

## Bottlenecks

### Dataset Building (CPU-bound)

| Stage | Bottleneck | Typical Time (per 1080p frame) |
|-------|------------|-------------------------------|
| Frame extraction | Video codec decoding (OpenCV) | 5-50 ms |
| Color jitter | Pixel manipulation | 1-3 ms |
| Gaussian blur | 2D convolution (OpenCV) | 3-15 ms |
| Motion blur | `filter2D` with kernel | 5-30 ms |
| Downsample | `cv2.resize` | 2-10 ms |
| Gaussian noise | Random sampling (NumPy) | 1-5 ms |
| Poisson noise | Random sampling (NumPy) | 2-8 ms |
| JPEG encode/decode | DCT + entropy coding | 5-20 ms |
| JPEG2000 encode/decode | Wavelet transform | 20-100 ms |
| PNG write + read | File I/O | 2-5 ms |

**Total per frame (all stages enabled):** ~50-200 ms. With 8 parallel processes: ~25 frames/second.

**Dominant cost:** JPEG2000 (when enabled) and motion blur with large kernels.

### Training (GPU-bound)

| Operation | Bottleneck | Typical Time (A100, 128×128 patch) |
|-----------|------------|-----------------------------------|
| DataLoader I/O | Disk read + decode | < 5 ms (prefetch) |
| Forward (RRDB) | Convolutions | 2-5 ms |
| Forward (SwinIR) | Window attention | 5-15 ms |
| Loss computation | VGG19 forward (perceptual) | 3-8 ms |
| Backward | Gradient computation | 2× forward time |
| Optimizer step | Momentum update | < 1 ms |
| Validation (PSNR/SSIM) | Whole-dataset forward | 10-60s per epoch |

**Training throughput:**
- RRDB: ~50-200 batches/s on A100 (batch_size=16, 128×128)
- SwinIR: ~20-80 batches/s on A100 (batch_size=16, 128×128)

SwinIR is typically 2-4× slower than RRDB due to window attention overhead.

### Inference (GPU-bound)

| Input Size | Model | No Tiling (ms) | Tiled 512px (ms) |
|------------|-------|----------------|-------------------|
| 512×512 | RRDB | 10-30 | 10-30 |
| 1920×1080 | RRDB | 100-300 | 150-450 |
| 3840×2160 | RRDB | OOM (8GB) | 600-1800 |
| 512×512 | SwinIR | 30-100 | 30-100 |
| 1920×1080 | SwinIR | 300-1000 | 400-1200 |

## Scaling Factors

| Factor | Effect | Recommendation |
|--------|--------|----------------|
| **Batch size** | Linear memory scaling, sub-linear throughput scaling (GPU utilization saturates) | Max: VRAM / (model_memory + patch_memory × batch) |
| **Patch size** | Quadratic memory scaling (H×W) | Max: VRAM / (model_memory × patch_area × batch) |
| **Number of workers** | Linear dataset I/O throughput | `num_workers=4-8` (saturates at ~8 for most systems) |
| **Process pool workers** | Linear degradation throughput (CPU-bound) | `min(cpu_count, len(frames))` |
| **Scale factor** | Quadratic output memory (H×W×scale²) | Larger scale = larger output = more VRAM for inference |
| **Model depth** | Linear parameter count, memory, and time | 23 RRDB blocks, 6×6 RSTB stages are default — adjust based on quality needs |

## Optimization Opportunities

### Dataset Building

1. **Disable JPEG2000** unless specifically needed — it's 4-10× slower than JPEG
2. **Reduce motion blur kernel size** — large kernels (31+) are expensive
3. **Increase `frame_rate` skip** — fewer frames = faster build, but less training data
4. **Use NVMe SSDs** — PNG write throughput becomes I/O-bound on HDDs at high parallelism
5. **Pre-extract frames** with external tool (ffmpeg) for large videos — OpenCV's decoder can be slow

### Training

1. **Use bf16** on compatible GPUs — 2× throughput vs fp32, no loss scaling needed
2. **Increase `metrics_frequency`** — fewer callback calls = less overhead
3. **Use `num_workers=4-8`** with `prefetch_factor=2` for DataLoader throughput
4. **Disable perceptual loss** during initial training — add it for fine-tuning
5. **Use gradient accumulation** for effective large batch size without OOM
   ```python
   # Config: batch_size=4, accumulation_steps=4 → effective batch_size=16
   loss = loss / accumulation_steps
   loss.backward()
   if (batch + 1) % accumulation_steps == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```

### Inference

1. **Disable tiling** on high-VRAM GPUs — tiling adds blending overhead
2. **Use FP16 inference** — `model.half()` + `input.half()` for 2× throughput
3. **Batch inference** on multiple images (model is fully convolutional, batch dimension is flexible)
4. **Use ONNX Runtime** for deployment — often faster than PyTorch eager mode
5. **Use TensorRT** (via ONNX) for maximum inference throughput

### General

1. **Profile first, optimize second** — measure with `srengine env bench` before tuning
2. **Use `--dump-config`** to verify config resolution before long runs
3. **Monitor GPU utilization** with `nvidia-smi` or `rocm-smi` — low utilization indicates a bottleneck elsewhere (I/O or CPU)

## Resource Usage Estimates

### Memory (Training)

| Component | Memory per Element |
|-----------|-------------------|
| Model parameters (RRDB, 23 blocks) | ~16 MB (fp32) / ~8 MB (fp16) |
| Model parameters (SwinIR, embed_dim=180) | ~12 MB (fp32) / ~6 MB (fp16) |
| Input tensor (128×128, batch=16) | ~3 MB (fp32) |
| Output tensor (512×512, batch=16) | ~48 MB (fp32) |
| Gradient memory | ~2× parameter memory |
| Optimizer state (Adam) | ~2× parameter memory |
| VGG19 backbone (perceptual loss) | ~20 MB (frozen) |
| Activation memory (SwinIR) | Higher — attention stores intermediate values |

**Peak memory estimate (batch=16, patch=128, SwinIR):** ~4-8 GB
**Peak memory estimate (batch=16, patch=128, RRDB):** ~2-4 GB

### Disk (Dataset)

| Source | HR Storage | LR Storage (4×) |
|--------|-----------|-----------------|
| 10 min 1080p @ 30fps (18,000 frames) | ~50 GB (PNG) | ~3 GB (PNG) |
| 1 min 4K @ 24fps (1,440 frames) | ~20 GB (PNG) | ~1.25 GB (PNG) |
| Per frame cost | ~2.8 MB (1920×1080) | ~0.18 MB (480×270) |

---

# Reliability and Resilience

## Fault Tolerance

| Component | Failure Mode | Impact | Recovery |
|-----------|-------------|--------|----------|
| Dataset building | Worker process crash | Loss of one frame's LR | Pool continues; frame omitted from pairs |
| Dataset building | Video read error | Zero frames extracted | Error raised; no partial dataset created |
| Training | GPU out of memory | `torch.cuda.OutOfMemoryError` | Training aborts; last checkpoint is intact |
| Training | NaN loss | Training diverges | No automatic recovery — inspect LR, model, data |
| Training | SIGTERM (cancel) | Training stops mid-epoch | Last checkpoint is intact; resume from it |
| Checkpoint save | Crash mid-write | `.tmp` file left on disk | `.pt` file is untouched (atomic write) |
| GUI bridge | Subprocess crash | Job fails | Exit code recorded in manifest; `done` event broadcast |
| GUI bridge | Client disconnect | Client stops receiving events | No impact on server or other clients |
| Config loading | Missing key | Validation error at startup | Descriptive error message shows expected keys |

## Recovery Strategies

### Training Resume

```bash
# If training crashes or is cancelled:
srengine train run \
  --dataset my_set \
  --model swinir \
  --resume checkpoints/epoch_050.pt \
  --max-epochs 100
```

The trainer:
1. Loads model weights from checkpoint
2. Loads optimizer state (Adam momentum, etc.)
3. Sets `current_epoch` to the saved step
4. Resumes the LR scheduler from the correct position

### Dataset Rebuild

If dataset build fails mid-way:
```bash
# Clean up partial output
rm -rf ./datasets/my_partial_set

# Fix the issue (e.g., disk space, video codec)
# Then rebuild
srengine dataset build --input video.mp4
```

The pipeline is designed to fail early: if validation fails, the manifest is deleted, preventing training on a partial/invalid dataset.

### Workspace Recovery

```bash
# Re-check workspace health
srengine workspace check

# Re-initialize if marker is missing (preserves existing data)
srengine workspace init
```

## Observability

| Signal | Source | What It Reveals |
|--------|--------|-----------------|
| Training loss | MetricsStream JSONL | Convergence, overfitting, instability |
| Validation PSNR/SSIM | MetricsStream JSONL | Model quality, overfitting |
| Learning rate | MetricsStream JSONL | Scheduler behavior |
| GPU utilization | `nvidia-smi` / `rocm-smi` | Training efficiency |
| VRAM usage | `nvidia-smi` / `srengine env check` | Memory bottlenecks |
| System logs | `dmesg` | GPU errors, OOM kills |
| Job manifests | `<ws>/jobs/*.json` | Job completion status, exit codes |

## Monitoring

### Key Metrics to Monitor

1. **Training loss** — should decrease monotonically. Spikes indicate instability.
2. **Validation PSNR** — should increase. Plateau = convergence; decrease = overfitting.
3. **GPU memory** — if near capacity, training may OOM on larger batches.
4. **Dataset health** — run `srengine dataset health` periodically to catch corrupt frames.
5. **Disk space** — datasets can consume 10s of GB. Monitor with `df -h`.

## Alerting

Recommended alert thresholds:
- **PSNR drop > 2dB** on validation → model may have diverged
- **Loss = NaN** → training must be stopped and LR reduced
- **GPU memory > 90%** → reduce batch size
- **Disk space < 10%** → clean up old datasets/checkpoints
- **Job failure rate > 10%** → investigate system issues

---

# Scalability

## Vertical Scaling

| Resource | Scaling Strategy | Expected Gain |
|----------|-----------------|---------------|
| **CPU cores** | More ProcessPoolExecutor workers for dataset building | Linear up to ~16 cores (I/O bound beyond) |
| **GPU memory** | Larger batch sizes, larger patches, no tiling needed | Batch size scales linearly with VRAM |
| **GPU compute** | Higher-end GPU (A100 > RTX 4090 > RTX 3090) | 2-4× through computing generations |
| **Disk I/O** | NVMe SSD for dataset storage | 5-10× vs HDD for frame loading |
| **RAM** | Larger dataset caching, more DataLoader workers | Diminishing returns beyond 32 GB |

## Horizontal Scaling

sr-engine is designed for **single-node** operation. Horizontal scaling approaches:

### Multiple Dataset Build Workers

For very large video corpora, split videos across machines and build datasets independently:

```bash
# Machine 1
srengine dataset build --input videos_1.mp4 --output datasets/shared/dataset_1

# Machine 2
srengine dataset build --input videos_2.mp4 --output datasets/shared/dataset_2

# Merge manifests (manual or script)
```

### Distributed Training

sr-engine does not natively support distributed training (DDP/FSDP). For multi-GPU training:
- Use a single GPU per training run (most common for SR research)
- Use PyTorch DDP wrapper for multi-GPU (requires code modification)
- Leverage gradient checkpointing for memory efficiency on a single GPU

### Multiple GUI Bridge Clients

The GUI bridge server supports multiple simultaneous GUI clients. All clients receive all events (broadcast model). The server does not have per-client state beyond the TCP connection.

## Load Balancing

The GUI bridge uses a single-threaded async socket model. For very high event rates (>1000 events/second), the server may become a bottleneck. Mitigations:
- Reduce `metrics_frequency` (log every 10 batches instead of every batch)
- Batch progress updates in `SocketReporter`
- Use multiple server instances on different ports for different workspaces

## Caching

- **Dataset caching**: `PairedImageFolderDataset` loads images on-the-fly. No RAM caching. For faster training, consider `torchdata` or pre-loading datasets into RAM for small datasets.
- **Config caching**: Configs are loaded once at startup and reused. No runtime config reloading.
- **Model caching**: Models are loaded once per training or inference session. For inference servers, keep the model in GPU memory between requests.

---

# Operational Guide

## Deployment

### Production Setup

```bash
# 1. Install system dependencies
sudo apt-get install python3.11 python3.11-venv ffmpeg libsm6 libxext6

# 2. Clone and build
git clone https://github.com/your-org/sr-engine.git
cd sr-engine
./envs/build.sh --backend cuda

# 3. Initialize workspace
srengine workspace init --path /data/sr_workspace

# 4. Create production project
srengine project create production

# 5. Set up monitoring (cron job)
(crontab -l 2>/dev/null; echo "0 * * * * /usr/bin/df -h /data >> /var/log/sr_disk.log") | crontab -
```

### GUI Bridge as System Service

```systemd
# /etc/systemd/system/sr-engine.service
[Unit]
Description=sr-engine GUI Bridge Server
After=network.target

[Service]
Type=simple
User=sruser
WorkingDirectory=/data/sr_workspace
Environment=SRENGINE_WORKSPACE=/data/sr_workspace
ExecStart=/home/sruser/sr-engine/.venv/bin/srengine serve start --port 8765
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable sr-engine
sudo systemctl start sr-engine
```

## Maintenance

### Regular Tasks

| Frequency | Task | Command |
|-----------|------|---------|
| Daily | Check disk usage | `df -h /data` |
| Weekly | Prune old checkpoints | `find . -name "*.pt" -mtime +90 -delete` |
| Weekly | Archive old datasets | `tar czf archive.tar.gz datasets/old_dataset` |
| Monthly | Verify workspace health | `srengine workspace check` |
| Monthly | Update dependencies | `uv sync --upgrade` |

### Dataset Lifecycle Management

```bash
# List all datasets
srengine workspace info

# Archive old dataset
tar czf archived_dataset.tar.gz datasets/old_data

# Remove archived dataset
rm -rf datasets/old_data

# But keep manifest for reference
cp datasets/old_data/manifest.json manifests/
```

## Upgrades

```bash
# Update sr-engine
git pull origin main
uv sync

# Check for breaking changes
git log --oneline HEAD..origin/main

# Verify installation
srengine env check
srengine workspace check
```

## Rollback Procedures

### Code Rollback

```bash
# Roll back to previous version
git checkout v0.0.9
uv sync

# Verify
srengine env check
```

### Checkpoint Rollback

```bash
# List checkpoints in descending mtime order
ls -lt models/production/checkpoints/

# Revert to an earlier checkpoint for inference
srengine infer run \
  --model models/production/checkpoints/epoch_100.pt \
  --input test.png \
  --output output.png
```

## Monitoring

### Health Check Endpoints

For integration with monitoring systems (Prometheus, Datadog, etc.):

```bash
# Basic health
srengine workspace check

# Hardware diagnostics
srengine env check

# Dataset health (for each active dataset)
srengine dataset health --path datasets/my_set
```

### Logging

Log output format: `[module_name] LEVEL: message`

```python
# From utils/logging.py
log = get_logger(__name__)
log.info("Training started")
log.warning("Low disk space")
```

All logs go to stdout. Redirect to a file in production:

```bash
srengine train run ... 2>&1 | tee -a /var/log/sr_training.log
```

## Incident Response

### Training Divergence (Loss → NaN)

1. **Detect**: Monitor `metrics.jsonl` for `loss_total: NaN`
2. **Stop**: Send SIGTERM to training process (or `job.cancel` via GUI)
3. **Diagnose**: Check last few valid loss values — was there a sudden spike?
4. **Fix**: Reduce learning rate by 10×, check for corrupted data in recent batches
5. **Resume**: From the last valid checkpoint with reduced LR

### OOM (Out of Memory)

1. **Detect**: Training or inference crashes with `torch.cuda.OutOfMemoryError`
2. **Diagnose**: Check `srengine env check` for VRAM usage. If near capacity, reduce batch size.
3. **Fix**:
   - Reduce `--batch-size` (most effective)
   - Reduce `--patch-size` (quadratic reduction)
   - Enable tiling for inference (`--tile 512`)
4. **Resume**: From last checkpoint with reduced batch size

### Disk Full

1. **Detect**: `OSError: No space left on device`
2. **Diagnose**: `df -h` to identify full partition
3. **Fix**:
   - Remove old checkpoints: `find . -name "*.pt" -mtime +30 -delete`
   - Archive old datasets: `tar czf archive.tar.gz datasets/old && rm -rf datasets/old`
   - Clear job manifests: `rm jobs/*.json`
4. **Resume**: Resume training once space is available

---

# Troubleshooting

## Common Issues and Resolutions

| Symptom | Cause | Diagnosis | Resolution |
|---------|-------|-----------|------------|
| `Error: Workspace not found` | No `.sr_workspace` marker in path | `ls .sr_workspace` returns nothing | `srengine workspace init` or set `SRENGINE_WORKSPACE` env var |
| `Error: Dataset not found` | Path resolution failed | Check absolute, CWD-relative, and workspace-relative paths | Specify full path or move dataset into workspace datasets/ |
| `Error: Could not open video file` | Unsupported codec, corrupt file, or missing library | `ffprobe video.mp4` to check codec | Install codec support or re-encode with `ffmpeg -i input.mp4 -c:v libx264 output.mp4` |
| `CUDA out of memory` | Batch size + patch size too large | `nvidia-smi` shows VRAM near 100% | Reduce `--batch-size`, reduce `--patch-size`, or enable gradient accumulation |
| `loss_total: NaN` in metrics | Training diverged | Check LR, data values, model stability | Reduce `--learning-rate`, check for NaN in dataset, validate model forward pass |
| Validation PSNR decreases | Overfitting | Train loss decreases but val loss increases | Reduce validation split, add data augmentation, reduce model capacity |
| Validation PSNR always zero | Scale factor mismatch in validation | Check `--patch-size` vs dataset scale | Set `patch_size` to match dataset scale factor (e.g., 128 LR → 512 HR at 4×) |
| `FileNotFoundError: No such config file` | Config path is wrong | `ls <path>` | Use absolute path or relative to CWD |
| `yaml.YAMLError` in config | Malformed YAML | `python -c "import yaml; yaml.safe_load(open('config.yaml'))"` | Fix YAML syntax (check indentation, quotes) |
| `RuntimeError: Dataset validation failed` | Dimension mismatch or corrupt files | Run `srengine dataset validate --path <dataset>` for details | Rebuild dataset with correct scale factor |
| Checkpoint load fails | Architecture mismatch | Compare checkpoint config with current model config | Train with the same model config as the checkpoint |
| `env check` shows CPU when GPU exists | Wrong PyTorch build | `python -c "import torch; print(torch.cuda.is_available())"` | Reinstall PyTorch with CUDA/ROCm: `./envs/build.sh --backend cuda` |
| GUI Bridge connection refused | Server not running | `srengine serve start --port 8765` | Start the server in another terminal or as a systemd service |
| Subprocess never connects control socket | Subprocess crashed before handshake | Check `<ws>/jobs/<job_id>.json` for exit code | Inspect subprocess stdout/stderr for errors |
| Events arrive on one client but not another | Client connected mid-stream | Events are broadcast to currently-connected clients only | Missed events are not replayed; implement client-side reconnection |
| Very slow dataset build | Large video, all stages enabled | Time individual stages | Disable JPEG2000, reduce motion blur kernel size, increase frame skip |
| Black frame false positives | Very dark but valid content | Check `computed_threshold` in health report | Tune threshold manually or exclude night footage from training set |

## Step-by-Step Debugging Procedures

### Procedure 1: Training Troubleshooting

```
Step 1: Check basic health
├── srengine env check          ← GPU, VRAM, PyTorch version
├── srengine workspace check    ← Workspace structure
└── srengine dataset validate --path <dataset>  ← Dataset integrity

Step 2: Verify config
├── srengine train run --dataset <d> --model <m> --dump-config
└── Check: batch_size, learning_rate, max_epochs, patch_size, scale

Step 3: Run quick test
├── srengine train run --dataset <d> --model <m> --max-epochs 5 --batch-size 2
├── Check: loss decreases? → proceed
├── Check: loss = NaN? → go to Step 5
└── Check: OOM? → reduce batch_size or patch_size

Step 4: Monitor training
├── tail -f models/<p>/metrics/*.jsonl
├── Check: loss decreases monotonically?
├── Check: validation PSNR increases?
└── Check: LR follows cosine schedule?

Step 5: Debug NaN loss
├── Reduce learning_rate by 10×
├── Disable perceptual loss (perceptual_weight: 0)
├── Check dataset for corrupt images (run dataset health)
├── Try fp32 instead of bf16/fp16
└── Validate model forward pass: model(lr).sum().backward() on a single batch
```

### Procedure 2: Inference Troubleshooting

```
Step 1: Verify checkpoint
├── srengine model info --model model.pth
├── Check: Config matches expected architecture?
└── Check: Step count looks reasonable?

Step 2: Run inference on small image
├── srengine infer run --model model.pth --input small.png --output out.png --tile 0
├── Check: Output exists? Dimensions correct (scale × input)?
├── Check: Visual quality acceptable?
└── OOM? → proceed to Step 3

Step 3: Debug tiling
├── srengine infer run --input large.png --output out.png --tile 512 --overlap 64
├── Check: Seam artifacts visible? → increase overlap
├── Check: Still OOM? → reduce tile size
└── Check: Slow? → reduce overlap, fewer tiles

Step 4: Video inference
├── Check: Source FPS and resolution with ffprobe
├── Check: Output video playable?
├── Check: Frame-by-frame consistency (no flickering)?
└── Slow? → use fewer frames (reduce video length)
```

### Procedure 3: Dataset Building Troubleshooting

```
Step 1: Verify source video
├── ffprobe video.mp4
├── Check: Codec, resolution, duration, frame rate
├── Check: No corruption (play video)
└── Missing codec? → re-encode: ffmpeg -i input -c:v libx264 output.mp4

Step 2: Quick build test
├── srengine dataset build --input video.mp4 --output test_ds --degradations jpeg
├── Check: Frames extracted? (ls test_ds/HR/)
├── Check: LR frames created? (ls test_ds/LR/)
└── Failure? → check error message

Step 3: Full build
├── srengine dataset build --input video.mp4
├── Progress bar timing → estimate total time
├── Check final validation message
└── Validation failed? → srengine dataset validate --path <dataset>

Step 4: Health check
├── srengine dataset health --path <dataset>
├── Check: Resolution distribution consistent?
├── Check: Black frames detected?
└── Prune if needed: srengine dataset health --path <dataset> --yes
```

### Procedure 4: GUI Bridge Troubleshooting

```
Step 1: Start server
├── srengine serve start --port 8765
├── Check: "Server started on port 8765"
└── Error: "Workspace not found"? → Initialize workspace

Step 2: Test connection
├── echo '{"id":"1","command":"hello"}' | nc 127.0.0.1 8765
├── Expected: {"id":"1","type":"result","data":{"schema_version":1,...}}
├── No response? → Wrong port or firewall
└── Malformed response? → Check nc version or use Python socket test

Step 3: Test async command
├── echo '{"id":"2","command":"train.start","params":{"model_name":"rrdb_esrgan","dataset":"...","max_epochs":2}}' | nc 127.0.0.1 8765
├── Expected: {"id":"2","type":"accepted","data":{"job_id":"train_...","status":"accepted"}}
├── No accepted? → Check params match CLI requirements
└── Job fails? → Check <ws>/jobs/<job_id>.json for exit code

Step 4: Monitor job
├── curl http://127.0.0.1:8765 (not HTTP — TCP only)
├── Use Python: socket.connect(host, port); socket.send(json + "\n"); socket.recv(65536)
└── Implement client reconnection: exponential backoff, 5 attempts
```

## Error Reference

| Error Message | Module | Likely Cause | Fix |
|--------------|--------|-------------|-----|
| `FileNotFoundError: Could not open video file` | `video_extract.py` | Missing file, wrong path, or unsupported codec | Check path, re-encode video |
| `ValueError: No frames were extracted` | `dataset_builder.py` | Video unreadable or `start_time` > duration | Check video, adjust `start_time` |
| `RuntimeError: Dataset validation failed` | `dataset_validator.py` | Dimension mismatch, missing files | Run validate for details, rebuild with correct scale |
| `ValueError: Model 'X' not found in registry` | `registry.py` | Typo in model name or model not imported | Check `--model` name, verify model is registered |
| `CUDA out of memory. Tried to allocate X MiB` | PyTorch | Batch/patch too large for VRAM | Reduce batch_size or patch_size |
| `KeyError: 'Missing required config key: scale'` | `config.py` | Config file missing required key | Add key or use built-in defaults |
| `FileNotFoundError: Checkpoint file not found` | `checkpoint.py` | Wrong path to `.pt` file | Check path, use absolute path |
| `ConnectionRefusedError` | GUI bridge | Server not running | Start server with `srengine serve start` |
| `Exception: hello handshake rejected` | GUI bridge | Wrong token or job_id | Check `SRENGINE_GUI_SOCKET` environment variable |
| `ValueError: Unknown VGG19 layer name` | `losses.py` | Wrong layer name in perceptual_layers config | Use valid names: `relu1_1` through `relu5_4` |
| `ValueError: Unsupported gan_type` | `losses.py` | Wrong GAN type in config | Use `vanilla` or `lsgan` |
| `warnings.warn: Loading with weights_only=False` | `checkpoint.py` | Older PyTorch version | Upgrade PyTorch or accept the warning |

---

# Common Mistakes

## Beginner Mistakes

### Mistake 1: Using Absolute Paths Everywhere

**Problem:** Hard-coded paths break when moving projects between machines or directories.

**Solution:** Initialize a workspace and use workspace-relative paths:

```bash
# Bad
srengine train run --dataset /home/user/data/my_set --model swinir

# Good
srengine workspace init
srengine train run --dataset my_set --model swinir  # Auto-resolved to <ws>/datasets/my_set
```

### Mistake 2: Editing Builtin Configs

**Problem:** Changes to `utils/configs/*.yaml` are lost on `git pull` or package reinstall.

**Solution:** Copy configs to workspace and edit there:

```bash
mkdir -p <ws>/configs/train/
cp .venv/lib/python3.11/site-packages/sr_engine/utils/configs/train/base.yaml <ws>/configs/train/
# Now edit <ws>/configs/train/base.yaml — it takes precedence over built-in
```

### Mistake 3: Using Wrong Scale Factor

**Problem:** Setting `--scale 4` but dataset was built with `--scale 2`. LR dimensions don't match expectations.

**Solution:** Verify scale factor consistency:
- Dataset building: `--scale` in dataset config
- Training: `scale` in both dataset config and model config
- Inference: Model's built-in `scale` parameter must match training

```bash
# Check dataset scale
python -c "import json; d=json.load(open('datasets/my_set/manifest.json')); print(d['config']['scale'])"

# Verify with validation
srengine dataset validate --path datasets/my_set
```

### Mistake 4: Not Validating Datasets Before Training

**Problem:** Training fails mid-way due to corrupt or mismatched images.

**Solution:** Always validate after building:

```bash
srengine dataset build --input video.mp4  # Auto-validates
srengine dataset health --path datasets/my_set  # Optional but recommended
```

## Architectural Mistakes

### Mistake 5: Ignoring the Process Pool Memory Cost

**Problem:** Setting `--num-workers 32` on a 32-core machine for dataset building, causing system OOM.

**Root cause:** Each process loads Python + OpenCV + NumPy independently — ~100 MB per process. 32 processes = 3.2 GB overhead.

**Solution:** Use `min(cpu_count, 8-16)` workers. For large frame counts, process memory dominates total memory.

```bash
# Reasonable default: number of physical cores, capped at 16
srengine dataset build --input video.mp4  # Uses min(cpu_count, 16) by default
```

### Mistake 6: Training Without Validation Split

**Problem:** No signal for overfitting or convergence. Model may memorize training data.

**Solution:** Always use at least 5-10% validation split:

```bash
srengine train run \
  --dataset my_set \
  --model swinir \
  --validation-split 0.1 \
  --save-per-epoch 5
```

### Mistake 7: Assuming Perceptual Loss Always Helps

**Problem:** Adding perceptual loss (`--perceptual-weight 0.1`) to every training run slows training and can hurt PSNR.

**Root cause:** Perceptual loss optimizes for feature-space similarity, not pixel accuracy. It can reduce PSNR while improving perceptual quality.

**Solution:** Use perceptual loss only for fine-tuning or when visual quality matters more than pixel accuracy. Train with pixel loss first, then add perceptual loss.

### Mistake 8: Mixing Training and Inference Code Paths

**Problem:** Calling `model.eval()` at the wrong time (before inference instead of after training), causing batch-norm/ dropout layers to behave differently.

**Solution:** The `Trainer` handles `model.train()` / `model.eval()` transitions automatically. Only set manually if customizing the inference pipeline.

## Operational Mistakes

### Mistake 9: Running Out of Disk During Dataset Build

**Problem:** Dataset build fails after hours of processing when disk fills up.

**Solution:** Estimate disk needs before building:
```
HR size per frame: H × W × 3 bytes (PNG compressed: ~2-4 MB for 1080p)
LR size per frame: H/scale × W/scale × 3 bytes (PNG: ~0.2-0.5 MB for 480p at 4×)
Total estimate: frames × (HR_size + LR_size)
```

Check available space:
```bash
df -h /data
# At least 2× estimated dataset size for headroom
```

### Mistake 10: Not Setting `SRENGINE_WORKSPACE` in Cron/Systemd

**Problem:** Automated scripts fail because they can't auto-discover the workspace from CWD.

**Solution:** Always set the environment variable explicitly:

```systemd
[Service]
Environment=SRENGINE_WORKSPACE=/data/sr_workspace
```

### Mistake 11: Killing Training Process Hard (SIGKILL)

**Problem:** `kill -9` leaves checkpoint in inconsistent state (if `.tmp` file wasn't renamed).

**Solution:** Use SIGTERM (`kill <pid>` or `Ctrl+C`), which allows the `atexit` handler to complete. The atomic write pattern protects against partial `.pt` files, but SIGKILL gives zero chance for cleanup.

## Security Mistakes

### Mistake 12: Downloading and Loading Untrusted Checkpoints

**Problem:** Loading a `.pt` file from the internet (even with `weights_only=True`) is safer with `weights_only=True` but still risky — the config inside could trigger unexpected behavior.

**Solution:** Convert untrusted checkpoints to SafeTensors before loading:
```bash
srengine model export \
  --model-name swinir \
  --ckpt untrusted_model.pth \
  --format safetensors \
  --out safe_model.safetensors
```

### Mistake 13: Exposing GUI Bridge to Network

**Problem:** Starting `srengine serve start --host 0.0.0.0` allows any network client to control training jobs.

**Solution:** Always bind to localhost (default). Use SSH tunneling for remote access:
```bash
# On remote machine
srengine serve start --host 127.0.0.1 --port 8765

# On local machine
ssh -L 8765:127.0.0.1:8765 user@remote-machine
# Now connect to localhost:8765
```

## Performance Mistakes

### Mistake 14: Training with fp32 on bf16-Capable GPUs

**Problem:** Training is 2× slower than necessary on A100/RTX 4090.

**Solution:** sr-engine auto-selects bf16 on compatible GPUs. Verify with `srengine env check`. To force a specific mode:
```yaml
# In train config:
dtype: bf16  # or fp16 or fp32
```

### Mistake 15: Enabling All Degradations for Every Dataset

**Problem:** JPEG2000 adds 4-10× processing time and is rarely needed. Color jitter is often irrelevant.

**Solution:** Only enable degradations that match the target deployment scenario. Production pipeline example:
```yaml
degradation:
  blur:
    enabled: true           # Almost always needed
  noise:
    enabled: true           # Almost always needed
  jpeg:
    enabled: true           # Common in real-world
  jpeg2000:
    enabled: false          # Rarely needed
  color_jitter:
    enabled: false          # Only for camera-array datasets
```

---

# Best Practices

## Dataset Building

1. **Use diverse source video**: Different scenes, lighting conditions, camera types, and compression levels produce more robust models. A single video source leads to overfitting on that source's characteristics.

2. **Validate scale factor consistency**: The `scale` parameter must be consistent across dataset building, training config, and model config. Run `srengine dataset validate` after every build.

3. **Run health checks**: `srengine dataset health --path <dataset>` catches black frames, corrupt images, and resolution mismatches before training wastes GPU time.

4. **Estimate disk needs upfront**: For large datasets, calculate storage requirements before building. Cache manifests (keep `manifest.json` even if you delete old datasets).

5. **Use appropriate frame rate**: 10-24 fps is generally sufficient for training. Higher frame rates create many near-duplicate frames that waste disk space and processing time.

## Training

6. **Start with a quick overfit test**: Train on a small subset (100-500 frames) for 10-20 epochs. If training loss doesn't decrease, something is wrong with the model, data, or config.

7. **Use cosine LR schedule with warmup**: Reduces the risk of early divergence and improves convergence quality. Default sr-engine config includes this.

8. **Monitor both loss and validation metrics**: Decreasing loss with decreasing validation PSNR = overfitting. Increasing loss = divergence. Stable loss with increasing PSNR = good training.

9. **Save checkpoints frequently**: Set `--save-per-epoch 5` at minimum. A crash mid-training loses at most 5 epochs of work. The last checkpoint is always valid (atomic write).

10. **Resume from best checkpoint**: At the end of training, evaluate all checkpoints on a held-out test set and deploy the best one, not necessarily the last one.

## Inference

11. **Match inference device to training**: Models trained with mixed precision (bf16/fp16) should be inferred with the same dtype for consistent results.

12. **Use tiling for large inputs**: Always enable tiling (`--tile 512`) for images larger than approximately 2K on 8 GB GPUs. The blending makes seams invisible with `--overlap 64-128`.

13. **Validate output dimensions**: For an input of resolution (H, W) and scale factor S, output should be (H×S, W×S). If not, check scale factor consistency.

14. **Batch inference for throughput**: The model handles arbitrary batch sizes. Process multiple images simultaneously:
```python
# Pseudocode: batch inference
images = [load_image(p) for p in image_paths]
batch = torch.stack(images).to(device)
with torch.no_grad():
    sr_batch = model(batch)
```

## Model Selection

15. **Match model to use case**:
    - **RRDB**: Higher throughput, lower memory, good quality. Best for video SR, real-time applications, and production deployment.
    - **SwinIR**: Higher quality, lower throughput, higher memory. Best for image SR, maximum quality requirements, and research.

16. **Export to ONNX for deployment**: ONNX Runtime is faster than PyTorch eager mode for inference. Test both and compare.

17. **Use SafeTensors for distribution**: The safetensors format is safe from pickle exploits and compatible with HuggingFace ecosystem.

## Operational

18. **Use `--machine` mode for automation**: JSONL metrics are machine-parseable. Pipe them to monitoring systems:
```bash
srengine train run ... --machine
# Write metrics to models/<p>/metrics/<experiment_id>.jsonl
```

19. **Set `SRENGINE_WORKSPACE` in all automated contexts**: Crontab, systemd, Docker — everywhere. Saves debugging time vs. relying on CWD-based auto-discovery.

20. **Keep workspace separate from code**: `/data/sr_workspace/` or `/mnt/storage/sr_workspace/` — never inside the git repository. Datasets and checkpoints are large and should not be version-controlled.

---

# Comparative Analysis

## sr-engine vs. Alternatives

| Feature | sr-engine | BasicSR | Real-ESRGAN | SwinIR (official) |
|---------|-----------|---------|-------------|-------------------|
| **CLI** | Full (Click) | Minimal (Python API) | Script-based | Script-based |
| **Config system** | 4-level merge | YAML config files | YAML config files | Hard-coded params |
| **GUI bridge** | Built-in (TCP/NDJSON) | None | None | None |
| **CUDA** | Full | Full | Full | Full |
| **ROCm** | Full | Partial | None | None |
| **Model registry** | Decorator-based | Manual registration | Single model | Single model |
| **Tiled inference** | Built-in | Built-in | Built-in | Built-in |
| **Model export** | ONNX, SafeTensors, TorchScript | ONNX | ONNX | ONNX |
| **Degradation** | Configurable 6-stage | Fixed pipeline | High-order (2-stage) | Basic bicubic |
| **Dataset building** | Built-in CLI | External scripts | External scripts | Manual |
| **Progress reporting** | tqdm + NDJSON | tqdm | tqdm | None |
| **Loss functions** | L1 + Perceptual + GAN | L1 + Perceptual + GAN | L1 + Perceptual + GAN | L1 |
| **Trainer callbacks** | Yes (extensible) | Limited | No | No |
| **PyTorch version** | No hard dependency | Hard dependency | Hard dependency | Hard dependency |
| **Install** | `uv sync` | `pip install` | `pip install` | `pip install` |

### Strengths vs. Weaknesses

#### sr-engine

**Strengths:**
- Production-ready CLI with all functionality exposed
- GPU-agnostic (CUDA + ROCm) without config changes
- Built-in GUI bridge for Godot integration
- 4-level config merge (flexible and powerful)
- Extensible model registry
- Process-pool accelerated dataset building
- No disk-usage threshold issues (atomic writes, self-cleaning manifests)
- Standalone entry points (no `srengine` prefix needed)

**Weaknesses:**
- No distributed training (single-GPU only)
- No TensorRT export (ONNX → TensorRT manual)
- No high-order degradation (Real-ESRGAN style 2-stage)
- Single-node only (no cluster support)
- No built-in experiment tracking (MLflow, W&B integration)
- No REST API for inference (must use GUI bridge or CLI)

#### BasicSR

**Strengths:**
- Mature, widely used in research
- Supports many architectures (RRDB, SwinIR, HAT, etc.)
- Distributed training support
- Extensive YAML config system

**Weaknesses:**
- Python API only (no rich CLI)
- No GUI integration
- No ROCm support
- Complex dependency chain
- No built-in dataset building

#### Real-ESRGAN

**Strengths:**
- High-order degradation model (best real-world performance)
- Pre-trained models available
- Well-documented training pipeline
- Active development

**Weaknesses:**
- Single architecture (RRDB) — no SwinIR
- No ROCm support
- Script-based (no CLI framework)
- No model export to multiple formats
- No GUI integration

#### SwinIR (Official)

**Strengths:**
- Reference implementation of SwinIR architecture
- Pre-trained models for all scales
- Clean, well-structured code

**Weaknesses:**
- Single architecture only
- No training pipeline (inference only)
- Basic degradation (bicubic only)
- No ROCm support
- No CLI, no GUI, no config system

## When to Choose sr-engine

| Scenario | Recommendation |
|----------|---------------|
| Production SR deployment | **sr-engine** — CLI, export, GPU abstraction |
| Research on new architectures | **sr-engine** — decorator registry, extensible |
| Real-world blind SR | **Real-ESRGAN** (better degradation) or **sr-engine** + custom degradation |
| ROCm/AMD deployment | **sr-engine** (only option with native support) |
| GUI application integration | **sr-engine** (only option with built-in bridge) |
| Multi-GPU training | **BasicSR** (DDP support) |
| Quick academic comparison | **BasicSR** (more architectures) |
| Video SR pipeline | **sr-engine** (dataset building, tiling, video inference) |

---

# Real-World Case Studies

## Case Study 1: Video Restoration Pipeline for Archive Footage

**Scenario:** A media company needed to upscale 10,000 hours of standard-definition (480p) archival footage to 1080p for streaming.

**Requirements:**
- Throughput: 1 hour of footage processed per GPU-hour
- Quality: No visible artifacts, film grain preserved
- Automation: Fully unattended pipeline with monitoring

**Architecture:**
```
[Archive Storage] → [Extract I-frames] → [sr-engine inference with tiling] → [Re-encode] → [Streaming CDN]
                          │                         │                               │
                    5-minute segments          Tiled inference (512px)        H.264/AV1 encode
```

**Configuration:**
- Model: RRDB (faster than SwinIR, 3× throughput)
- Scale: 3× (480p → 1440p, downscaled to 1080p for headroom)
- Tiling: 512px, overlap 64 (handles variable-resolution archive material)
- Device: 8× RTX 4090 (parallel processing on independent segments)

**Results:**
- Throughput: 45 minutes of processing per hour of footage per GPU
- Total time: ~2 weeks for 10,000 hours (8 GPUs parallel)
- Quality: 32.5 dB PSNR average on held-out test clips
- Grain preservation: Subjective quality score 4.2/5 (expert panel)

**Lesson learned:** Tiling overlap of 64 was insufficient for archive material with heavy film grain; increased to 128 for seamless blending.

## Case Study 2: Real-Time Satellite Image Enhancement

**Scenario:** A geospatial analytics company needed to enhance 50 cm satellite imagery to 12.5 cm resolution.

**Challenges:**
- Massive images (10000×10000+ pixels) — tiling mandatory
- Harsh degradation profile (atmospheric haze, sensor noise, JPEG2000 compression)
- Regulatory requirement: no hallucinated details (only enhancement, not generation)

**Architecture:**
```bash
# Training pipeline
srengine dataset build \
  --input satellite_video.mp4 \
  --degradations blur,noise,jpeg2000 \
  --resize-method lanczos

srengine train run \
  --instance satellite_sr \
  --model swinir \
  --batch-size 4 \
  --patch-size 64 \
  --max-epochs 200 \
  --learning-rate 1e-4 \
  --validation-split 0.15

# Inference pipeline
for tile in $(list_tiles large_image.tif); do
  srengine infer run \
    --model satellite_sr_model.pth \
    --input "$tile" \
    --output "enhanced/$tile" \
    --tile 384 \
    --overlap 128
done
python merge_tiles.py enhanced/ output.tif
```

**Configuration specifics:**
- JPEG2000 degradation enabled (matches satellite downlink compression)
- Lanczos downsampling preserves edge sharpness better than area for this domain
- Small patch size (64) due to large receptive field of SwinIR
- Low learning rate (1e-4) for stable convergence with small patches

**Results:**
- PSNR: 28.7 dB (vs 24.1 dB bicubic baseline)
- SSIM: 0.89 (vs 0.72 bicubic baseline)
- False positive rate (hallucinated features): 0.02% (validated against held-out ground truth)

**Lesson learned:** JP2 config needed quality range adjustment — default `[30, 95]` covered too wide a range. Narrowed to `[60, 90]` for production.

## Case Study 3: Medical X-Ray Super-Resolution

**Scenario:** A hospital network needed to enhance legacy X-ray images (stored as lossy JPEG at low resolution) to improve diagnostic utility.

**Constraints:**
- Medical regulatory requirements (no hallucination, documented confidence)
- Single model for both skeletal and soft-tissue imaging
- Must run on CPU due to hospital IT security policies

**Architecture:**
- Model: SwinIR-small (reduced embed_dim=96, depths=[4,4,4,4])
- Scale: 2× (matching the most common use case)
- Device: CPU with FP32
- Export: ONNX with ONNX Runtime CPU provider

**Training approach:**
1. Built dataset with focus on JPEG compression degradation (quality range [20, 80])
2. Mixed skeletal and soft-tissue images in equal proportion
3. Perceptual loss disabled (risk of hallucination)
4. GAN loss disabled (risk of unrealistic texture generation)
5. Only L1 loss with very conservative training

**Results:**
- CPU inference: 2.4 seconds per 1024×1024 image
- Diagnostic agreement: 94% between enhanced image and original high-res (where available)
- No reported hallucination cases in 6 months of clinical use

**Lesson learned:** For medical applications, strict L1-only training with conservative scale (2×, not 4×) was essential. Perceptual and GAN losses introduced unrealistic textures that radiologists correctly rejected.

---

# Advanced Topics

## Custom Model Architecture Integration

### Adding a New Model

```python
# models/archs/my_hat.py
from ..registry import register

@register("hat")
class HAT(nn.Module):
    """Hybrid Attention Transformer for SR."""

    def __init__(self, num_in_ch=3, num_out_ch=3, embed_dim=180,
                 depths=[6,6,6,6,6,6], num_heads=[6,6,6,6,6,6],
                 window_size=8, scale=4, **kwargs):
        super().__init__()
        # ... architecture implementation ...

    def forward(self, x):
        # ... forward pass ...
        return x
```

```python
# models/archs/__init__.py
from . import rrdbnet
from . import swinir
from . import my_hat  # Add this line
```

```yaml
# utils/configs/models/hat.yaml
name: hat
num_in_ch: 3
num_out_ch: 3
embed_dim: 180
depths: [6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6]
window_size: 8
scale: 4
```

The model is now available:
```bash
srengine train run --dataset my_set --model hat
```

### Custom Loss Functions

```python
# models/losses.py

class EdgeLoss(nn.Module):
    """Edge-preserving loss using Sobel gradients."""

    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.sobel_x = self.sobel_x.view(1, 1, 3, 3)
        self.sobel_y = self.sobel_y.view(1, 1, 3, 3)

    def forward(self, pred, target):
        # Gradient magnitude difference
        gx_pred = F.conv2d(pred, self.sobel_x.to(pred.device), padding=1)
        gy_pred = F.conv2d(pred, self.sobel_y.to(pred.device), padding=1)
        gx_target = F.conv2d(target, self.sobel_x.to(target.device), padding=1)
        gy_target = F.conv2d(target, self.sobel_y.to(target.device), padding=1)
        return F.l1_loss(gx_pred, gx_target) + F.l1_loss(gy_pred, gy_target)
```

To use it, modify the Trainer's loss computation (or subclass Trainer).

## ONNX Runtime Deployment

```python
import onnxruntime as ort
import numpy as np
import cv2

# Load model
session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])

# Get input name
input_name = session.get_inputs()[0].name

# Prepare input
img = cv2.imread("input.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)

# Run inference
outputs = session.run(None, {input_name: img})
sr = outputs[0][0]  # (3, H*scale, W*scale)

# Convert back
sr = np.transpose(sr, (1, 2, 0))
sr = np.clip(sr * 255.0, 0, 255).astype(np.uint8)
sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
cv2.imwrite("output.png", sr)
```

## TensorRT Deployment (via ONNX)

```bash
# Convert ONNX to TensorRT engine
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16

# In Python
import tensorrt as trt
import pycuda.driver as cuda

with open("model.engine", "rb") as f:
    engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(f.read())

# ... execution context setup ...
```

## Gradient Accumulation for Large Effective Batch Size

To simulate a batch size of 64 on a GPU that can only fit batch size 8:

```yaml
# Train config
batch_size: 8
accumulation_steps: 8  # Effective batch size: 8 × 8 = 64
```

The `Trainer` can be modified to accumulate gradients:

```python
# In training loop:
loss = loss / accumulation_steps
loss.backward()
if (batch + 1) % accumulation_steps == 0:
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

This feature is not yet in the Trainer (requires code modification).

## Gradient Checkpointing for Memory-Efficient Training

For SwinIR on memory-constrained GPUs, enable gradient checkpointing:

```python
# After model creation:
model = build_model("swinir", config)
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
```

This trades compute for memory: activations are not stored during forward, but recomputed during backward. Approximately 1.3-2× compute overhead for 30-50% memory reduction.

## Dataset Streaming for Very Large Video

For videos too large to extract all frames to disk at once, implement a streaming pipeline:

```python
# Pseudocode for streaming dataset building
cap = cv2.VideoCapture("massive_video.mp4")
frame_idx = 0
batch = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % frame_step == 0:
        batch.append(frame)
    if len(batch) == 100:  # Process in batches of 100
        degrade_batch(batch, output_dir, config)
        batch = []
    frame_idx += 1

# Process remaining
if batch:
    degrade_batch(batch, output_dir, config)
```

This is not built into sr-engine but can be implemented as a custom script using the degradation functions directly.

---

# Future Outlook

## Emerging Trends

### 1. Real-Time Video SR

The trend toward real-time SR (30+ fps on edge devices) drives demand for:
- Smaller, more efficient architectures (efficient SR, lightweight transformers)
- Quantization (ONNX → INT8)
- Hardware-specific optimization (TensorRT, CoreML, OpenVINO)
sr-engine's ONNX export path supports this direction.

### 2. High-Order Degradation Models

Real-ESRGAN's high-order degradation (repeated, randomly-ordered degradation stages) produces more realistic LR images than the classic single-pass model. Future sr-engine versions may adopt this approach.

### 3. Diffusion Models for SR

Diffusion-based SR models (SR3, ResShift, LDM-SR) achieve state-of-the-art perceptual quality but are 10-100× slower than feed-forward models. sr-engine's architecture may need to support diffusion inference loops.

### 4. Video SR with Temporal Consistency

Current frame-by-frame inference does not enforce temporal consistency, causing flickering. Future additions could include:
- Recurrent architectures (e.g., BasicVSR, EDVR)
- Temporal loss during training
- Post-processing temporal smoothing

### 5. Multi-GPU and Distributed Training

As dataset sizes and model capacities grow, distributed training support (DDP, FSDP) becomes essential for reasonable training times.

## Expected Evolution

### Short-Term (0-6 months)

- **High-order degradation** support in the pipeline
- **Additional model architectures** (HAT, DAT)
- **TensorRT export** (via ONNX → TRT)
- **Multi-GPU training** support (DDP)

### Medium-Term (6-12 months)

- **REST API** for inference (FastAPI-based HTTP server)
- **Experiment tracking** integration (MLflow, W&B)
- **Distributed dataset building** across multiple machines
- **Support for non-PNG formats** (WebP, JPEG XL)

### Long-Term (12+ months)

- **Video SR with temporal modeling** (recurrent/3D architectures)
- **Diffusion model support** for maximum quality
- **Cloud deployment** templates (Kubernetes, AWS Batch)
- **Active learning** for dataset curation (identify and prioritize informative frames)
- **Automated architecture search** for deployment-specific optimization

---

# References

## Official Documentation

| Resource | Link |
|----------|------|
| sr-engine README | [README.md](../README.md) |
| Architecture Guide | [docs/architecture.md](architecture.md) |
| CLI Reference | [docs/cli-reference.md](cli-reference.md) |
| Training Guide | [docs/training.md](training.md) |
| Data Pipeline Guide | [docs/data-pipeline.md](data-pipeline.md) |
| Degradation Pipeline Guide | [docs/degradation-pipeline.md](degradation-pipeline.md) |
| Inference Guide | [docs/inference.md](inference.md) |
| Device Backend Guide | [docs/device-backend.md](device-backend.md) |
| Workspace Guide | [docs/workspace.md](workspace.md) |
| GUI Bridge Guide | [docs/gui_bridge.md](gui_bridge.md) |
| Development Guide | [docs/development.md](development.md) |

## Academic Papers

| Paper | Year | Relevance |
|-------|------|-----------|
| [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219) | 2018 | RRDB architecture, perceptual+GAN loss |
| [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257) | 2021 | SwinIR architecture |
| [Blind Super-Resolution with Iterative Kernel Correction](https://arxiv.org/abs/1904.03377) | 2019 | Blind SR methodology |
| [Real-ESRGAN: Training Real-World Blind Super-Resolution](https://arxiv.org/abs/2107.10833) | 2021 | High-order degradation model |
| [BSRGAN: Designing a Practical Degradation Model](https://arxiv.org/abs/2103.14006) | 2021 | Classic degradation model design |
| [Perceptual Losses for Real-Time Style Transfer](https://arxiv.org/abs/1603.08155) | 2016 | Perceptual loss (VGG-based) |
| [Photo-Realistic Single Image Super-Resolution Using a GAN](https://arxiv.org/abs/1609.04802) | 2017 | SRGAN, GAN loss for SR |
| [Image Quality Assessment: From Error Visibility to SSIM](https://ieeexplore.ieee.org/document/1284395) | 2004 | SSIM metric |

## Standards and Specifications

| Standard | Description |
|----------|-------------|
| [ONNX](https://onnx.ai/) | Open Neural Network Exchange format |
| [SafeTensors](https://github.com/huggingface/safetensors) | Safe tensor storage format |
| [JSON Lines](https://jsonlines.org/) | NDJSON format for metrics streaming |
| [YAML 1.2](https://yaml.org/spec/1.2.2/) | Configuration file format |

## Libraries and Tools

| Tool | Version | Purpose |
|------|---------|---------|
| [PyTorch](https://pytorch.org/) | ≥2.0 | Deep learning framework |
| [Click](https://click.palletsprojects.com/) | ≥8.1 | CLI framework |
| [OpenCV](https://opencv.org/) | ≥4.8 | Image/video processing |
| [NumPy](https://numpy.org/) | ≥1.24 | Array operations |
| [PyYAML](https://pyyaml.org/) | ≥6.0 | YAML parsing |
| [tqdm](https://tqdm.github.io/) | ≥4.66 | Progress bars |
| [Pillow](https://python-pillow.org/) | ≥10.0 | Image format support |
| [ONNX Runtime](https://onnxruntime.ai/) | — | Cross-platform inference (optional) |

## Community and Support

| Resource | URL |
|----------|-----|
| GitHub Repository | [https://github.com/your-org/sr-engine](https://github.com/your-org/sr-engine) |
| Issue Tracker | GitHub Issues |
| License | MIT |

---

*This document is maintained as part of the sr-engine project. For corrections or additions, please submit a pull request or open an issue on the GitHub repository. Last updated: July 2026.*



