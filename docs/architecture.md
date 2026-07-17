# Architecture

## Layered Module Design

sr-engine follows a **layered modular architecture** with strict one-directional dependencies. Each layer depends only on the layers below it.

```
┌──────────────────────────────────────────────────────────────┐
│                    CLI Layer (cli/)                           │
│  main.py │ cmd_train.py │ cmd_infer.py │ cmd_dataset.py      │
│  cmd_model.py │ cmd_env.py │ cmd_serve.py │ workspace_cmds   │
│  helpers.py                                                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐      │
│  │           Config Layer (utils/config.py)            │      │
│  │  4-level merge: builtin → module → file → CLI      │      │
│  └────────────────────┬───────────────────────────────┘      │
│                       │                                      │
│  ┌────────────────────▼───────────────────────────────┐      │
│  │            Workspace Layer (workspace.py)           │      │
│  │  Auto-detect, project/model CRUD, dataset resolve  │      │
│  └────────────────────┬───────────────────────────────┘      │
│                       │                                      │
├───────────────────────┼──────────────────────────────────────┤
│                       │                                      │
│  ┌────────────────────▼──────────┐  ┌────────────────────┐   │
│  │     Data Pipeline (data/)     │  │  Engine (engine/)  │   │
│  │                               │  │                    │   │
│  │  video_extract.py             │  │  trainer.py        │   │
│  │  degrade.py                   │  │  inference.py      │   │
│  │  dataset_builder.py           │  │  tiling.py         │   │
│  │  dataset_validator.py         │  │  metrics.py        │   │
│  │  dataset_health.py            │  │  metrics_stream.py │   │
│  │  datasets.py  (Dataset)       │  │                    │   │
│  │  transforms.py                │  │                    │   │
│  └───────────────────────────────┘  └────────────────────┘   │
│                       │                                      │
│  ┌────────────────────▼───────────────────────────────┐      │
│  │            Models Layer (models/)                   │      │
│  │  registry.py │ checkpoint.py │ losses.py            │      │
│  │  archs/rrdbnet.py │ archs/swinir.py                │      │
│  └────────────────────┬───────────────────────────────┘      │
│                       │                                      │
│  ┌────────────────────▼───────────────────────────────┐      │
│  │       Device Layer (device/)                        │      │
│  │  backend.py │ kernels.py                            │      │
│  │  CUDA/ROCm detection, flash-attn, AMP               │      │
│  └────────────────────────────────────────────────────┘      │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│              GUI Bridge Layer (gui_bridge/)                   │
│  server.py │ jobs.py │ protocol.py                           │
│  TCP/JSON server for Godot client integration                │
└──────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

| Module | Package | Responsibility |
|--------|---------|----------------|
| **CLI** | `cli/` | Click command tree, argument parsing, terminal output |
| **Config** | `utils/config.py` | YAML loading, 4-level merge, validation |
| **Workspace** | `workspace.py` | Directory auto-discovery, path resolution, project/instance CRUD |
| **Data** | `data/` | Video frame extraction, HR→LR degradation, dataset building/validation, PyTorch Dataset, transforms |
| **Engine** | `engine/` | Training loop, inference pipeline, metrics computation, tiled inference |
| **Models** | `models/` | Architecture definitions, decorator-based registry, checkpoint save/load/export, loss functions |
| **Device** | `device/` | CUDA/ROCm detection, dtype selection, backend-aware kernels |
| **GUI Bridge** | `gui_bridge/` | TCP server, subprocess lifecycle, NDJSON protocol |

## Data Flow

### Training

```
User CLI input
      │
      ▼
  CLI layer ──► Config layer (merge defaults + file + flags)
      │
      ├──► Workspace layer (resolve dataset/project paths)
      │
      ├──► Data layer (PairedImageFolderDataset)
      │       │
      │       ├── Load HR image
      │       ├── Load LR image
      │       └── Apply transforms (crop, flip, rotate)
      │
      ├──► Models layer
      │       │
      │       ├── registry.build_model("swinir", config)
      │       ├── Move to device (CUDA/ROCm/CPU)
      │       └── Build optimizer + scheduler
      │
      └──► Engine layer (Trainer)
              │
              ├── For each epoch:
              │     ├── Training loop (forward → loss → backward → step)
              │     ├── Validation loop (PSNR, SSIM)
              │     └── Checkpoint save
              │
              └── Via callbacks:
                    ├── MetricsStream → JSONL file
                    ├── TqdmReporter → terminal
                    └── SocketCallback → GUI
```

### Dataset Building

```
Video file (.mp4, .avi)
      │
      ▼
  video_extract.py ──► Frames (PNG sequence)
      │
      ▼
  degrade.py ──► LR frames
      │
      ├── Blur (Gaussian kernel)
      ├── Noise (additive Gaussian)
      ├── Downscale (bicubic)
      └── JPEG compression (optional)
      │
      ▼
  Dataset directory:
    HR/ (original frames)
    LR/ (degraded frames)
    manifest.json (pairs index)
      │
      ▼
  dataset_validator.py
      │
      ├── Check HR/LR structure
      ├── Verify dimension ratios
      └── Confirm all files readable
```

### Inference

```
Input image/video
      │
      ▼
  inference.py
      │
      ├── Image path → load → model → save
      │
      └── Video path → extract frames → for each:
              │
              ├── tiling.py (if --tile > 0)
              │     ├── tile_image() → split into overlapping patches
              │     ├── model → upscale each tile
              │     └── stitch_tiles() → reassemble with blending
              │
              └── No tiling → model → full-frame upscale
              │
              ▼
      Write output (image or encoded video)
```

## Design Patterns

### Registry (Decorator)

`models/registry.py` provides a `@register("name")` decorator that auto-registers model classes. New models self-register at import time without modifying any registration code.

```python
@register("swinir")
class SwinIR(nn.Module):
    ...

# Usage:
model = build_model("swinir", config)
```

### Callback

`engine/trainer.py` defines `TrainerCallback` with lifecycle hooks. The trainer calls these at each phase, step, validation, and completion. Multiple callbacks can be attached simultaneously.

```python
class TrainerCallback:
    def on_phase(self, phase, ...): pass
    def on_step(self, epoch, batch, ...): pass
    def on_validate(self, epoch, psnr, ssim): pass
    def on_done(self, ...): pass
```

Used by:
- `_MetricsStreamCallback` — writes JSONL metrics
- `SocketCallback` — streams events to GUI over TCP

### Strategy (via Composition)

`cli/helpers.py:resolve_reporter()` returns different implementations based on environment:

| Environment | Returns | Behaviour |
|---|---|---|
| `SRENGINE_GUI_SOCKET` set | `SocketReporter` | Sends NDJSON progress events over TCP |
| Terminal (default) | `TqdmReporter` | Renders tqdm progress bar |
| `--machine` flag | `SocketReporter` via MetricsStream | Writes JSONL to file |

### Template Method

`utils/progress.py:ProgressReporter` is an abstract base with no-op defaults. Subclasses override specific methods:

```
ProgressReporter (abstract)
    ├── TqdmReporter    — overrides start/update/end
    └── SocketReporter   — overrides start/update/end/postfix
```

### Builder

`data/dataset_builder.py` orchestrates multi-step dataset creation:

```
build_from_video(video_path, output_dir)
    ├── extract_frames(video)         → raw frames
    ├── degrade_frames(frames, cfg)   → HR/LR pairs
    ├── validate(directory)           → integrity check
    └── write_manifest(directory)     → index file
```

### Adapter

`engine/trainer.py:_MetricsStreamCallback` adapts `MetricsStream` (a file writer) into the `TrainerCallback` interface, enabling metrics logging without modifying the trainer's core loop.

### Config Layering

`utils/config.py:DefaultConfigs` implements a 4-level precedence:

```
1. Built-in YAMLs     (utils/configs/*.yaml)        ← lowest priority
2. Workspace YAMLs    (<ws>/configs/**/*.yaml)      │
3. --config file      (user-provided YAML)           │
4. CLI flags          (--batch-size, --lr, etc.)    ← highest priority
```

Each level recursively merges onto the previous. CLI flags win everything.

### Subprocess (GUI Bridge)

`gui_bridge/jobs.py:JobManager` spawns long-running tasks (train, infer, dataset build) as subprocesses. Each subprocess connects back to the server via a control socket for progress streaming and cancellation detection. The server tracks job state in manifest files persisted to the workspace.

## Directory Layout

```
src/sr_engine/
├── __init__.py
├── workspace.py                  # Workspace discovery, project/model CRUD
├── cli/
│   ├── main.py                   # Root srengine Click group
│   ├── cmd_train.py              # srengine train run
│   ├── cmd_infer.py              # srengine infer run
│   ├── cmd_dataset.py            # srengine dataset {build,validate,health,merge}
│   ├── cmd_model.py              # srengine model {export,info,list-instances,...}
│   ├── cmd_env.py                # srengine env {check,bench}
│   ├── cmd_serve.py              # srengine serve start
│   ├── workspace_commands.py     # workspace CLI commands (init, info, check)
│   └── helpers.py                # Shared CLI utilities
├── data/
│   ├── datasets.py               # PairedImageFolderDataset
│   ├── dataset_builder.py        # build_from_video(), build_from_preprocessed()
│   ├── dataset_validator.py      # validate() → ValidationReport
│   ├── dataset_health.py         # Profile, black frame detection/pruning
│   ├── video_extract.py          # Frame extraction from video
│   ├── degrade.py                # HR→LR degradation pipeline
│   └── transforms.py             # RandomCrop, RandomFlip, RandomRotate, Compose
├── engine/
│   ├── trainer.py                # Trainer with callback hooks
│   ├── inference.py              # infer_image(), infer_video()
│   ├── tiling.py                 # tile_image(), stitch_tiles()
│   ├── metrics.py                # psnr(), ssim(), lpips()
│   └── metrics_stream.py         # JSONL metrics writer
├── models/
│   ├── registry.py               # @register decorator, build_model()
│   ├── checkpoint.py             # Save/load/export (ONNX, safetensors, TorchScript)
│   ├── losses.py                 # Charbonnier L1, PerceptualLoss, GANLoss
│   └── archs/
│       ├── rrdbnet.py            # RRDBNet (CNN-based)
│       └── swinir.py             # SwinIR (Transformer-based)
├── device/
│   ├── backend.py                # get_device(), is_rocm(), autocast_dtype()
│   └── kernels.py                # scaled_dot_product_attention(), get_conv2d()
├── utils/
│   ├── config.py                 # DefaultConfigs, merge_overrides, validate_config
│   ├── io.py                     # read_image(), write_image(), ensure_dir()
│   ├── logging.py                # get_logger()
│   ├── progress.py               # ProgressReporter, TqdmReporter
│   └── configs/                  # Built-in YAML configs
    │       ├── train/base.yaml
│       ├── datasets/video_pairs.yaml
│       ├── models/swinir.yaml
│       └── models/rrdb_esrgan.yaml
└── gui_bridge/
    ├── server.py                 # TCP Server with dual accept loops
    ├── jobs.py                   # JobManager, subprocess lifecycle
    └── protocol.py               # SocketReporter, SocketCallback, handshake
```

## Dependencies

sr-engine minimizes runtime dependencies. PyTorch is deliberately excluded from `pyproject.toml` — it is installed separately via `envs/build.sh` with a backend-specific index URL (CPU, CUDA 12.4, ROCm 6.2), avoiding unnecessary CUDA library downloads.

| Dependency | Version | Purpose |
|---|---|---|
| `click` | >=8.1 | CLI framework |
| `pyyaml` | >=6.0 | YAML config parsing |
| `numpy` | >=1.24 | Array operations |
| `opencv-python` | >=4.8 | Image/video I/O |
| `pillow` | >=10.0 | Image format support |
| `tqdm` | >=4.66 | Progress bars |

Optional (lazily imported): `torchvision` (VGG19 for PerceptualLoss), `lpips` (LPIPS metric), `safetensors` (SafeTensors export).
