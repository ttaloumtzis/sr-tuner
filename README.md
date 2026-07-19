# sr-engine

Super-resolution (SR) engine for training and running deep learning-based super-resolution models on images and video. Supports both NVIDIA (CUDA) and AMD (ROCm) GPUs with a unified CLI, a FastAPI REST API, a configurable desktop GUI (React/Tauri), model versioning, and configurable degradation pipelines.

## Features

- **Dataset building** — Extract frames from video files and apply a configurable degradation pipeline (blur, noise, JPEG/JPEG2000 compression, color jitter, downsampling) to produce paired HR/LR training datasets
- **Dataset validation & health checks** — Validate dataset structure, detect corrupt images, verify scale-factor dimension alignment, detect and prune black frames
- **Dataset merging** — Combine multiple datasets grouped by scale factor into unified training sets
- **Model training** — Train RRDB (ESRGAN-style) and SwinIR super-resolution models with configurable hyperparameters, mixed-precision (bf16/fp16), cosine LR scheduling with warmup, validation split, and streaming JSONL metrics
- **Model inference** — Run inference on images or videos with VRAM-safe tiled processing
- **Model export** — Export trained models to ONNX, TorchScript, or SafeTensors format
- **Model versioning** — Named model instances with versioned checkpoints and per-run metadata tracking
- **Workspace management** — Structured project directories (`datasets/`, `models/`, `experiments/`, `configs/`) with auto-discovery via workspace marker file
- **Device abstraction** — Unified CUDA/ROCm backend detection with autocast dtype selection and flash attention support
- **HTTP API server** — FastAPI-based REST API with real-time SSE progress streaming, background job management, and comprehensive endpoints for all operations
- **Desktop GUI** — React/TypeScript frontend with Tauri 2 desktop shell, providing a native interface for project management, dataset operations, model training, inference, and checkpoint management
- **CLI** — Full-featured Click-based command-line interface with standalone command aliases

## Requirements

- Python 3.11 (strictly pinned in `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Linux recommended (ROCm/CUDA); Windows works for CUDA/CPU
- GPU optional — CPU-only mode works everywhere

## Installation

```bash
# Clone the repository
git clone https://github.com/ttaloumtzis/sr-tuner.git
cd sr-tuner

# CPU-only (no GPU)
./envs/build.sh --backend cpu

# NVIDIA CUDA
./envs/build.sh --backend cuda

# AMD ROCm
./envs/build.sh --backend rocm
```

The build script creates a `.venv`, installs dependencies via `uv sync`, pins the correct PyTorch index, and runs `envs/verify_env.py` to confirm the setup.

Manual alternative:
```bash
uv venv
uv sync
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

## Usage

### CLI entry point

All commands are accessible via the `srengine` CLI:

```bash
srengine --help
```

### Workspace management

```bash
# Initialize a workspace tree
srengine workspace init

# Show workspace summary
srengine workspace info

# Validate workspace structure
srengine workspace check
```

### Dataset operations

```bash
# Build a dataset from a video file (extract frames + apply degradations)
srengine dataset build --input video.mp4 --out ./datasets/my_set

# Validate an existing dataset directory
srengine dataset validate --path ./datasets/my_set

# Run a health check (detect black frames, resolution distribution, etc.)
srengine dataset health --path ./datasets/my_set

# Merge multiple datasets grouped by scale
srengine dataset merge --input ./datasets/sources --out ./datasets/merged
```

### Model management

```bash
# Create a named model instance (stores arch config, checkpoints, runs)
srengine model create-instance my_model --model rrdb_esrgan

# List instances
srengine model list-instances

# View instance details
srengine model info --instance my_model

# Export a trained model
srengine model export --instance my_model --format onnx --out model.onnx
```

### Training

```bash
# Train with default config
srengine train run --model rrdb_esrgan --dataset ./datasets/my_set

# Train with a custom config and mixed precision
srengine train run --config my_config.yaml --dataset ./datasets/my_set \
    --model swinir --bf16

# Train with a model instance (auto-versioning and checkpoints)
srengine train run --instance my_model --dataset ./datasets/my_set

# Resume from a checkpoint
srengine train run --instance my_model --dataset ./datasets/my_set --resume v3

# Emit metrics as JSONL (for programmatic consumption)
srengine train run --instance my_model --dataset ./datasets/my_set --machine
```

### Inference

```bash
# Super-resolve an image
srengine infer run --model checkpoints/model.pth \
    --input input.png --output output.png

# Use a model instance (auto-resolves latest version)
srengine infer run --instance my_model \
    --input input.png --output output.png

# Super-resolve a video
srengine infer run --model checkpoints/model.pth \
    --input video.mp4 --output output.mp4

# Tiled inference (VRAM-safe for large images)
srengine infer run --model checkpoints/model.pth \
    --input large.png --output output.png --tile 512 --overlap 64
```

### Environment diagnostics

```bash
srengine env check
srengine env bench --model rrdb_esrgan --iterations 50
```

### API server

Start the FastAPI-based HTTP server to enable the desktop GUI or third-party integrations:

```bash
# Start on default port (8765)
srengine serve start

# Custom port and host
srengine serve start --host 0.0.0.0 --port 8080
```

The server provides:
- **RESTful endpoints** for all operations (workspace, datasets, models, training, inference)
- **Server-Sent Events (SSE)** for real-time job progress
- **Background job management** with cancellation and status tracking
- **CORS-enabled** for cross-origin frontend access

### Desktop GUI

The project includes a native desktop application built with **Tauri 2** (Rust shell) and **React 18 + TypeScript**:

```bash
# Prerequisites: Rust toolchain, Node.js 18+
cd frontend
npm install

# Development mode (requires API server running)
npm run dev

# Production build
npm run build
# Or build the Tauri desktop app
npx tauri build
```

The GUI provides tab-based navigation for:
- **Project management** — Create, open, and manage workspaces
- **Dataset operations** — Build from video, browse, validate, merge, health check
- **Model management** — Create instances, view version history, export checkpoints
- **Training** — Configure hyperparameters, launch runs, live metrics dashboard
- **Inference** — Drag-and-drop image input, before/after comparison with metrics
- **Checkpoints** — Browse, sort, filter, and manage model checkpoints

### Python API

The engine modules can also be used directly from Python:

```python
from sr_engine.models.registry import build_model
from sr_engine.data.datasets import PairedImageFolderDataset
from sr_engine.engine.trainer import Trainer

# Build a model
model = build_model("swinir", {"name": "swinir", "scale": 4, "embed_dim": 180})

# Load a dataset
dataset = PairedImageFolderDataset("datasets/my_set")

# Run inference on an image
from sr_engine.engine.inference import infer_image
infer_image(
    model_checkpoint="checkpoints/model.pth",
    input_path="input.png",
    output_path="output.png",
    device="cuda",
)
```

## Project Structure

```
├── envs/
│   ├── build.sh                 # Environment builder (uv + PyTorch backend)
│   ├── verify_env.py            # Post-install verification script
│   └── docker/                  # Dockerfiles (if any)
├── src/
│   └── sr_engine/
│       ├── cli/                 # Click CLI commands
│       │   ├── main.py          # CLI entry point (srengine)
│       │   ├── cmd_dataset.py   # dataset build, validate, health, merge
│       │   ├── cmd_train.py     # train run
│       │   ├── cmd_infer.py     # infer run
│       │   ├── cmd_model.py     # model create-instance, list-instances, export, info
│       │   ├── cmd_env.py       # env check, env bench
│       │   ├── cmd_serve.py     # serve start
│       │   ├── workspace_commands.py  # workspace init, info, check
│       │   └── helpers.py       # Config loading, progress, workspace resolution
│       ├── api/                 # FastAPI REST API server
│       │   ├── app.py           # FastAPI app, lifespan, CORS, SSE endpoint
│       │   ├── schemas.py       # Pydantic request/response models
│       │   ├── deps.py          # Dependency injection (workspace, configs)
│       │   ├── task_manager.py  # Background task registry
│       │   ├── event_manager.py # SSE event bus
│       │   ├── callbacks.py     # SSECallback for trainer integration
│       │   ├── progress.py      # SSEProgressReporter
│       │   ├── workers.py       # Background worker threads
│       │   └── routes/          # API route handlers
│       │       ├── workspace.py
│       │       ├── models.py
│       │       ├── training.py
│       │       ├── inference.py
│       │       ├── datasets.py
│       │       ├── jobs.py
│       │       └── env.py
│       ├── data/                # Data pipeline
│       │   ├── dataset_builder.py  # build_from_video, build_from_preprocessed
│       │   ├── dataset_validator.py# Dataset validation (ValidationReport, validate)
│       │   ├── dataset_health.py   # Health checks, black frame detection/pruning
│       │   ├── dataset_merge.py    # Multi-dataset merging by scale
│       │   ├── datasets.py         # PairedImageFolderDataset (PyTorch Dataset)
│       │   ├── degrade.py          # HR→LR degradation pipeline
│       │   ├── transforms.py       # RandomCrop, RandomFlip, RandomRotate, CenterCrop
│       │   └── video_extract.py    # Video frame extraction
│       ├── device/                  # Hardware abstraction
│       │   ├── backend.py          # CUDA/ROCm detection, autocast, flash attention
│       │   └── kernels.py          # Backend-aware op dispatchers (SDPA, Conv2d)
│       ├── engine/                  # Training & inference engine
│       │   ├── trainer.py          # Trainer, TrainerCallback, _MetricsStreamCallback
│       │   ├── inference.py        # infer_image, infer_video
│       │   ├── metrics.py          # psnr, ssim, lpips
│       │   ├── metrics_stream.py   # MetricsStream (JSONL output)
│       │   └── tiling.py           # tile_image, stitch_tiles
│       ├── models/                  # Model architectures
│       │   ├── archs/
│       │   │   ├── rrdbnet.py      # RRDB-ESRGAN (registered as "rrdb_esrgan")
│       │   │   └── swinir.py       # SwinIR (registered as "swinir")
│       │   ├── registry.py         # Model registry (register, build_model)
│       │   ├── checkpoint.py       # Save/load checkpoints, export to ONNX/TorchScript/SafeTensors
│       │   └── losses.py           # L1Loss (Charbonnier), PerceptualLoss, GANLoss
│       ├── utils/                   # Utilities
│       │   ├── config.py           # YAML config loader/merger (DefaultConfigs)
│       │   ├── configs/            # Built-in YAML config files
│       │   │   ├── train/base.yaml
│       │   │   ├── datasets/video_pairs.yaml
│       │   │   └── models/{swinir,rrdb_esrgan}.yaml
│       │   ├── io.py               # Image read/write utilities
│       │   ├── logging.py          # get_logger
│       │   └── progress.py         # ProgressReporter, TqdmReporter
│       └── workspace.py           # Workspace discovery/init, ModelInstance, versioning
├── frontend/              # React/TypeScript desktop GUI
│   ├── src/
│   │   ├── App.tsx        # Root component with tab routing
│   │   ├── screens/       # 6 tab screens (project, dataset, model, training, metrics, inference, checkpoints)
│   │   ├── store/         # 9 Zustand state stores
│   │   ├── hooks/         # Custom React hooks (SSE, etc.)
│   │   ├── components/    # Reusable UI components
│   │   └── lib/           # API client, types, utilities
│   ├── package.json
│   └── vite.config.ts
├── src-tauri/             # Tauri 2 desktop shell (Rust)
│   ├── src/lib.rs         # Tauri commands (Python server lifecycle, filesystem)
│   ├── Cargo.toml
│   └── tauri.conf.json
├── tests/                # Test suite (pytest)
│   ├── conftest.py       # Shared fixtures and helpers
│   ├── test_trainer.py   # Trainer tests
│   ├── test_inference.py # Inference tests
│   ├── test_backend.py   # Device backend tests
│   ├── test_datasets.py  # Dataset tests
│   ├── test_degrade.py   # Degradation tests
│   ├── test_models.py    # Model architecture tests
│   ├── test_cli_*.py     # CLI command tests
│   └── ...               # Additional test modules
├── docs/                 # Documentation files
│   ├── architecture.md
│   ├── cli-reference.md
│   ├── api-reference.md
│   ├── frontend.md
│   ├── desktop.md
│   ├── training.md
│   ├── data-pipeline.md
│   ├── inference.md
│   ├── workspace.md
│   ├── device-backend.md
│   └── ...
├── pyproject.toml        # Project metadata and build configuration
├── LICENSE               # MIT License
└── README.md
```

## Configuration

Default configuration files are bundled in `src/sr_engine/utils/configs/` and are automatically copied to the workspace on `workspace init`. You can override any setting by editing the workspace copies, or pass a custom YAML via `--config`.

| Config file | Contents |
|---|---|
| `configs/train/base.yaml` | Training hyperparameters (batch size, LR, epochs, loss weights, mixed precision) |
| `configs/datasets/video_pairs.yaml` | Degradation pipeline settings (blur, noise, JPEG, JPEG2000, color jitter, resize) |
| `configs/models/swinir.yaml` | SwinIR architecture parameters |
| `configs/models/rrdb_esrgan.yaml` | RRDB-ESRGAN architecture parameters |

## Supported Models

| Name | Registry key | Architecture | Scale |
|---|---|---|---|
| RRDB-ESRGAN | `rrdb_esrgan` | Residual-in-Residual Dense Blocks + nearest-neighbour upsampler | 4× (configurable) |
| SwinIR | `swinir` | Swin Transformer + pixel-shuffle upsampler | 4× (configurable) |

## Development

```bash
# Install dev dependencies (testing, linting)
uv sync --group dev

# Run tests
uv run pytest tests/

# Run tests with coverage
uv run pytest tests/ --cov=sr_engine

# Lint
uv run ruff check src/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes and add tests
4. Run the test suite (`uv run pytest tests/`)
5. Run the linter (`uv run ruff check src/`)
6. Submit a pull request

Issues and feature requests can be filed at [https://github.com/ttaloumtzis/sr-tuner/issues](https://github.com/ttaloumtzis/sr-tuner/issues).

## License

MIT License — see [LICENSE](LICENSE) for details.
