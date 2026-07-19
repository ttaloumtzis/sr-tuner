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

## Dependencies

### Python (CLI + API Server)

The project requires **Python >=3.11, <3.13** and uses [uv](https://docs.astral.sh/uv/) as the package manager.

```bash
# CPU-only
./envs/build.sh --backend cpu

# NVIDIA CUDA
./envs/build.sh --backend cuda

# AMD ROCm
./envs/build.sh --backend rocm
```

The build script creates a `.venv`, runs `uv sync --no-dev`, installs the correct PyTorch wheel for the chosen backend (`cpu`, `cu121`, or `rocm6.2`), and runs `envs/verify_env.py` to confirm the setup.

| Package | Purpose |
|---|---|
| PyTorch + torchvision | Deep learning engine |
| click | CLI framework |
| fastapi + uvicorn | REST API server |
| opencv-python | Video frame extraction |
| pillow | Image I/O |
| numpy | Array operations |
| pyyaml | Configuration files |
| psutil | System monitoring |
| tqdm | Progress bars |

### GUI Frontend (Node.js / React)

Requires **Node.js >=18**. Dependencies are managed via npm:

```bash
cd frontend
npm install         # installs everything in package.json
npm run dev         # Vite dev server (requires API on :8765)
npm run build       # production build
```

| Package | Purpose |
|---|---|
| react + react-dom | UI framework |
| zustand | State management |
| @tauri-apps/api | Desktop integration |
| @tauri-apps/plugin-dialog | Native file dialogs |
| @tauri-apps/plugin-fs | Native filesystem access |
| @tauri-apps/plugin-shell | Subprocess management |
| typescript | Type safety |
| vite | Dev server & bundler |

### GUI Shell (Rust / Tauri)

Requires the **Rust toolchain** (edition 2021). Dependencies are fetched by Cargo during the Tauri build:

```bash
npx tauri build     # also compiles the Rust shell
```

| Crate | Purpose |
|---|---|
| tauri | Desktop shell |
| tauri-plugin-dialog | Native file dialogs |
| tauri-plugin-fs | Native filesystem access |
| ureq | Python server health checks |
| serde + serde_json | Serialization |

## Quick Start

First, set up the environment (see [Dependencies](#dependencies) above), then:

```bash
# Initialize a workspace
srengine workspace init

# Build a dataset from a video
srengine dataset build --input video.mp4 --out ./datasets/my_set

# Train a model
srengine train run --model rrdb_esrgan --dataset ./datasets/my_set

# Run inference
srengine infer run --model checkpoints/model.pth --input input.png --output output.png

# Start the API server (required for the GUI)
srengine serve start
```

Full CLI reference is available in [`docs/cli-reference.md`](docs/cli-reference.md) and all commands support `--help`.

### Desktop GUI

```bash
cd frontend
npm install
npm run dev         # development (requires API server on :8765)
npm run build       # production build
npx tauri build     # standalone desktop binary
```

The GUI provides tab-based navigation for project management, dataset operations, model management, training (with live metrics), inference (drag-and-drop), and checkpoint browsing.

### Python API

```python
from sr_engine.models.registry import build_model
from sr_engine.data.datasets import PairedImageFolderDataset
from sr_engine.engine.inference import infer_image

model = build_model("swinir", {"name": "swinir", "scale": 4, "embed_dim": 180})
dataset = PairedImageFolderDataset("datasets/my_set")
infer_image(model_checkpoint="checkpoints/model.pth", input_path="input.png",
            output_path="output.png", device="cuda")
```

## Project Structure

```
├── envs/               # Environment builder & verification scripts
├── src/sr_engine/
│   ├── cli/            # Click CLI commands
│   ├── api/            # FastAPI REST API (routes, schemas, SSE, task mgmt)
│   ├── data/           # Dataset pipeline (build, validate, degrade, merge)
│   ├── device/         # CUDA/ROCm abstraction layer
│   ├── engine/         # Training loop, inference, metrics
│   ├── models/         # Model architectures (RRDB, SwinIR), checkpointing
│   └── utils/          # Config loading, I/O, logging, progress reporting
├── frontend/           # React/TypeScript desktop GUI
├── src-tauri/          # Tauri 2 Rust shell
├── tests/              # pytest test suite
└── docs/               # Documentation
```

Detailed architecture is documented in [`docs/architecture.md`](docs/architecture.md).

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
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest tests/

# Lint
uv run ruff check src/
```

## License

MIT License — see [LICENSE](LICENSE) for details.
