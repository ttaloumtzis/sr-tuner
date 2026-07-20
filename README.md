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

## Prerequisites

The project uses three runtimes. You only need the ones relevant to how you use sr-engine.

| Component | Runtime | Needed for |
|-----------|---------|------------|
| CLI + API server | **Python** ≥3.11, <3.13 + [uv](https://docs.astral.sh/uv/) | All ML operations, dataset pipeline, REST API |
| Desktop GUI | **Node.js** ≥18 + npm | React frontend development and builds |
| Desktop shell | **Rust** (edition 2021) | Building the native Tauri desktop binary |

### Quick setup

Once the relevant runtimes are installed, set up the Python environment:

```bash
# Linux / macOS
./envs/build.sh --backend cpu       # or --backend cuda / --backend rocm

# Windows (PowerShell)
.\envs\build.ps1 -Backend cpu        # or -Backend cuda
```

The build script creates a `.venv` with `uv sync`, installs the correct PyTorch wheel, and verifies the setup.

### Installing runtimes

**Python + uv:**

```bash
# Linux / macOS
curl -fsSL https://astral.sh/uv/install.sh | bash

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> Python 3.11 or 3.12 must be available. On Linux: `apt install python3.11 python3.11-venv`. On macOS: `brew install python@3.12`. On Windows: download from [python.org](https://www.python.org/downloads/).

**Node.js (requires 18+):**

**Linux / macOS:**
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
nvm install 18
```

**Windows:**
```powershell
# via nvm-windows
winget install nvm-windows
# or download from https://github.com/coreybutler/nvm-windows/releases
nvm install 18
nvm use 18
```

**Rust (via rustup):**

**Linux / macOS:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Windows:**
Download and run [rustup-init.exe](https://rustup.rs/) — or use `winget install Rustlang.Rustup`.

### Per-platform notes

**Linux** — Tauri desktop build requires system libraries:

```bash
sudo apt install -y build-essential libwebkit2gtk-4.1-dev librsvg2-dev patchelf libssl-dev
```

**macOS** — Install via [Homebrew](https://brew.sh/): `brew install python@3.12 uv node rust`.

**Windows:**
- **ROCm** is not supported on Windows. Use `-Backend cpu` or `-Backend cuda`.
- **WSL** (recommended): run `./envs/build.sh` inside Ubuntu on WSL for full bash compatibility. The frontend and desktop app run from Windows and connect to the API server in WSL.
- **Tauri desktop build** requires [MSVC Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), WebView2 (included in Win10 1803+), and the Rust MSVC toolchain. Output is `.msi` / `.exe` in `src-tauri/target/release/bundle/`.
- **Known caveats:**

| Issue | Mitigation |
|-------|------------|
| `nvidia-smi` outputs `,` as decimal separator on some European locale systems | The GPU polling code normalizes `,` → `.` before parsing float values |
| Frontend path separators | All path operations use cross-platform `join()`/`basename()` utilities |
| Virtual environment activation | Use `.venv\Scripts\Activate.ps1` (not `source .venv/bin/activate`) |

### Python packages

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

### Frontend packages (React / TypeScript)

Managed via npm — `cd frontend && npm install`:

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

### Rust crates (Tauri shell)

Fetched by Cargo during `npx tauri build`:

| Crate | Purpose |
|---|---|
| tauri | Desktop shell |
| tauri-plugin-dialog | Native file dialogs |
| tauri-plugin-fs | Native filesystem access |
| ureq | Python server health checks |
| serde + serde_json | Serialization |

## Quick Start

### Datasets

Extract frames from video files, apply a configurable degradation pipeline (blur, noise, compression, downscaling), and produce paired HR/LR training datasets.

```
srengine dataset build --input video.mp4 --out ./datasets/my_set
```

Additional operations: validation, health checks (corrupt image detection, black frame pruning), and merging multiple datasets grouped by scale factor.

### Training

Train RRDB (ESRGAN-style) or SwinIR models with configurable hyperparameters, mixed-precision (bf16/fp16), cosine LR scheduling with warmup, validation split, and live JSONL metrics streaming.

```
srengine train run --model rrdb_esrgan --dataset ./datasets/my_set
```

### Inference

Run trained models on images or videos with VRAM-safe tiled processing. Supports ONNX, TorchScript, and SafeTensors export.

```
srengine infer run --model checkpoints/model.pth --input input.png --output output.png
```

### API Server

Start the FastAPI REST API on port 8765, required for the desktop GUI. Provides SSE-based real-time progress, background job management, and comprehensive endpoints for all operations.

```
srengine serve start
```

### Desktop GUI

The React/TypeScript frontend with Tauri 2 desktop shell provides tab-based navigation for project management, dataset operations, model training with live metrics, drag-and-drop inference, and checkpoint browsing.

```bash
cd frontend
npm install               # install frontend dependencies
npm run dev               # Vite dev server (requires API on :8765)
npx tauri build           # standalone desktop binary
```

### Python API

Use sr-engine programmatically from your own Python scripts:

```python
from sr_engine.models.registry import build_model
from sr_engine.data.datasets import PairedImageFolderDataset
from sr_engine.engine.inference import infer_image

model = build_model("swinir", {"name": "swinir", "scale": 4, "embed_dim": 180})
dataset = PairedImageFolderDataset("datasets/my_set")
infer_image(model_checkpoint="checkpoints/model.pth", input_path="input.png",
            output_path="output.png", device="cuda")
```

Full CLI reference is in [`docs/cli-reference.md`](docs/cli-reference.md). All commands support `--help`.

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
