# sr-engine

Super-resolution engine for training and running deep learning models on images and video. NVIDIA (CUDA) and AMD (ROCm) GPUs, unified CLI, FastAPI REST API, Tauri desktop GUI, model versioning, configurable degradation pipelines.

## Features

- **Dataset building** — extract frames from video (with OpenCV compatibility validation), apply a configurable degradation pipeline (blur, noise, JPEG/JPEG2000, color jitter, downsampling) to produce paired HR/LR training data
- **Dataset validation** — structure checks, corrupt-image detection, scale-factor alignment, black-frame pruning
- **Dataset merging** — combine multiple datasets grouped by scale factor
- **Training** — RRDB (ESRGAN-style) and SwinIR, mixed-precision (bf16/fp16), cosine LR schedule with warmup, validation split, streaming JSONL metrics
- **Inference** — image/video, VRAM-safe tiled processing
- **Export** — ONNX, TorchScript, SafeTensors
- **Model versioning** — named instances, versioned checkpoints, per-run metadata
- **Workspace management** — structured project dirs (`datasets/`, `models/`, `experiments/`, `configs/`), auto-discovered via a marker file
- **Device abstraction** — unified CUDA/ROCm detection, autocast dtype selection, flash attention support
- **REST API** — FastAPI, SSE progress streaming, background job management
- **Desktop GUI** — React/TypeScript + Tauri 2
- **CLI** — Click-based, with standalone command aliases

## Prerequisites

Only install what you actually need:

| Component | Runtime | Needed for |
|---|---|---|
| CLI + API server | Python ≥3.11, <3.13 + [uv](https://docs.astral.sh/uv/) | ML operations, dataset pipeline, REST API |
| Desktop GUI | Node.js ≥18 + npm | Frontend dev and builds |
| Desktop shell | Rust (edition 2021) | Building the native Tauri binary + bundling the Python backend (PyInstaller is installed automatically by the build script — no manual install needed) |

### 1. Install runtimes

**Python + uv**
```bash
# Linux / macOS
curl -fsSL https://astral.sh/uv/install.sh | bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Requires Python 3.11 or 3.12. Linux: `apt install python3.11 python3.11-venv`. macOS: `brew install python@3.12`. Windows: [python.org](https://www.python.org/downloads/).

**Node.js 18+**
```bash
# Linux / macOS (nvm)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
nvm install 18
```
```powershell
# Windows (nvm-windows)
winget install nvm-windows
nvm install 18 && nvm use 18
```

**Rust**
```bash
# Linux / macOS
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Windows: run [rustup-init.exe](https://rustup.rs/) or `winget install Rustlang.Rustup`.

### 2. Build the Python environment

```bash
# Linux / macOS
./envs/build.sh --backend cpu       # or --backend cuda / --backend rocm

# Windows (PowerShell)
.\envs\build.ps1 -Backend cpu       # or -Backend cuda, -Backend rocm
```

This creates `.venv` via `uv sync`, installs the matching PyTorch wheel, and verifies the install (device detection, a micro forward/backward pass).

> **ROCm users:** `--backend rocm` installs PyTorch's wheel for a specific ROCm release (currently pinned in `envs/build.sh` — check the `TORCH_INDEX` map). This must match the ROCm version installed on your system; HIP's runtime ABI is not guaranteed compatible across major versions (e.g. 6.x → 7.x). Check your installed version with `cat /opt/rocm/.info/version`, and update the pin in `envs/build.sh` if it's out of date.

### Per-platform notes

**Linux** — Tauri build requires:
```bash
sudo apt install -y build-essential libwebkit2gtk-4.1-dev librsvg2-dev patchelf libssl-dev
```

**macOS** — `brew install python@3.12 uv node rust`

**Windows:**
- All three backends (cpu, cuda, rocm) are supported. For AMD ROCm support on native Windows, see [AMD ROCm on Windows](#amd-rocm-on-windows).
- WSL is recommended: run `./envs/build.sh` inside Ubuntu on WSL for full bash compatibility. Frontend and desktop app run from Windows and connect to the API server in WSL.
- Tauri build needs [MSVC Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), WebView2 (bundled in Win10 1803+), and the Rust MSVC toolchain. Output: `.msi`/`.exe` in `src-tauri/target/release/bundle/`.
- Activate the venv with `.venv\Scripts\Activate.ps1`, not `source .venv/bin/activate`.
- `nvidia-smi` uses `,` as a decimal separator on some European-locale systems — the GPU polling code normalizes this before parsing.
- **ffmpeg must be on PATH** for frame extraction from video — `shutil.which("ffmpeg")` finds it automatically. Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin/` directory to your system PATH.

### AMD ROCm on Windows

ROCm on Windows is only available via the AMD Adrenalin driver (24.12.1+),
which bundles its own Python 3.12 and a custom PyTorch build with ROCm
support. Official PyTorch ROCm wheels from `download.pytorch.org` are
Linux-only and do not work on Windows.

Key details:

- **Python 3.12 only** — the Adrenalin tool uses Python 3.12, which is
  within the project's `>=3.11,<3.13` range.
- Create your environment with the Adrenalin GUI, then point uv at it by
  placing or symlinking the `.venv` in the project root. The build script
  automatically skips PyTorch install for `--backend rocm`:
  ```powershell
  .\envs\build.ps1 -Backend rocm
  ```
  This runs `uv sync --no-dev` for project dependencies while preserving
  the Adrenalin-provided PyTorch, then verifies ROCm is working.
- All subsequent commands (`.\scripts\build.ps1`, `srengine`, etc.) work
  identically to a CUDA or CPU setup.

## Building the desktop app

`scripts/build.sh` (Linux/macOS) or `scripts/build.ps1` (Windows) is the single entry point for frontend, sidecar, and Tauri builds.

| Command | Does |
|---|---|
| *(none)* | Build frontend + Tauri |
| `all` | Clean + build everything, including the sidecar |
| `frontend` | Frontend only |
| `tauri` | Tauri only (frontend must already be built) |
| `sidecar [cpu\|cuda\|rocm]` | Build the Python sidecar. Omit the backend to auto-detect from `.venv` |
| `dev` | Vite + Tauri hot-reload dev server |
| `clean` | Remove frontend dist, sidecar build artifacts, and stale Tauri bundle output (`.deb`/`.rpm`/`.AppImage`/`.dmg`) |
| `deep-clean` | `clean` + wipe the full Cargo `target/` (next build is slow) |
| `rebuild` | `clean` + frontend + Tauri (no sidecar) |
| `check` | Verify prerequisites |
| `help` | Show usage |

Options: `-j`/`--parallel` / `-Parallel` builds frontend and sidecar concurrently.

**Linux / macOS**
```bash
./scripts/build.sh                # frontend + Tauri
./scripts/build.sh dev            # hot-reload dev mode
./scripts/build.sh check          # verify prerequisites
./scripts/build.sh all            # clean + full build, sidecar backend auto-detected
./scripts/build.sh all rocm -j    # clean + full build, ROCm sidecar, parallel
./scripts/build.sh sidecar cuda   # CUDA sidecar only
```

**Windows (PowerShell)**
```powershell
.\scripts\build.ps1               # frontend + Tauri
.\scripts\build.ps1 dev           # hot-reload dev mode
.\scripts\build.ps1 check         # verify prerequisites
.\scripts\build.ps1 all           # clean + full build, sidecar backend auto-detected
.\scripts\build.ps1 all rocm -j   # clean + full build, ROCm sidecar, parallel
.\scripts\build.ps1 sidecar cuda  # CUDA sidecar only
```

**Dev mode** runs the Python backend via `uv run uvicorn` straight from `.venv` — Python changes are picked up on restart, no packaging step.

**Release builds** package the backend as a standalone binary via `scripts/build-sidecar.sh` (Linux/macOS) or `scripts/build-sidecar.ps1` (Windows), so the app runs on any machine without Python or `.venv` installed. The sidecar script never modifies your dev `.venv` — it copies it into a scratch directory, installs PyInstaller there, builds, and discards the copy.

### Running the built app

Output lands in `src-tauri/target/release/bundle/`.

**Linux**
```bash
# AppImage — no install needed
chmod +x "src-tauri/target/release/bundle/appimage/SR Tuner_0.1.0_amd64.AppImage"
"src-tauri/target/release/bundle/appimage/SR Tuner_0.1.0_amd64.AppImage"

# .deb
sudo dpkg -i "src-tauri/target/release/bundle/deb/SR Tuner_0.1.0_amd64.deb"
sr-tuner

# .rpm
sudo rpm -i "src-tauri/target/release/bundle/rpm/SR Tuner-0.1.0-1.x86_64.rpm"
sr-tuner
```

**macOS**
```bash
open "src-tauri/target/release/bundle/dmg/SR Tuner_0.1.0_x64.dmg"
# drag SR Tuner.app into Applications, then:
open -a "SR Tuner"
```
First launch may need a Gatekeeper bypass since the build isn't notarized: `xattr -cr "/Applications/SR Tuner.app"`, or right-click → Open.

**Windows**
```powershell
# .msi — double-click, or silently:
msiexec /i "src-tauri\target\release\bundle\msi\SR Tuner_0.1.0_x64_en-US.msi" /quiet

# or run the standalone .exe directly
& "src-tauri\target\release\SR Tuner.exe"
```

**Dev mode**, any OS:
```bash
./scripts/build.sh dev
```

## Quick Start

### Datasets
```bash
srengine dataset build --input video.mp4 --out ./datasets/my_set
```
Also: validation, health checks (corrupt-image detection, black-frame pruning), and merging datasets grouped by scale factor.

### Training
```bash
srengine train run --model rrdb_esrgan --dataset ./datasets/my_set
```

### Inference
```bash
srengine infer run --model checkpoints/model.pth --input input.png --output output.png
```

### API server
```bash
srengine serve start   # FastAPI on :8765, required by the desktop GUI
```

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

Full CLI reference: [`docs/cli-reference.md`](docs/cli-reference.md). All commands support `--help`.

## Project Structure

```
├── envs/               # Environment builder & verification scripts
├── scripts/            # Build orchestration (build.sh, build-sidecar.sh)
├── src/sr_engine/
│   ├── cli/            # Click CLI commands
│   ├── api/            # FastAPI REST API (routes, schemas, SSE, task mgmt)
│   ├── data/            # Dataset pipeline (build, validate, degrade, merge)
│   ├── device/          # CUDA/ROCm abstraction layer
│   ├── engine/          # Training loop, inference, metrics
│   ├── models/          # Model architectures (RRDB, SwinIR), checkpointing
│   └── utils/            # Config loading, I/O, logging, progress reporting
├── frontend/            # React/TypeScript desktop GUI
├── src-tauri/           # Tauri 2 Rust shell
├── tests/               # pytest suite
└── docs/                # Documentation
```

Architecture details: [`docs/architecture.md`](docs/architecture.md).

## Configuration

Defaults ship in `src/sr_engine/utils/configs/` and are copied to the workspace on `workspace init`. Override by editing the workspace copies or passing a custom YAML via `--config`.

| Config file | Contents |
|---|---|
| `configs/train/base.yaml` | Training hyperparameters (batch size, LR, epochs, loss weights, mixed precision) |
| `configs/datasets/video_pairs.yaml` | Degradation pipeline (blur, noise, JPEG, JPEG2000, color jitter, resize) |
| `configs/models/swinir.yaml` | SwinIR architecture |
| `configs/models/rrdb_esrgan.yaml` | RRDB-ESRGAN architecture |

## Supported Models

| Name | Registry key | Architecture | Scale |
|---|---|---|---|
| RRDB-ESRGAN | `rrdb_esrgan` | Residual-in-Residual Dense Blocks + nearest-neighbour upsampler | 4× (configurable) |
| SwinIR | `swinir` | Swin Transformer + pixel-shuffle upsampler | 4× (configurable) |

## Development

```bash
uv sync --group dev        # pytest, ruff
uv run pytest tests/
uv run ruff check src/
```

PyInstaller isn't a dev dependency — `scripts/build-sidecar.sh` installs it into a disposable scratch venv only when packaging a release, so it never pollutes `.venv`.

## License

MIT — see [LICENSE](LICENSE).