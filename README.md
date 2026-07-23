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

## Requirements

Only install what you actually need. Dependencies that are **auto-installed by build scripts** are listed separately below.

### Manual prerequisites

| Component | Runtime | Needed for |
| --------- | ------- | ---------- |
| CLI + API server | Python ≥3.11, <3.13 + [uv](https://docs.astral.sh/uv/) | ML operations, dataset pipeline, REST API |
| CLI + API server (GPU) | NVIDIA driver ≥535 (CUDA) or ROCm 6.3 (AMD) | GPU-accelerated training and inference (see [GPU backend prerequisites](#gpu-backend-prerequisites)) |
| Desktop GUI | Node.js ≥18 + npm | Frontend dev and builds |
| Desktop shell | Rust ≥1.77 (edition 2021) via [rustup](https://rustup.rs/) | Building the native Tauri binary |
| Frame extraction | `ffmpeg` on PATH | Video frame extraction for dataset building |

---

#### Linux

| Dep | Notes |
| --- | ----- |
| Python 3.11 or 3.12 | `sudo apt install python3.12 python3.12-venv` (Ubuntu 24.04+) or `python3.11` for older distros. Or use `uv python install 3.12` |
| `uv` | [astral.sh/uv](https://docs.astral.sh/uv/) — `curl -fsSL https://astral.sh/uv/install.sh \| bash` |
| Node.js 18+ | [nvm](https://github.com/nvm-sh/nvm) recommended: `nvm install 18` |
| Rust ≥1.77 | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` then `rustup default stable` |
| Tauri system deps | See table below by distro |
| `ffmpeg` | `sudo apt install ffmpeg` / `sudo pacman -S ffmpeg` / `sudo dnf install ffmpeg` |
| `pkg-config` | Used by `build.sh check` to verify system libraries — recommended but not required to build |

| Distro | Install command |
| ------ | --------------- |
| Debian / Ubuntu | `sudo apt install -y build-essential libwebkit2gtk-4.1-dev librsvg2-dev patchelf libssl-dev fuse libayatana-appindicator3-dev` |
| Arch / CachyOS / Manjaro | `sudo pacman -S webkit2gtk-4.1 patchelf fuse2` |
| Fedora | `sudo dnf install webkit2gtk4.1-devel patchelf fuse gcc-c++` |

> **`libayatana-appindicator3-dev`** (Debian/Ubuntu only) is needed for system tray support in the Tauri app. The app compiles without it; the package is optional.

> **Rust toolchain on Linux:** If Rust was installed via a system package (`apt`/`dnf`), run `rustup default stable` to set the default toolchain. The `rustup`-based install from `rustup.rs` does this automatically.

> **AppImage bundling notes (all distros):**
> - `patchelf` is required by `linuxdeploy` to patch ELF rpaths inside the AppImage.
> - `fuse` (or `fuse2` on Arch) is needed to run `linuxdeploy` itself (it's distributed as an AppImage).
> - **Arch / CachyOS only:** Python wheels may link against hash-versioned library sonames while system packages use plain names. If `linuxdeploy` fails with `Could not find dependency: libavif-<hash>.so`, create a compat symlink (see [Commands → Linux fixes](#linux-fixes)).

---

#### macOS

| Dep | Install |
| --- | ------- |
| Xcode Command Line Tools | `xcode-select --install` |
| Python 3.12 | `brew install python@3.12` — or `uv python install 3.12` |
| `uv` | `brew install uv` — or `curl -fsSL https://astral.sh/uv/install.sh \| bash` |
| Node.js 18+ | `brew install node` — or `nvm install 18` |
| Rust ≥1.77 | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| `ffmpeg` | `brew install ffmpeg` (CLI-only use). The desktop app bundles ffmpeg automatically via PyInstaller. |

> **Apple Silicon (M1/M2/M3):** Homebrew installs to `/opt/homebrew`. PyTorch has native MPS acceleration — no CUDA/ROCm setup needed. All tools work natively on ARM64.

---

#### Windows

| Dep | Install |
| --- | ------- |
| Python 3.11 or 3.12 | [python.org](https://www.python.org/downloads/) — check "Add to PATH" during install |
| `uv` | `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| Node.js 18+ | `winget install nvm-windows` then `nvm install 18 && nvm use 18` |
| Rust (MSVC toolchain) | `winget install Rustlang.Rustup` |
| VS Build Tools | [visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/) — select "Desktop development with C++" workload |
| WebView2 | Bundled with Windows 10 1803+ and Windows 11. **Windows Server / LTSC / pre-1803:** install [manually](https://developer.microsoft.com/en-us/microsoft-edge/webview2/) |
| `ffmpeg` | Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH |

**Windows-specific setup:**

- **PowerShell execution policy:** `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` — required to run `.ps1` build scripts.
- **Long path support:** Enable to avoid MAX_PATH (260 char) issues with `uv` and PyInstaller:
  ```powershell
  New-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
  ```
  (Run as Administrator, then reboot.)
- **WSL recommended** for the backend build: run `envs/build.sh` inside Ubuntu-on-WSL for full bash compatibility. The frontend and desktop app run from Windows and connect to the API server in WSL.
- Activate the venv with `.venv\Scripts\Activate.ps1`, not `source .venv/bin/activate`.
- `nvidia-smi` uses `,` as a decimal separator on some European-locale systems — the GPU polling code normalizes this before parsing.

##### AMD ROCm on Windows

ROCm on Windows is only available via the AMD Adrenalin driver (24.12.1+), which bundles its own Python 3.12 and a custom PyTorch build with ROCm support. Official PyTorch ROCm wheels from `download.pytorch.org` are Linux-only and do not work on Windows.

- **Python 3.12 only** — the Adrenalin tool uses Python 3.12, which is within the project's `>=3.11,<3.13` range.
- Create your environment with the Adrenalin GUI, then point `uv` at it by placing or symlinking the `.venv` in the project root. The build script automatically skips the PyTorch install for `--backend rocm` and just runs `uv sync --no-dev` for project dependencies, preserving the Adrenalin-provided PyTorch.
- All subsequent commands (`build.ps1`, `srengine`, etc.) work identically to a CUDA or CPU setup.

---

### GPU backend prerequisites

The `envs/build.sh` script installs PyTorch from a backend-specific index URL. The installed PyTorch wheel must match the GPU runtime on your system.

| Backend | Driver / Runtime required | PyTorch index (from `envs/build.sh`) |
| ------- | ------------------------- | ------------------------------------ |
| `cpu` | None (runs on any system) | `https://download.pytorch.org/whl/cpu` |
| `cuda` | NVIDIA driver ≥535 (supports CUDA 12.1) | `https://download.pytorch.org/whl/cu121` |
| `rocm` | ROCm 6.3 runtime — verify with `cat /opt/rocm/.info/version` | `https://download.pytorch.org/whl/rocm6.3` |

> **ROCm on Linux:** The PyTorch ROCm wheel must match the ROCm version installed on your system. HIP's runtime ABI is not guaranteed compatible across major versions (e.g. 6.x → 7.x). If the pin in `envs/build.sh` is outdated for your ROCm version, update the `TORCH_INDEX` map in that file.

---

### Auto-installed dependencies

These are handled automatically by build scripts — no manual install needed:

| Dep | Installed by | Notes |
| --- | ------------ | ----- |
| Tauri CLI (`cargo-tauri` ^2) | `scripts/build.sh check` / `build.ps1` | Installed via `cargo install tauri-cli --version "^2"` |
| PyTorch + torchvision | `envs/build.sh` / `build.ps1` | From the backend-specific index URL |
| LPIPS | `envs/build.sh` / `build.ps1` | Perceptual loss metric |
| npm packages | `scripts/build.sh frontend` / `npm install` | React, Vite, TypeScript, Tauri API plugins |
| PyInstaller | `scripts/build-sidecar.sh` / `build-sidecar.ps1` | Installed into a disposable scratch venv, never touches your dev `.venv` |

---

### Release builds only

These are only needed when running `envs/build-release.sh` for CI or release packaging:

| Dep | Distro | Purpose |
| --- | ------ | ------- |
| `zsync` | All Linux | AppImage delta updates |
| `appimage-builder` | All Linux | AppImage packaging (installed via pip into a temp venv) |

---

### Docker alternative

Pre-built Docker images for CUDA and ROCm are available in `envs/docker/`:

- `envs/docker/Dockerfile.cuda` — based on `nvidia/cuda:12.1-runtime-ubuntu22.04`
- `envs/docker/Dockerfile.rocm` — based on `rocm/dev-ubuntu-22.04:6.2`

These handle all system dependencies and Python environment setup inside the container. Use them if you prefer not to install runtimes on your host machine.

---

## Commands

All commands below are copy-paste ready — hover a block and click the copy icon.

### Install runtimes

**Python + uv — Linux / macOS**
```bash
curl -fsSL https://astral.sh/uv/install.sh | bash
```

**Python + uv — Windows (PowerShell)**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Python 3.12 — Linux (apt)**
```bash
# Ubuntu 24.04+ / Debian testing
sudo apt install python3.12 python3.12-venv
# Older distros: use python3.11 or let uv manage Python for you
```

**Python (any version) — all platforms (uv-managed)**
```bash
uv python install 3.12
```
This avoids installing Python via a system package manager. uv downloads and manages the interpreter inside `.venv`.

**Node.js 18+ — Linux / macOS (nvm)**
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
nvm install 18
```

**Node.js 18+ — Windows (nvm-windows)**
```powershell
winget install nvm-windows
nvm install 18 && nvm use 18
```

**Rust — Linux / macOS**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# If installed via system package (apt/dnf), also run:
# rustup default stable
```

**Rust — Windows**
```powershell
winget install Rustlang.Rustup
```

**Windows — PowerShell execution policy** (first-time setup)
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Windows — enable long paths** (admin PowerShell, then reboot)
```powershell
New-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
```

### Linux system packages

```bash
# Debian / Ubuntu
sudo apt install -y build-essential libwebkit2gtk-4.1-dev librsvg2-dev patchelf libssl-dev fuse libayatana-appindicator3-dev
```
```bash
# Arch / CachyOS / Manjaro
sudo pacman -S webkit2gtk-4.1 patchelf fuse2
```
```bash
# Fedora
sudo dnf install webkit2gtk4.1-devel patchelf fuse gcc-c++
```

<a id="linux-fixes"></a>
**Arch/CachyOS AppImage fix** (only if `linuxdeploy` fails with `Could not find dependency: libavif-<hash>.so`):
```bash
sudo ln -s /usr/lib/libavif.so.16.4.2 /usr/lib/libavif-8a7f9d56.so.16.4.2
```

### macOS system packages

```bash
brew install python@3.12 uv node
# Rust is installed via rustup (see "Install runtimes" above)
```

### Build the Python environment

**Linux / macOS**
```bash
./envs/build.sh --backend cpu       # or --backend cuda / --backend rocm
```

**Windows (PowerShell)**
```powershell
.\envs\build.ps1 -Backend cpu       # or -Backend cuda, -Backend rocm
```

**Windows, AMD ROCm (Adrenalin-managed venv)**
```powershell
.\envs\build.ps1 -Backend rocm
```

This creates `.venv` via `uv sync`, installs the matching PyTorch wheel, and verifies the install (device detection, a micro forward/backward pass).

### Build the desktop app

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

| Command                   | Does                                                                                                             |
| --------------------------- | -------------------------------------------------------------------------------------------------------------- |
| *(none)*                  | Build frontend + Tauri                                                                                           |
| `all`                     | Clean + build everything, including the sidecar                                                                  |
| `frontend`                | Frontend only                                                                                                    |
| `tauri`                   | Tauri only (frontend must already be built)                                                                      |
| `sidecar [cpu\|cuda\|rocm]` | Build the Python sidecar. Omit the backend to auto-detect from `.venv`                                          |
| `dev`                     | Vite + Tauri hot-reload dev server                                                                               |
| `clean`                   | Remove frontend dist, sidecar build artifacts, and stale Tauri bundle output (`.deb`/`.rpm`/`.AppImage`/`.dmg`) |
| `deep-clean`              | `clean` + wipe the full Cargo `target/` (next build is slow)                                                    |
| `rebuild`                 | `clean` + frontend + Tauri (no sidecar)                                                                          |
| `check`                   | Verify prerequisites                                                                                              |
| `help`                    | Show usage                                                                                                       |

Options: `-j`/`--parallel` / `-Parallel` builds frontend and sidecar concurrently.

**Dev mode** runs the Python backend via `uv run uvicorn` straight from `.venv` — Python changes are picked up on restart, no packaging step.

**Release builds** package the backend as a standalone binary via `scripts/build-sidecar.sh` (Linux/macOS) or `scripts/build-sidecar.ps1` (Windows), so the app runs on any machine without Python or `.venv` installed. The sidecar script never modifies your dev `.venv` — it copies it into a scratch directory, installs PyInstaller there, builds, and discards the copy.

### Run the built app

**Linux — AppImage**
```bash
chmod +x "src-tauri/target/release/bundle/appimage/SR Tuner_0.1.0_amd64.AppImage"
"src-tauri/target/release/bundle/appimage/SR Tuner_0.1.0_amd64.AppImage"
```

**Linux — .deb**
```bash
sudo dpkg -i "src-tauri/target/release/bundle/deb/SR Tuner_0.1.0_amd64.deb"
sr-tuner
```

**Linux — .rpm**
```bash
sudo rpm -i "src-tauri/target/release/bundle/rpm/SR Tuner-0.1.0-1.x86_64.rpm"
sr-tuner
```

**macOS**
```bash
open "src-tauri/target/release/bundle/dmg/SR Tuner_0.1.0_x64.dmg"
```
```bash
# after dragging SR Tuner.app into Applications:
open -a "SR Tuner"
```
```bash
# first launch only, if Gatekeeper blocks the unnotarized build:
xattr -cr "/Applications/SR Tuner.app"
```

**Windows — .msi**
```powershell
msiexec /i "src-tauri\target\release\bundle\msi\SR Tuner_0.1.0_x64_en-US.msi" /quiet
```

**Windows — standalone .exe**
```powershell
& "src-tauri\target\release\SR Tuner.exe"
```

**Dev mode, any OS**
```bash
./scripts/build.sh dev
```

### Quick start — CLI

**Build a dataset**
```bash
srengine dataset build --input video.mp4 --out ./datasets/my_set
```

**Train a model**
```bash
srengine train run --model rrdb_esrgan --dataset ./datasets/my_set
```

**Run inference**
```bash
srengine infer run --model checkpoints/model.pth --input input.png --output output.png
```

**Start the API server**
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

### Development

```bash
uv sync --group dev        # pytest, ruff
uv run pytest tests/
uv run ruff check src/
```

PyInstaller isn't a dev dependency — `scripts/build-sidecar.sh` installs it into a disposable scratch venv only when packaging a release, so it never pollutes `.venv`.

Full CLI reference: [`docs/cli-reference.md`](https://github.com/ttaloumtzis/sr-tuner/blob/main/docs/cli-reference.md). All commands support `--help`.

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

Architecture details: [`docs/architecture.md`](https://github.com/ttaloumtzis/sr-tuner/blob/main/docs/architecture.md).

## Configuration

Defaults ship in `src/sr_engine/utils/configs/` and are copied to the workspace on `workspace init`. Override by editing the workspace copies or passing a custom YAML via `--config`.

| Config file                         | Contents                                                                          |
| ------------------------------------ | ------------------------------------------------------------------------------- |
| `configs/train/base.yaml`           | Training hyperparameters (batch size, LR, epochs, loss weights, mixed precision) |
| `configs/datasets/video_pairs.yaml` | Degradation pipeline (blur, noise, JPEG, JPEG2000, color jitter, resize)         |
| `configs/models/swinir.yaml`        | SwinIR architecture                                                              |
| `configs/models/rrdb_esrgan.yaml`   | RRDB-ESRGAN architecture                                                          |

## Supported Models

| Name        | Registry key  | Architecture                                                     | Scale              |
| ----------- | ------------- | ------------------------------------------------------------------ | -------------------- |
| RRDB-ESRGAN | `rrdb_esrgan` | Residual-in-Residual Dense Blocks + nearest-neighbour upsampler   | 4× (configurable) |
| SwinIR      | `swinir`      | Swin Transformer + pixel-shuffle upsampler                        | 4× (configurable) |

## License

MIT — see [LICENSE](https://github.com/ttaloumtzis/sr-tuner/blob/main/LICENSE).