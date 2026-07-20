#!/usr/bin/env bash
set -euo pipefail

# ── Build the Python backend as a standalone sidecar binary ──────────────
# Usage:  ./scripts/build-sidecar.sh [--backend cpu|rocm|cuda]
#
# Backend determines build mode:
#   cpu/cuda  → --onefile  (single binary, ~300 MB / ~1.5 GB)
#   rocm      → --onedir   (directory, ~3 GB, faster build & startup)
#
# Produces:  src-tauri/binaries/sr-engine-<target-triple>
# which Tauri bundles into the desktop app via externalBin.

SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SRC_DIR"

# Parse command-line arguments
BACKEND=""  # empty = auto-detect
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    *) BACKEND="$1"; shift ;;
  esac
done

# ── Auto-detect torch backend from the installed venv ────────────────────

DETECTED="cpu"
if [ -f "$SRC_DIR/.venv/lib/python3.11/site-packages/torch/lib/libtorch_hip.so" ]; then
    DETECTED="rocm"
elif [ -f "$SRC_DIR/.venv/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so" ]; then
    DETECTED="cuda"
fi

if [ -z "$BACKEND" ]; then
    BACKEND="$DETECTED"
elif [ "$BACKEND" != "$DETECTED" ]; then
    echo "  Warning: --backend=$BACKEND but venv has ${DETECTED} torch"
fi

echo "→ Building sidecar (backend=${BACKEND}, detected=${DETECTED})"

# ── ROCm-specific setup ──────────────────────────────────────────────────

EXTRA_DATA=""
EXTRA_BINARIES=""
BUILD_MODE="--onefile"

if [ "$BACKEND" = "rocm" ]; then
    export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    # Bundle the ROCm HIP runtime into the build. PyInstaller --add-binary
    # doesn't support renaming at the destination, so it lands as
    # libamdhip64.so.7. A post-build symlink (below) creates the
    # libamdhip64.so.6 alias that torch expects.
    ROCM_HIP_LIB="/opt/rocm/lib/libamdhip64.so.7"
    if [ -f "$ROCM_HIP_LIB" ]; then
        EXTRA_BINARIES="--add-binary $ROCM_HIP_LIB:torch/lib"
    else
        echo "  Warning: $ROCM_HIP_LIB not found, GPU support may be broken"
    fi
    # Bundle AMD GPU ID metadata for GPU detection
    TORCH_SHARE="$SRC_DIR/.venv/lib/python3.11/site-packages/torch/share"
    if [ -f "$TORCH_SHARE/libdrm/amdgpu.ids" ]; then
        EXTRA_DATA="--add-data $TORCH_SHARE/libdrm/amdgpu.ids:torch/share/libdrm"
    fi
    # ROCm is huge — --onedir avoids the slow serial archive step and gives
    # instant startup (no extraction to tmpdir on every launch)
    BUILD_MODE="--onedir"
fi

# ── Ensure PyInstaller is available ──────────────────────────────────────

if ! uv run pyinstaller --version &>/dev/null; then
    echo "  Installing PyInstaller …"
    uv pip install pyinstaller
fi

# ── Ensure the .venv exists ──────────────────────────────────────────────

if [ ! -d .venv ]; then
    echo "  No .venv found – run envs/build.sh first"
    exit 1
fi

# ── Run PyInstaller ──────────────────────────────────────────────────────

echo "  Running PyInstaller (${BUILD_MODE}) …"
# shellcheck disable=SC2086
uv run pyinstaller $BUILD_MODE \
    --name sr-engine \
    --add-data "$SRC_DIR/src/sr_engine/utils/configs:sr_engine/utils/configs" \
    $EXTRA_DATA \
    $EXTRA_BINARIES \
    --hidden-import uvicorn \
    --hidden-import uvicorn.logging \
    --hidden-import uvicorn.loops.auto \
    --hidden-import uvicorn.protocols.http.auto \
    --hidden-import uvicorn.middleware.wsgi \
    --hidden-import starlette \
    --hidden-import starlette.applications \
    --hidden-import sr_engine.api.app \
    --hidden-import sr_engine.api.routes \
    --hidden-import sr_engine.models.archs \
    --hidden-import torchvision \
    --hidden-import torchvision.models \
    --hidden-import lpips \
    --hidden-import safetensors \
    --hidden-import cv2 \
    --hidden-import pydantic \
    "$SRC_DIR/scripts/sidecar_entry.py"

# ── Copy result to Tauri's binaries directory ────────────────────────────

TRIPLE=$(rustc -vV | sed -n 's/.*host: *//p')
echo "  Target triple: ${TRIPLE}"

BINARIES_DIR="$SRC_DIR/src-tauri/binaries"
mkdir -p "$BINARIES_DIR"

# Remove old sidecar for this triple (file or directory)
rm -rf "$BINARIES_DIR/sr-engine-${TRIPLE}" "$BINARIES_DIR/sr-engine-${TRIPLE}.exe"

if [ "$BUILD_MODE" = "--onedir" ]; then
    # --onedir produces dist/sr-engine/ (directory with _internal/ + binary)
    cp -r "$SRC_DIR/dist/sr-engine" "$BINARIES_DIR/sr-engine-${TRIPLE}"

    # PyInstaller's --add-binary doesn't support renaming at the destination.
    # The ROCm HIP runtime was bundled as libamdhip64.so.7, but torch expects
    # libamdhip64.so.6. Create the compat symlink in the bundled directory.
    TORCH_LIB="$BINARIES_DIR/sr-engine-${TRIPLE}/_internal/torch/lib"
    if [ "$BACKEND" = "rocm" ] && [ -f "$TORCH_LIB/libamdhip64.so.7" ] && [ ! -f "$TORCH_LIB/libamdhip64.so.6" ]; then
        ln -sf libamdhip64.so.7 "$TORCH_LIB/libamdhip64.so.6"
    fi

    SIZE=$(du -sh "$BINARIES_DIR/sr-engine-${TRIPLE}" | cut -f1)
else
    # --onefile produces dist/sr-engine (single binary)
    cp "$SRC_DIR/dist/sr-engine" "$BINARIES_DIR/sr-engine-${TRIPLE}"
    SIZE=$(du -h "$BINARIES_DIR/sr-engine-${TRIPLE}" | cut -f1)
fi

# Clean up build artifacts
rm -rf "$SRC_DIR/dist" "$SRC_DIR/build" "$SRC_DIR/sr-engine.spec"

echo "✓ Sidecar built (${SIZE}): src-tauri/binaries/sr-engine-${TRIPLE}"
echo "  Run with:  cargo tauri dev"
echo "  Release:   cargo tauri build"