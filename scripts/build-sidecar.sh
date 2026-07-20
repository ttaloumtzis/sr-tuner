#!/usr/bin/env bash
set -euo pipefail

# ── Build the Python backend as a standalone sidecar binary ──────────────
# Usage:  ./scripts/build-sidecar.sh [--backend cpu]
#
# Produces:  src-tauri/binaries/sr-engine-<target-triple>
# which Tauri bundles into the desktop app via externalBin.

SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SRC_DIR"

BACKEND="cpu"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    *) BACKEND="$1"; shift ;;
  esac
done

echo "→ Building sidecar for backend=${BACKEND}"

# Ensure PyInstaller is available
if ! uv run pyinstaller --version &>/dev/null; then
    echo "  Installing PyInstaller …"
    uv pip install pyinstaller
fi

# Ensure the .venv has all deps (skip if already synced)
if [ ! -d .venv ]; then
    echo "  No .venv found – run envs/build.sh first"
    exit 1
fi

# Build the sidecar binary via PyInstaller (--onefile = single self-contained binary)
# --add-data uses absolute path so it works regardless of --specpath
echo "  Running PyInstaller (--onefile) …"
uv run pyinstaller --onefile \
    --name sr-engine \
    --add-data "$SRC_DIR/src/sr_engine/utils/configs:sr_engine/utils/configs" \
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

# Determine target triple (must match what Rust/Cargo uses)
TRIPLE=$(rustc -vV | sed -n 's/.*host: *//p')
echo "  Target triple: ${TRIPLE}"

# Place the binary where Tauri's externalBin expects it
BINARIES_DIR="$SRC_DIR/src-tauri/binaries"
mkdir -p "$BINARIES_DIR"

# Remove old sidecar for this triple
rm -f "$BINARIES_DIR/sr-engine-${TRIPLE}" "$BINARIES_DIR/sr-engine-${TRIPLE}.exe"

# PyInstaller --onefile creates dist/sr-engine (the single binary)
cp "$SRC_DIR/dist/sr-engine" \
   "$BINARIES_DIR/sr-engine-${TRIPLE}"

# Clean up build artifacts
rm -rf "$SRC_DIR/dist" "$SRC_DIR/build" "$SRC_DIR/sr-engine.spec"

echo "✓ Sidecar built: src-tauri/binaries/sr-engine-${TRIPLE}"
echo "  Run with:  cargo tauri dev"
echo "  Release:   cargo tauri build"