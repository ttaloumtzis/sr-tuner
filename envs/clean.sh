#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "==> Cleaning build artifacts..."

rm -rf AppDir AppDirsrc-tauri appimage-build
rm -rf build dist *.egg-info
rm -rf frontend/node_modules frontend/dist
rm -rf src-tauri/target
rm -f SR_Tuner-*.AppImage.zsync

echo "Keeping AppImage(s):"
ls -lh SR_Tuner-*.AppImage 2>/dev/null || echo "(none)"
