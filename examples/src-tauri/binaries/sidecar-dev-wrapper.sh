#!/usr/bin/env bash
# SR Tuner sidecar dev wrapper — delegates to `uv run python -m sidecar.main`
# Installed by scripts/build.sh dev as sidecar-{triple} in src-tauri/binaries/
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIDECAR_DIR="$(cd "$SCRIPT_DIR/../../sidecar" && pwd)"
cd "$SIDECAR_DIR"
exec uv run python -m sidecar.main "$@"