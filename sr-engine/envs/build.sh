#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --backend {cuda|rocm|cpu}"
    exit 1
}

BACKEND=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) BACKEND="$2"; shift 2 ;;
        *) usage ;;
    esac
done

if [[ -z "$BACKEND" ]]; then
    usage
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

case "$BACKEND" in
    cuda|rocm|cpu) ;;
    *) echo "Error: backend must be cuda, rocm, or cpu"; exit 1 ;;
esac

echo "==> Checking for existing build lock caches..."
if [ -d ".venv" ]; then
    echo "    ♻️ Found existing '.venv' directory. Purging for clean workspace rebuild..."
    rm -rf .venv
fi

if [ -f "uv.lock" ]; then
    echo "    ♻️ Found matching 'uv.lock' snapshot footprint. Dropping to force pristine indexing resolution..."
    rm -f uv.lock
fi

echo "==> Ensuring clean virtual environment structure..."
uv venv

echo "==> Syncing hardware workspace environment targeting configuration extra: [$BACKEND]..."
# 💡 Native and clean execution:
uv sync --extra "$BACKEND"

echo "==> Running environment verification metrics..."
uv run python "$SCRIPT_DIR/verify_env.py"

echo "==> Build complete."