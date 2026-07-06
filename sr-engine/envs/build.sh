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
    cuda)   INDEX_URL="https://download.pytorch.org/whl/cu121" ;;
    rocm)   INDEX_URL="https://download.pytorch.org/whl/rocm6.2" ;;
    cpu)    INDEX_URL="https://download.pytorch.org/whl/cpu" ;;
    *)      echo "Error: backend must be cuda, rocm, or cpu"; exit 1 ;;
esac

echo "==> Creating virtual environment..."
uv venv

echo "==> Installing with --extra $BACKEND (index: $INDEX_URL)..."
uv sync --extra "$BACKEND" --extra-index-url "$INDEX_URL"

echo "==> Running environment verification..."
uv run python "$SCRIPT_DIR/verify_env.py"

echo "==> Build complete."
