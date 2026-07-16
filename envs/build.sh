#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# SR Engine Environment Builder
#
# Creates a uv virtual environment
# Installs project dependencies
# Installs backend-specific PyTorch
# Verifies the installation
#
# Usage:
#   ./build.sh --backend cpu
#   ./build.sh --backend cuda
#   ./build.sh --backend rocm
#
# Optional:
#   --clean     Remove .venv and uv.lock before building
###############################################################################

################################################################################
# Configuration
################################################################################

declare -Ar TORCH_INDEX=(
    [cpu]="https://download.pytorch.org/whl/cpu"
    [cuda]="https://download.pytorch.org/whl/cu121"
    [rocm]="https://download.pytorch.org/whl/rocm6.2"
)

BACKEND=""
CLEAN=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

################################################################################
# Pretty printing
################################################################################

info() {
    printf "\n==> %s\n" "$*"
}

success() {
    printf "\n✓ %s\n" "$*"
}

warn() {
    printf "\n⚠ %s\n" "$*"
}

die() {
    printf "\n✗ %s\n" "$*" >&2
    exit 1
}

################################################################################
# Usage
################################################################################

usage() {
cat <<EOF
Usage:
    $0 --backend {cpu|cuda|rocm} [--clean]

Options

    --backend    PyTorch backend
    --clean      Remove existing environment first

Examples

    $0 --backend cpu
    $0 --backend cuda
    $0 --backend rocm --clean
EOF
exit 1
}

################################################################################
# Parse arguments
################################################################################

parse_args() {

    while [[ $# -gt 0 ]]; do
        case "$1" in

            --backend)
                [[ $# -lt 2 ]] && usage
                BACKEND="$2"
                shift 2
                ;;

            --clean)
                CLEAN=true
                shift
                ;;

            -h|--help)
                usage
                ;;

            *)
                usage
                ;;
        esac
    done

    [[ -z "$BACKEND" ]] && usage

    case "$BACKEND" in
        cpu|cuda|rocm) ;;
        *)
            die "Unknown backend: $BACKEND"
            ;;
    esac
}

################################################################################
# Requirements
################################################################################

check_requirements() {

    command -v uv >/dev/null \
        || die "uv is not installed."

    command -v python3 >/dev/null \
        || die "python3 is not installed."
}

################################################################################
# Clean
################################################################################

clean_environment() {

    if ! $CLEAN; then
        return
    fi

    info "Removing existing environment..."

    rm -rf "$PROJECT_DIR/.venv"
    rm -f "$PROJECT_DIR/uv.lock"
}

################################################################################
# Create venv
################################################################################

create_environment() {

    info "Creating virtual environment..."

    cd "$PROJECT_DIR"

    uv venv
}

################################################################################
# Install base dependencies
################################################################################

install_base() {
    info "Installing project dependencies (excluding dev group)..."
    cd "$PROJECT_DIR"

    # Use --no-dev to avoid installing the dev-only dependencies
    # (which previously included the default torch)
    uv sync --no-dev
}

################################################################################
# Retry helper
################################################################################

retry() {

    local attempts=3
    local delay=5

    for ((i=1;i<=attempts;i++)); do

        "$@" && return 0

        if [[ $i -lt $attempts ]]; then
            warn "Attempt $i failed. Retrying in ${delay}s..."
            sleep "$delay"
        fi
    done

    return 1
}

################################################################################
# Install torch
################################################################################

install_backend() {

    local index="${TORCH_INDEX[$BACKEND]}"

    info "Installing PyTorch backend: $BACKEND"
    echo "Index: $index"

    retry uv pip install \
        --index-url "$index" \
        torch torchvision

    success "PyTorch installed."
}

################################################################################
# Verify installation
################################################################################

verify_environment() {

    info "Running backend verification..."

    uv run python - "$BACKEND" <<'PY'
import sys
import torch

backend = sys.argv[1]

print()
print("=" * 60)
print("Torch Version :", torch.__version__)
print("Backend       :", backend)
print()

if backend == "cpu":

    if torch.cuda.is_available():
        raise RuntimeError("GPU backend detected while CPU backend requested.")

    print("✓ CPU backend verified")

elif backend == "cuda":

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if torch.version.cuda is None:
        raise RuntimeError("CUDA wheel not installed.")

    print("✓ CUDA backend verified")
    print("CUDA Version :", torch.version.cuda)
    print("GPU          :", torch.cuda.get_device_name(0))

elif backend == "rocm":

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/HIP unavailable.")

    if torch.version.hip is None:
        raise RuntimeError("ROCm wheel not installed.")

    print("✓ ROCm backend verified")
    print("HIP Version  :", torch.version.hip)
    print("GPU          :", torch.cuda.get_device_name(0))

print("=" * 60)
PY
}

################################################################################
# Main
################################################################################

main() {

    parse_args "$@"

    info "SR Engine Environment Builder"
    echo "Backend : $BACKEND"

    check_requirements
    clean_environment
    create_environment
    install_base
    install_backend
    verify_environment

    success "Environment successfully created."

    echo
    echo "Activate with:"
    echo
    echo "    source .venv/bin/activate"
    echo
}

main "$@"