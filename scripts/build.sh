#!/usr/bin/env bash
#
# SR Tuner Build Orchestrator
#
# Usage:
#   ./scripts/build.sh [command] [options]
#
# Commands:
#   (default)   Build frontend + Tauri app
#   all         Clean + build everything (including sidecar)
#   frontend    Build frontend only
#   tauri       Build Tauri app only (frontend must be pre-built)
#   sidecar     Build Python sidecar
#   dev         Start development server (Vite + Tauri hot-reload)
#   clean       Delete all build artifacts
#   rebuild     Clean then build (frontend + Tauri, no sidecar)
#   check       Verify all build prerequisites
#   help        Show this help
#
# Options:
#   cpu | cuda | rocm   Sidecar variant for 'sidecar' / 'all' (default: auto-detect from .venv)
#   -j, --parallel      Build frontend and sidecar concurrently
#
# Examples:
#   ./scripts/build.sh                   # frontend + Tauri
#   ./scripts/build.sh dev               # hot-reload dev mode
#   ./scripts/build.sh check             # verify prerequisites
#   ./scripts/build.sh all               # clean + all (cpu sidecar)
#   ./scripts/build.sh all rocm -j       # clean + all ROCm, parallel
#   ./scripts/build.sh sidecar cuda      # CUDA sidecar only
#   ./scripts/build.sh rebuild           # clean + frontend + tauri
#   ./scripts/build.sh clean             # remove all artifacts

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$PROJECT_DIR/frontend"
SRC_TAURI_DIR="$PROJECT_DIR/src-tauri"

# ── Platform detection ────────────────────────────────────────────────────────
PLATFORM="linux"
[[ "$(uname -s)" == "Darwin" ]] && PLATFORM="macos"

# ── Color helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()    { echo -e "${GREEN}  ✓${NC} $1"; }
log_warn()    { echo -e "${YELLOW}  ⚠${NC} $1"; }
log_error()   { echo -e "${RED}  ✗${NC} $1" >&2; }
log_section() { echo -e "\n${BOLD}${BLUE}▶ $1${NC}"; }
log_step()    { echo -e "${CYAN}  →${NC} $1"; }

# ── Help ──────────────────────────────────────────────────────────────────────
show_help() {
    cat <<EOF

${BOLD}SR Tuner Build Script${NC} — ${PLATFORM}

${BOLD}Usage:${NC}  ./scripts/build.sh [command] [options]

${BOLD}Commands:${NC}
  (default)   Build frontend + Tauri app
  all         Clean + build everything (including sidecar)
  frontend    Build frontend only
  tauri       Build Tauri app only (frontend must be pre-built)
  sidecar     Build Python sidecar
  dev         Start development server (Vite + Tauri hot-reload)
  clean       Delete all build artifacts
  rebuild     Clean then build (frontend + Tauri, no sidecar)
  check       Verify all build prerequisites
  help        Show this help

${BOLD}Options:${NC}
  cpu | cuda | rocm   Sidecar variant for 'sidecar' / 'all' (default: auto-detect from .venv)
  -j, --parallel      Build frontend and sidecar concurrently

${BOLD}Examples:${NC}
  ./scripts/build.sh                   # frontend + Tauri
  ./scripts/build.sh dev               # hot-reload dev mode
  ./scripts/build.sh check             # verify prerequisites
  ./scripts/build.sh all               # clean + all (cpu sidecar)
  ./scripts/build.sh all rocm -j       # clean + all ROCm, parallel
  ./scripts/build.sh sidecar cuda      # CUDA sidecar only
  ./scripts/build.sh rebuild           # clean + frontend + tauri
  ./scripts/build.sh clean             # remove all artifacts

EOF
}

# ── Prerequisite check ────────────────────────────────────────────────────────
check_prereqs() {
    log_section "Checking prerequisites"
    local failed=false

    # Rust / Cargo
    if command -v cargo &>/dev/null; then
        log_info "cargo $(cargo --version 2>/dev/null | cut -d' ' -f2)"
    else
        log_error "cargo not found — install Rust from https://rustup.rs/"
        failed=true
    fi

    # Tauri CLI (cargo subcommand)
    if cargo tauri --version &>/dev/null 2>&1; then
        log_info "cargo-tauri $(cargo tauri --version 2>/dev/null)"
    else
        log_warn "cargo-tauri not found — attempting install via cargo..."
        if cargo install tauri-cli --version "^2"; then
            log_info "cargo-tauri installed"
        else
            log_error "Failed to install cargo-tauri"
            failed=true
        fi
    fi

    # Node.js / npm
    if command -v node &>/dev/null; then
        log_info "node $(node --version)"
    else
        log_error "node not found — install Node.js from https://nodejs.org/"
        failed=true
    fi

    if command -v npm &>/dev/null; then
        log_info "npm $(npm --version)"
    else
        log_error "npm not found"
        failed=true
    fi

    # uv (Python package manager)
    if command -v uv &>/dev/null; then
        log_info "uv $(uv --version 2>/dev/null | cut -d' ' -f2)"
    else
        log_error "uv not found — https://github.com/astral-sh/uv"
        failed=true
    fi

    # Python venv
    if [ -d "$PROJECT_DIR/.venv" ]; then
        log_info "Python virtual environment (.venv)"
    else
        log_warn "No .venv found — run: envs/build.sh --backend cpu"
    fi

    # Linux system libraries
    if [[ "$PLATFORM" == "linux" ]]; then
        log_step "Checking Linux system libraries..."

        local pkg_ok=true
        if command -v pkg-config &>/dev/null; then
            for lib in webkit2gtk-4.1 openssl; do
                if pkg-config --exists "$lib" 2>/dev/null; then
                    log_info "$lib $(pkg-config --modversion "$lib" 2>/dev/null)"
                else
                    log_warn "$lib not found"
                    pkg_ok=false
                fi
            done
        else
            log_warn "pkg-config not found — cannot verify system libraries"
        fi

        if ! $pkg_ok; then
            log_warn "Missing system libraries. Install on Arch:"
            log_warn "  sudo pacman -S webkit2gtk-4.1 openssl"
            log_warn "Install on Debian/Ubuntu:"
            log_warn "  sudo apt install libwebkit2gtk-4.1-dev libssl-dev"
            log_warn "Install on Fedora:"
            log_warn "  sudo dnf install webkit2gtk4.1-devel openssl-devel"
        fi

        if command -v patchelf &>/dev/null; then
            log_info "patchelf (AppImage support)"
        else
            log_warn "patchelf not found — AppImage bundles may fail"
        fi
    fi

    # macOS system requirements
    if [[ "$PLATFORM" == "macos" ]]; then
        if xcode-select -p &>/dev/null 2>&1; then
            log_info "Xcode CLI tools: $(xcode-select -p)"
        else
            log_error "Xcode Command Line Tools not found — run: xcode-select --install"
            failed=true
        fi
    fi

    echo ""
    if $failed; then
        log_error "Prerequisites check failed. Install the above and retry."
        exit 1
    fi
    log_info "All prerequisites satisfied!"
}

# ── Build Frontend ────────────────────────────────────────────────────────────
build_frontend() {
    log_section "Building Frontend"
    log_step "Installing npm dependencies..."
    cd "$FRONTEND_DIR"
    npm install
    log_step "Compiling TypeScript + bundling with Vite..."
    npm run build
    cd "$PROJECT_DIR"
    log_info "Frontend ready → frontend/dist/"
}

# ── Build Tauri App ───────────────────────────────────────────────────────────
build_tauri() {
    log_section "Building Tauri App"
    log_step "Compiling Rust + creating platform bundles..."
    cd "$PROJECT_DIR"
    cargo tauri build

    echo ""
    local bundle_dir="$PROJECT_DIR/src-tauri/target/release/bundle"
    log_info "Build complete! Output:"
    if [[ "$PLATFORM" == "linux" ]]; then
        echo "    AppImage → $bundle_dir/appimage/"
        echo "    .deb     → $bundle_dir/deb/"
        echo "    .rpm     → $bundle_dir/rpm/"
        echo "    Binary   → $PROJECT_DIR/src-tauri/target/release/sr-tuner"
    else
        echo "    .dmg     → $bundle_dir/dmg/"
        echo "    .app     → $bundle_dir/macos/"
    fi
}

# ── Build Sidecar (delegates to build-sidecar.sh) ─────────────────────────────
build_sidecar() {
    # Empty variant = not explicitly requested -> let build-sidecar.sh
    # auto-detect the backend from .venv instead of assuming cpu.
    local variant="${1:-}"

    log_section "Building Sidecar ${variant:+($variant)}"

    if [ ! -d "$PROJECT_DIR/.venv" ]; then
        log_error "No .venv found — run envs/build.sh --backend <cpu|cuda|rocm> first"
        exit 1
    fi

    log_step "Running PyInstaller via build-sidecar.sh..."
    if [ -n "$variant" ]; then
        "$SCRIPT_DIR/build-sidecar.sh" --backend "$variant"
    else
        "$SCRIPT_DIR/build-sidecar.sh"
    fi
}

# ── Development mode ──────────────────────────────────────────────────────────
dev_mode() {
    log_section "Starting Development Server"
    log_step "Starting Vite dev server + Tauri (hot-reload)..."
    log_warn "Press Ctrl+C to stop."
    echo ""
    cd "$PROJECT_DIR"
    cargo tauri dev
}

# ── Clean ─────────────────────────────────────────────────────────────────────
clean() {
    log_section "Cleaning Build Artifacts"
    rm -rf "$FRONTEND_DIR/dist"
    rm -rf "$PROJECT_DIR/dist"
    rm -rf "$PROJECT_DIR/build"
    rm -rf "$PROJECT_DIR/sr-engine.spec"
    rm -rf "$SRC_TAURI_DIR/binaries"
    mkdir -p "$SRC_TAURI_DIR/binaries"
    log_info "Clean complete."
}

# ── Parallel build (frontend + sidecar concurrently, then Tauri) ──────────────
build_parallel() {
    local variant="${1:-}"

    log_section "Parallel Build: Frontend + Sidecar ${variant:+($variant)}"
    log_step "Starting both builds concurrently..."
    echo ""

    build_frontend       > /tmp/sr-tuner-frontend.log 2>&1 &
    local frontend_pid=$!

    build_sidecar "$variant" > /tmp/sr-tuner-sidecar.log 2>&1 &
    local sidecar_pid=$!

    local frontend_done=false sidecar_done=false frontend_ok=true sidecar_ok=true

    while ! $frontend_done || ! $sidecar_done; do
        if ! $frontend_done && ! kill -0 "$frontend_pid" 2>/dev/null; then
            frontend_done=true
            if wait "$frontend_pid"; then
                log_info "[frontend] Done"
                frontend_ok=true
            else
                frontend_ok=false
                log_error "[frontend] FAILED"
            fi
        fi
        if ! $sidecar_done && ! kill -0 "$sidecar_pid" 2>/dev/null; then
            sidecar_done=true
            if wait "$sidecar_pid"; then
                log_info "[sidecar]  Done"
                sidecar_ok=true
            else
                sidecar_ok=false
                log_error "[sidecar] FAILED"
            fi
        fi
        sleep 0.5
    done

    if ! $frontend_ok; then
        log_error "Frontend build failed. Log:"
        cat /tmp/sr-tuner-frontend.log
        exit 1
    fi
    if ! $sidecar_ok; then
        log_error "Sidecar build failed. Log:"
        cat /tmp/sr-tuner-sidecar.log
        exit 1
    fi

    echo ""
    build_tauri
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    local command="${1:-}"
    local parallel=false
    # Empty = not explicitly requested; build_sidecar/build_parallel treat
    # this as "auto-detect backend from .venv" rather than assuming cpu.
    local variant=""

    # Shift past the command, parse remaining flags
    [[ $# -gt 0 ]] && shift
    for arg in "$@"; do
        case "$arg" in
            -j|--parallel)  parallel=true  ;;
            cpu|cuda|rocm)  variant="$arg" ;;
            *)
                log_error "Unknown option: $arg"
                show_help
                exit 1
                ;;
        esac
    done

    case "$command" in
        help|--help|-h) show_help ;;
        check)     check_prereqs ;;
        dev)       dev_mode ;;
        clean)     clean ;;
        frontend)  build_frontend ;;
        tauri)     build_tauri ;;
        sidecar)   build_sidecar "$variant" ;;
        all)
            clean
            if $parallel; then
                build_parallel "$variant"
            else
                build_frontend
                build_sidecar "$variant"
                build_tauri
            fi
            ;;
        rebuild)
            clean
            build_frontend
            build_tauri
            ;;
        "")
            build_frontend
            build_tauri
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"