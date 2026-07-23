#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# SR Engine Sidecar Builder
#
# Packages the Python backend (src/sr_engine) into a standalone PyInstaller
# binary for bundling into the Tauri desktop app via externalBin.
#
# Layout assumed:
#   envs/build.sh | envs/build.ps1   -> build the dev .venv (root of project)
#   scripts/build-sidecar.sh (this)  -> package that .venv into a sidecar
#   scripts/sidecar_entry.py         -> PyInstaller entrypoint
#
# IMPORTANT: this script never installs anything into your dev .venv.
# It hardlinks .venv into a scratch build venv, installs PyInstaller there,
# builds, then deletes the scratch venv. Your dev environment stays untouched.
#
# Usage:
#   ./scripts/build-sidecar.sh                  # auto-detect backend from .venv
#   ./scripts/build-sidecar.sh --backend cpu
#   ./scripts/build-sidecar.sh --backend cuda
#   ./scripts/build-sidecar.sh --backend rocm
#
# Options:
#   --backend cpu|cuda|rocm   Force a backend (default: auto-detect from .venv)
#   --keep-tmp                Don't delete the scratch build venv (debugging)
#   --force                   Continue even if --backend disagrees with .venv
###############################################################################

################################################################################
# Configuration / globals
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
BINARIES_DIR="$PROJECT_DIR/src-tauri/binaries"

BACKEND=""
KEEP_TMP=false
FORCE=false

BUILD_VENV=""
BUILD_MODE="--onedir"
EXTRA_BINARIES=()
EXTRA_DATA=()
ROCM_NEEDS_ALIAS=false
ROCM_NEEDED_SONAME=""
ROCM_FALLBACK_NAME=""

################################################################################
# Pretty printing
################################################################################

info()    { printf "\n==> %s\n" "$*"; }
success() { printf "\n✓ %s\n" "$*"; }
warn()    { printf "\n⚠ %s\n" "$*"; }
die()     { printf "\n✗ %s\n" "$*" >&2; exit 1; }

usage() {
cat <<EOF
Usage:
    $0 [--backend cpu|cuda|rocm] [--keep-tmp] [--force]

Options
    --backend    Force torch backend. Default: auto-detect from .venv
    --keep-tmp   Keep the scratch build venv after finishing (for debugging)
    --force      Build even if --backend disagrees with what's installed in .venv

Examples
    $0                            # auto-detect from .venv
    $0 --backend cuda
    $0 --backend rocm --keep-tmp
EOF
exit 1
}

################################################################################
# Cleanup
################################################################################

cleanup() {
    local status=$?
    if [[ -n "$BUILD_VENV" && -d "$BUILD_VENV" ]]; then
        if $KEEP_TMP; then
            warn "Keeping scratch build venv for inspection: $BUILD_VENV"
        else
            rm -rf "$(dirname "$BUILD_VENV")"
        fi
    fi
    if [[ $status -ne 0 ]]; then
        printf "\n✗ Sidecar build failed (exit %s)\n" "$status" >&2
    fi
}
trap cleanup EXIT

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
            --keep-tmp) KEEP_TMP=true; shift ;;
            --force)    FORCE=true; shift ;;
            -h|--help)  usage ;;
            *)          usage ;;
        esac
    done

    if [[ -n "$BACKEND" ]]; then
        case "$BACKEND" in
            cpu|cuda|rocm) ;;
            *) die "Unknown backend: $BACKEND (expected cpu|cuda|rocm)" ;;
        esac
    fi
}

################################################################################
# Requirements
################################################################################

check_requirements() {
    command -v uv >/dev/null || die "uv is not installed. https://github.com/astral-sh/uv"

    [[ -d "$VENV_DIR" ]] \
        || die "No .venv found at $VENV_DIR — run envs/build.sh --backend <cpu|cuda|rocm> first."

    [[ -x "$VENV_DIR/bin/python" ]] \
        || die ".venv looks broken (no bin/python) — rebuild it with envs/build.sh."

    local site_packages
    site_packages="$(find "$VENV_DIR/lib" -maxdepth 1 -type d -name 'python3.*' 2>/dev/null | head -n1)"
    [[ -n "$site_packages" ]] || die "Could not locate site-packages under $VENV_DIR/lib — .venv looks broken."
    [[ -d "$site_packages/site-packages/torch" ]] \
        || die "torch is not installed in .venv — run envs/build.sh first."

    [[ -f "$SCRIPT_DIR/sidecar_entry.py" ]] \
        || die "sidecar_entry.py not found in $SCRIPT_DIR"
}

################################################################################
# Detect / validate backend from the dev .venv
################################################################################

detect_backend() {
    info "Detecting installed torch backend in .venv..."

    local detected
    detected="$(cd "$PROJECT_DIR" && uv run --no-sync python - <<'PY'
import torch
if getattr(torch.version, "hip", None):
    print("rocm")
elif getattr(torch.version, "cuda", None):
    print("cuda")
else:
    print("cpu")
PY
)" || die "Failed to inspect torch in .venv. Is it installed correctly? (envs/build.sh)"

    echo "Detected in .venv: $detected"

    if [[ -z "$BACKEND" ]]; then
        BACKEND="$detected"
    elif [[ "$BACKEND" != "$detected" ]]; then
        if $FORCE; then
            warn "--backend=$BACKEND but .venv has '$detected' torch installed. Continuing anyway (--force)."
        else
            die "--backend=$BACKEND but .venv has '$detected' torch installed. Rebuild with envs/build.sh --backend $BACKEND, or pass --force."
        fi
    fi

    echo "Building for: $BACKEND"
}

################################################################################
# Scratch build venv (hardlink copy of the dev .venv)
################################################################################

create_build_venv() {
    info "Hardlinking dev .venv → scratch build venv (dev .venv is left untouched)..."

    local build_dir="$PROJECT_DIR/build"
    mkdir -p "$build_dir"

    local tmp_parent
    tmp_parent="$(mktemp -d "$build_dir/.venv-sidecar.XXXXXX")" \
        || die "Failed to create scratch directory in $build_dir."

    BUILD_VENV="$tmp_parent/venv"

    "$VENV_DIR/bin/python" - "$VENV_DIR" "$BUILD_VENV" <<'PY' || die "Failed to hardlink .venv into scratch directory."
import os, sys, shutil

src = sys.argv[1]
dst = sys.argv[2]

def hardlink_copy(src_path, dst_path, *args, **kwargs):
    try:
        os.link(src_path, dst_path)
    except OSError:
        shutil.copy2(src_path, dst_path)

shutil.copytree(src, dst, symlinks=True, copy_function=hardlink_copy)
PY

    [[ -x "$BUILD_VENV/bin/python" ]] || die "Scratch venv hardlink copy is broken (no bin/python)."

    success "Scratch venv ready: $BUILD_VENV"
}

install_pyinstaller() {
    info "Installing PyInstaller into scratch venv..."
    uv pip install --python "$BUILD_VENV/bin/python" pyinstaller \
        || die "Failed to install PyInstaller into scratch venv."
}

################################################################################
# ROCm-specific extras
################################################################################

setup_rocm_extras() {
    [[ "$BACKEND" == "rocm" ]] || return 0

    export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    local site_packages torch_hip_lib needed_soname candidate
    site_packages="$(find "$BUILD_VENV/lib" -maxdepth 1 -type d -name 'python3.*' | head -n1)/site-packages" || true
    torch_hip_lib="$(find "$site_packages/torch/lib" -maxdepth 1 -name 'libtorch_hip.so' 2>/dev/null | head -n1)" || true

    if [[ -z "$torch_hip_lib" ]]; then
        warn "Could not find torch/lib/libtorch_hip.so in .venv — is this really a ROCm torch build? Skipping HIP runtime bundling."
        return 0
    fi

    # Ask the ACTUAL torch build what libamdhip64 SONAME it links against,
    # instead of guessing the newest lib on the system. This is what exposed
    # a rocm6.2-vs-rocm7.2 mismatch here previously.
    needed_soname="$(ldd "$torch_hip_lib" 2>/dev/null | awk '/libamdhip64\.so/ {print $1; exit}')" || true
    if [[ -z "$needed_soname" ]]; then
        warn "Could not determine the required libamdhip64 SONAME from $torch_hip_lib. Skipping HIP runtime bundling."
        return 0
    fi
    echo "torch requires: $needed_soname"

    candidate="/opt/rocm/lib/${needed_soname}"
    if [[ -e "$candidate" ]]; then
        # Exact match — the installed ROCm runtime matches what torch was
        # built against. Bundle it under its real name, no aliasing needed.
        EXTRA_BINARIES+=(--add-binary "${candidate}:torch/lib")
        echo "Bundling matching ROCm HIP runtime: $candidate"
    else
        local fallback
        fallback="$(find /opt/rocm/lib -maxdepth 1 -name 'libamdhip64.so*' 2>/dev/null | sort -V | tail -n1)" || true
        [[ -n "$fallback" ]] \
            || die "torch needs $needed_soname but no libamdhip64.so* exists under /opt/rocm/lib at all."

        warn "torch needs ${needed_soname} but only $(basename "$fallback") is installed."
        warn "Bundling it as a compat alias — ROCm major versions are NOT guaranteed ABI-compatible"
        warn "(e.g. hipDeviceProp_t layout changed between HIP 6.x and 7.x), so this may crash at"
        warn "runtime. Fix properly by installing a ROCm runtime matching ${needed_soname}, or by"
        warn "rebuilding .venv (envs/build.sh) against the ROCm version actually on this system."

        EXTRA_BINARIES+=(--add-binary "${fallback}:torch/lib")
        ROCM_NEEDS_ALIAS=true
        ROCM_NEEDED_SONAME="$needed_soname"
        ROCM_FALLBACK_NAME="$(basename "$fallback")"
    fi

    local amdgpu_ids="$site_packages/torch/share/libdrm/amdgpu.ids"
    if [[ -f "$amdgpu_ids" ]]; then
        EXTRA_DATA+=(--add-data "$amdgpu_ids:torch/share/libdrm")
    else
        warn "amdgpu.ids not found — AMD GPU auto-detection in the packaged binary may be limited."
    fi
}

################################################################################
# Run PyInstaller
################################################################################

run_pyinstaller() {
    info "Running PyInstaller (${BUILD_MODE}, backend=${BACKEND})..."

    cd "$PROJECT_DIR"

    "$BUILD_VENV/bin/python" -m PyInstaller "$BUILD_MODE" -y \
        --name sr-engine \
        --add-data "$PROJECT_DIR/src/sr_engine/utils/configs:sr_engine/utils/configs" \
        "${EXTRA_DATA[@]}" \
        "${EXTRA_BINARIES[@]}" \
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
        "$SCRIPT_DIR/sidecar_entry.py" \
        || die "PyInstaller build failed. Re-run with --keep-tmp to inspect the scratch venv."
}

################################################################################
# Target triple resolution
################################################################################

resolve_triple() {
    if command -v rustc >/dev/null 2>&1; then
        rustc -vV | sed -n 's/^host: *//p'
        return
    fi

    warn "rustc not found — falling back to uname-based triple guess."
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"
    case "$os" in
        Linux)  echo "${arch}-unknown-linux-gnu" ;;
        Darwin) echo "${arch}-apple-darwin" ;;
        *) die "Cannot resolve target triple: rustc not found and OS '$os' is unrecognized." ;;
    esac
}

################################################################################
# Fix unresolved soname aliases in the bundled output
# PyInstaller bundles the exact files but may miss soname symlinks
# (e.g., torchvision needs libamdhip64.so.6 but only libamdhip64.so exists)
################################################################################

fix_rocm_sonames() {
    local lib_dir="$1"
    local fixed=0
    while IFS= read -r -d '' sofile; do
        local dir
        dir="$(dirname "$sofile")"
        while IFS= read -r soname; do
            [[ -z "$soname" ]] && continue
            # Skip standard system libs
            case "$soname" in
                libc.so*|libm.so*|libpthread*|librt.so*|libdl.so*|libutil.so*|libstdc++.so*|libgcc_s.so*|libz.so*|libresolv.so*|libnss_*|libBrokenLocale*|libanl.so*|libcrypt.so*|ld-linux*|ld64.so*)
                    continue ;;
            esac
            if ! find "$lib_dir" -name "$soname" -print -quit | grep -q .; then
                local base="${soname%%.so*}.so"
                local candidate target_dir
                candidate="$(find "$lib_dir" -name "${base}*" -not -name "$soname" -print -quit 2>/dev/null)"
                if [[ -n "$candidate" && ! -L "$(dirname "$candidate")/$soname" ]]; then
                    target_dir="$(dirname "$candidate")"
                    ln -sf "$(basename "$candidate")" "$target_dir/$soname"
                    warn "  Soname alias: $soname → $(basename "$candidate")"
                    fixed=$((fixed + 1))
                fi
            fi
        done < <(readelf -d "$sofile" 2>/dev/null | awk '/NEEDED/{print $NF}' | tr -d '[]' || true)
    done < <(find "$lib_dir" -name '*.so*' -type f -print0 2>/dev/null || true)
    if [[ $fixed -gt 0 ]]; then
        warn "Created $fixed soname aliases for ROCm library compatibility"
    fi
}

################################################################################
# Install result into src-tauri/binaries/
################################################################################

package_output() {
    local triple
    triple="$(resolve_triple)"
    echo "Target triple: ${triple}"

    mkdir -p "$BINARIES_DIR"
    rm -rf "$BINARIES_DIR/sr-engine-${triple}" "$BINARIES_DIR/sr-engine-${triple}.exe"

    local out_size
    if [[ "$BUILD_MODE" == "--onedir" ]]; then
        [[ -d "$PROJECT_DIR/dist/sr-engine" ]] || die "Expected PyInstaller output dir not found: dist/sr-engine"

        cp -r "$PROJECT_DIR/dist/sr-engine" "$BINARIES_DIR/sr-engine-${triple}"

        # PyInstaller's --add-binary doesn't support renaming at the destination.
        # Only relevant if setup_rocm_extras had to fall back to a mismatched
        # ROCm version (see the warning it printed during the build).
        if [[ "$BACKEND" == "rocm" && "$ROCM_NEEDS_ALIAS" == "true" ]]; then
            local torch_lib="$BINARIES_DIR/sr-engine-${triple}/_internal/torch/lib"
            if [[ -f "$torch_lib/$ROCM_FALLBACK_NAME" && ! -f "$torch_lib/$ROCM_NEEDED_SONAME" ]]; then
                ln -sf "$ROCM_FALLBACK_NAME" "$torch_lib/$ROCM_NEEDED_SONAME"
                warn "Created compat symlink: $ROCM_NEEDED_SONAME → $ROCM_FALLBACK_NAME (unverified ABI compatibility)"
            fi
        fi

        # Fix unresolved soname aliases (e.g., libamdhip64.so.6 needed by torchvision)
        fix_rocm_sonames "$BINARIES_DIR/sr-engine-${triple}/_internal"

        out_size="$(du -sh "$BINARIES_DIR/sr-engine-${triple}" | cut -f1)"
    else
        [[ -f "$PROJECT_DIR/dist/sr-engine" ]] || die "Expected PyInstaller binary not found: dist/sr-engine"

        cp "$PROJECT_DIR/dist/sr-engine" "$BINARIES_DIR/sr-engine-${triple}"
        chmod +x "$BINARIES_DIR/sr-engine-${triple}"
        out_size="$(du -h "$BINARIES_DIR/sr-engine-${triple}" | cut -f1)"
    fi

    rm -rf "$PROJECT_DIR/dist" "$PROJECT_DIR/build" "$PROJECT_DIR/sr-engine.spec"

    success "Sidecar built (${out_size}): src-tauri/binaries/sr-engine-${triple}"
}

################################################################################
# Main
################################################################################

main() {
    parse_args "$@"

    info "SR Engine Sidecar Builder"

    check_requirements
    detect_backend
    create_build_venv
    install_pyinstaller
    setup_rocm_extras
    run_pyinstaller
    package_output

    echo
    echo "Dev mode:      cargo tauri dev"
    echo "Release build: cargo tauri build"
    echo
}

main "$@"