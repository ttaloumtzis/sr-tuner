#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

detect_os() {
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_ID="${ID,,}"
    OS_ID_LIKE="${ID_LIKE,,}"
  else
    echo "✗ Cannot detect OS (/etc/os-release not found)"
    exit 1
  fi
}

install_deps() {
  case "$OS_ID" in
    arch|manjaro|endeavouros|artix|garuda|rebornos|cachyos)
      echo "==> Installing deps (pacman)..."
      sudo pacman -S --needed --noconfirm \
        rustup nodejs npm webkit2gtk-4.1 base-devel python-pip zsync 2>&1 | tail -3
      rustup default stable 2>&1 | tail -1
      ;;

    ubuntu|debian|pop|mint|elementary|zorin|kali)
      echo "==> Installing deps (apt)..."
      sudo apt-get update -qq
      sudo apt-get install -y -qq \
        build-essential curl wget file libssl-dev \
        libwebkit2gtk-4.1-dev libayatana-appindicator3-dev \
        librsvg2-dev patchelf nodejs npm python3-pip python3-venv \
        zsync 2>&1 | tail -3
      if ! command -v rustc &>/dev/null; then
        curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y -q
      fi
      ;;

    fedora)
      echo "==> Installing deps (dnf)..."
      sudo dnf install -y \
        rustup nodejs npm webkit2gtk4.1-devel \
        gcc-c++ python3-pip patchelf zsync 2>&1 | tail -3
      rustup default stable 2>&1 | tail -1
      ;;

    *)
      if [[ "$OS_ID_LIKE" == *"arch"* ]]; then
        sudo pacman -S --needed --noconfirm \
          rustup nodejs npm webkit2gtk-4.1 base-devel python-pip zsync 2>&1 | tail -3
        rustup default stable 2>&1 | tail -1
      elif [[ "$OS_ID_LIKE" == *"debian"* ]]; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq \
          build-essential curl libwebkit2gtk-4.1-dev \
          libayatana-appindicator3-dev librsvg2-dev patchelf \
          nodejs npm python3-pip python3-venv zsync 2>&1 | tail -3
        if ! command -v rustc &>/dev/null; then
          curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y -q
        fi
      elif [[ "$OS_ID_LIKE" == *"fedora"* ]]; then
        sudo dnf install -y \
          rustup nodejs npm webkit2gtk4.1-devel \
          gcc-c++ python3-pip patchelf zsync 2>&1 | tail -3
        rustup default stable 2>&1 | tail -1
      else
        echo "✗ Unsupported OS: $OS_ID (ID_LIKE=$OS_ID_LIKE)"
        exit 1
      fi
      ;;
  esac

  export PATH="$HOME/.cargo/bin:$PATH"
}

# ─── Main ────────────────────────────────────────────────────────
echo "==> SR Tuner Release Builder"
detect_os
echo "    OS: $OS_ID"
install_deps

echo "==> Installing frontend deps..."
cd "$PROJECT_DIR/frontend"
npm install 2>&1 | tail -3
cd "$PROJECT_DIR"

echo "==> Building Tauri binary (includes frontend build)..."
cargo tauri build --bundles deb 2>&1 | tail -5

cd "$PROJECT_DIR"

echo "==> Installing appimage-builder..."
python3 -m venv /tmp/sr-tuner-aib
set +u
source /tmp/sr-tuner-aib/bin/activate
set -u
pip install -q appimage-builder 2>&1 | tail -3

echo "==> Building AppImage..."
export VERSION
VERSION=$(grep '^version = ' src-tauri/Cargo.toml | cut -d'"' -f2)
export ARCH=x86_64
appimage-builder --recipe src-tauri/AppImageBuilder.yml --skip-tests 2>&1 | tail -10

deactivate 2>/dev/null || true
rm -rf /tmp/sr-tuner-aib

echo "==> Done!"
ls -lh "$PROJECT_DIR"/SR_Tuner-*.AppImage 2>/dev/null
