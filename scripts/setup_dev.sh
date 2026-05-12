#!/usr/bin/env bash
# scripts/setup_dev.sh — Full development environment setup for NixOrb.
# Run once after cloning the repo.
#
# Usage (bash or zsh):    bash scripts/setup_dev.sh
# Usage (fish shell):     fish install.fish
# Usage (CI/containers):  bash scripts/setup_dev.sh --no-pacman

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/..")"
cd "$PROJECT_ROOT"

NO_PACMAN=false
[[ "${1:-}" == "--no-pacman" ]] && NO_PACMAN=true

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║        NixOrb Dev Setup                         ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. System packages ────────────────────────────────────────────── #
if ! $NO_PACMAN && command -v pacman &>/dev/null; then
    echo "==> [1/5] Installing system packages…"
    sudo pacman -S --needed --noconfirm \
        python qt6-base qt6-declarative qt6-wayland qt6-multimedia qt6-tools \
        portaudio wl-clipboard grim ffmpeg base-devel git
    echo "  ✅ System packages"
else
    echo "==> [1/5] Skipping pacman (--no-pacman or not Arch)"
fi

# ── 2. Python venv ────────────────────────────────────────────────── #
echo ""
echo "==> [2/5] Python virtual environment…"
if [[ ! -d .venv ]]; then
    python -m venv .venv --system-site-packages
    echo "  Created .venv"
fi

# Detect shell and source accordingly
if [[ -n "${FISH_VERSION:-}" ]]; then
    echo "  (fish detected — run: source .venv/bin/activate.fish)"
else
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "  Activated .venv"
fi

# ── 3. PyTorch ────────────────────────────────────────────────────── #
echo ""
echo "==> [3/5] Installing PyTorch…"
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  ✅ PyTorch with CUDA already installed"
else
    echo "  Installing PyTorch CPU (use: pip install torch --index-url .../cu118 for CUDA)"
    pip install -q "torch>=2.7" --index-url https://download.pytorch.org/whl/cpu
fi

# ── 4. NixOrb + dev deps ──────────────────────────────────────────── #
echo ""
echo "==> [4/5] Installing NixOrb (editable) + dev deps…"
pip install -q --upgrade pip hatchling
pip install -q -e ".[dev]"
pip install -q piper-tts openwakeword sounddevice soundfile 2>/dev/null || true
echo "  ✅ NixOrb installed"

# ── 5. Shaders ────────────────────────────────────────────────────── #
echo ""
echo "==> [5/5] Compiling QML shaders…"
bash scripts/compile_shaders.sh || echo "  ⚠  Shader compilation failed (qt6-tools missing?)"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  ✅  Dev setup complete!                         ║"
echo "║                                                  ║"
echo "║  Run:  nixorb start                              ║"
echo "║  Test: bash scripts/run_tests.sh                 ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
