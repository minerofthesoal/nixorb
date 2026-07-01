#!/usr/bin/env bash
# install.sh — NixOrb installer for bash/zsh
# Usage: bash install.sh
# Run from inside the nixorb/ directory.

set -euo pipefail

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║         NixOrb Installer (bash/zsh)              ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

[[ -f pyproject.toml ]] || { echo "ERROR: run from inside nixorb/"; exit 1; }

# ── 1. System packages ─────────────────────────────────────────────── #
echo "==> [1/7] Installing system packages..."
sudo pacman -S --needed --noconfirm \
    python qt6-base qt6-declarative qt6-wayland qt6-multimedia qt6-tools \
    cuda cudnn portaudio wl-clipboard grim ffmpeg base-devel git

command -v yay &>/dev/null && \
    yay -S --needed --noconfirm kglobalacceld 2>/dev/null || \
    echo "  [SKIP] yay not found — using pynput hotkey fallback"

# ── 2. Venv ────────────────────────────────────────────────────────── #
echo ""
echo "==> [2/7] Creating Python virtual environment..."
[[ -d .venv ]] || python -m venv .venv --system-site-packages
# shellcheck disable=SC1091
source .venv/bin/activate
echo "  venv activated"

# ── 3. PyTorch CUDA 11.8 ───────────────────────────────────────────── #
echo ""
echo "==> [3/7] Installing PyTorch (CUDA 11.8)..."
pip install -q "torch==2.7.1+cu118" "torchaudio==2.7.1+cu118" \
    --index-url https://download.pytorch.org/whl/cu118

# ── 4. NixOrb (hatchling editable) ────────────────────────────────── #
echo ""
echo "==> [4/7] Installing NixOrb (editable)..."
pip install -q --upgrade pip hatchling
pip install -q -e ".[dev]"

# ── 5. pip-only extras ─────────────────────────────────────────────── #
echo ""
echo "==> [5/7] Installing piper-tts and openwakeword via pip..."
pip install -q piper-tts 2>/dev/null || pip install -q piper-phonemize 2>/dev/null || true
pip install -q openwakeword
pip install -q faster-whisper chromadb aiohttp qasync pygments cryptography \
    pynput hypernix openai "huggingface-hub>=0.23" sounddevice soundfile 2>/dev/null || true

# ── 6. Shaders ─────────────────────────────────────────────────────── #
echo ""
echo "==> [6/7] Compiling QML shaders..."
QSB=""
for p in /usr/lib/qt6/bin/qsb /usr/bin/qsb "$(command -v qsb 2>/dev/null)"; do
    [[ -x "$p" ]] && { QSB="$p"; break; }
done

if [[ -z "$QSB" ]]; then
    echo "  [WARN] qsb not in PATH. Trying /usr/lib/qt6/bin/qsb..."
    if [[ -x /usr/lib/qt6/bin/qsb ]]; then
        QSB=/usr/lib/qt6/bin/qsb
    else
        echo "  [WARN] qsb not found — install qt6-tools. Shaders NOT compiled."
        echo "         Run:  /usr/lib/qt6/bin/qsb --glsl '100es,120,150' --hlsl 50 --msl 12 \\"
        echo "                  assets/shaders/orb_glow.vert -o assets/shaders/orb_glow.vert.qsb"
    fi
fi

if [[ -n "$QSB" ]]; then
    "$QSB" --glsl "100es,120,150" --hlsl 50 --msl 12 \
        assets/shaders/orb_glow.vert -o assets/shaders/orb_glow.vert.qsb
    "$QSB" --glsl "100es,120,150" --hlsl 50 --msl 12 \
        assets/shaders/orb_glow.frag -o assets/shaders/orb_glow.frag.qsb
    echo "  Shaders compiled OK"
fi

# ── 7. Download models ──────────────────────────────────────────────── #
echo ""
echo "==> [7/8] Downloading ASR + wake-word models..."
echo "  (LLM/TTS/vision models download automatically on first use instead —"
echo "   they're much larger and picking the wrong one wastes bandwidth.)"
nixorb download-models || echo "  [WARN] Model download failed — run 'nixorb download-models' manually later."

# ── 8. Verify ──────────────────────────────────────────────────────── #
echo ""
echo "==> [8/8] Verifying..."
nixorb version && echo "  nixorb CLI: OK"
python -c "import torch; print('  torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import PySide6.QtCore; print('  PySide6: OK')" 2>/dev/null || echo "  [WARN] PySide6 issue"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  ✅  Done. Run:  nixorb start                    ║"
echo "║                                                    ║"
echo "║  Wake word is OFF by default. Enable it with:     ║"
echo "║    nixorb config wake_word_enabled true           ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
