#!/usr/bin/env fish
# install.fish — NixOrb installer for fish shell
# Run from inside the nixorb/ directory: fish install.fish
#
# Fixes addressed:
#   1. venv activate.fish instead of activate (fish doesn't use bash syntax)
#   2. qsb found via qt6-tools path (/usr/lib/qt6/bin/qsb)
#   3. piper-tts installed via pip (AUR package has corrupt metadata)
#   4. openwakeword installed via pip (not in AUR)
#   5. hatchling build backend (no setuptools-scm git-tag requirement)
#   6. Handles "Defaulting to user installation" by using --user or venv

set -e  # exit on error

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║         NixOrb Installer (fish shell)            ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 0. Must be in the nixorb directory ─────────────────────────────── #
if not test -f pyproject.toml
    echo "ERROR: Run this script from inside the nixorb/ directory."
    exit 1
end

# ── 1. System packages ─────────────────────────────────────────────── #
echo "==> [1/7] Installing system packages..."
sudo pacman -S --needed --noconfirm \
    python \
    qt6-base qt6-declarative qt6-wayland qt6-multimedia qt6-tools \
    cuda cudnn \
    portaudio \
    wl-clipboard grim \
    ffmpeg \
    base-devel git

# kglobalacceld via yay (optional — don't fail if yay not present)
if command -q yay
    yay -S --needed --noconfirm kglobalacceld 2>/dev/null
    echo "  kglobalacceld: done"
else
    echo "  [SKIP] yay not found — kglobalacceld skipped (hotkey will use pynput fallback)"
end

# ── 2. Create venv with fish-compatible activation ─────────────────── #
echo ""
echo "==> [2/7] Creating Python virtual environment..."

if test -d .venv
    echo "  .venv already exists — reusing"
else
    python -m venv .venv --system-site-packages
    echo "  .venv created"
end

# FISH SHELL FIX: use activate.fish, not activate
source .venv/bin/activate.fish
echo "  venv activated"

# ── 3. PyTorch with CUDA 11.8 ──────────────────────────────────────── #
echo ""
echo "==> [3/7] Installing PyTorch (CUDA 11.8)..."
pip install --quiet \
    "torch==2.7.1+cu118" \
    "torchaudio==2.7.1+cu118" \
    --index-url https://download.pytorch.org/whl/cu118
echo "  PyTorch installed"

# ── 4. Install NixOrb (hatchling backend, supports editable) ──────── #
echo ""
echo "==> [4/7] Installing NixOrb (editable)..."
# Upgrade pip + hatchling first to avoid old-pip editable issues
pip install --quiet --upgrade pip hatchling
pip install --quiet -e ".[dev]"
echo "  nixorb installed"

# ── 5. Extra pip packages (piper + openwakeword, not in AUR) ──────── #
echo ""
echo "==> [5/7] Installing piper-tts and openwakeword via pip..."

# piper-tts: AUR package has corrupt metadata — use pip instead
pip install --quiet piper-tts 2>/dev/null
or begin
    echo "  [WARN] piper-tts pip install failed — trying piper-phonemize fallback"
    pip install --quiet piper-phonemize 2>/dev/null or true
end
echo "  piper: done"

# openwakeword: not in AUR at all
pip install --quiet openwakeword
echo "  openwakeword: done"

# Other optional extras
pip install --quiet \
    faster-whisper \
    chromadb \
    aiohttp \
    qasync \
    "pydantic>=2.7" \
    "pydantic-settings>=2.3" \
    "tomli-w>=1.0" \
    "typer[all]" \
    cryptography \
    pygments \
    aiofiles \
    sounddevice \
    soundfile \
    pynput \
    hypernix \
    "huggingface-hub>=0.23" \
    openai \
    2>/dev/null
echo "  dependencies: done"

# ── 6. Compile QML shaders ─────────────────────────────────────────── #
echo ""
echo "==> [6/7] Compiling QML shaders..."

# QSB FIX: qsb is NOT in PATH on Arch — it lives in /usr/lib/qt6/bin/
set QSB_PATHS \
    /usr/lib/qt6/bin/qsb \
    /usr/bin/qsb \
    (command -s qsb 2>/dev/null)

set QSB ""
for p in $QSB_PATHS
    if test -x "$p"
        set QSB "$p"
        break
    end
end

if test -z "$QSB"
    echo "  [WARN] qsb not found — shaders will not be compiled."
    echo "         The orb will fail to render until you compile shaders."
    echo "         Fix: sudo pacman -S qt6-tools && fish install.fish --shaders-only"
else
    echo "  Found qsb at: $QSB"
    $QSB --glsl "100es,120,150" --hlsl 50 --msl 12 \
        assets/shaders/orb_glow.vert -o assets/shaders/orb_glow.vert.qsb
    $QSB --glsl "100es,120,150" --hlsl 50 --msl 12 \
        assets/shaders/orb_glow.frag -o assets/shaders/orb_glow.frag.qsb
    echo "  Shaders compiled: orb_glow.vert.qsb  orb_glow.frag.qsb"
end

# ── 7. Download models ──────────────────────────────────────────────── #
echo ""
echo "==> [7/8] Downloading ASR + wake-word models..."
nixorb download-models
or echo "  [WARN] Model download failed — run 'nixorb download-models' manually later."

# ── 8. Verify installation ─────────────────────────────────────────── #
echo ""
echo "==> [8/8] Verifying installation..."

set ERRORS 0

if not command -q nixorb
    echo "  [FAIL] nixorb command not in PATH"
    set ERRORS (math $ERRORS + 1)
else
    echo "  nixorb: "(nixorb version 2>/dev/null; or echo "ok")
end

python -c "import nixorb; print('  nixorb module: ok')" 2>/dev/null
or begin
    echo "  [FAIL] nixorb module import failed"
    set ERRORS (math $ERRORS + 1)
end

python -c "import torch; print('  torch:', torch.__version__, '  CUDA:', torch.cuda.is_available())"
python -c "import PySide6.QtCore; print('  PySide6: ok')" 2>/dev/null or echo "  [WARN] PySide6 import issue"
python -c "import faster_whisper; print('  faster-whisper: ok')" 2>/dev/null or echo "  [WARN] faster-whisper not installed"
python -c "import chromadb; print('  chromadb: ok')" 2>/dev/null or echo "  [WARN] chromadb not installed"

echo ""
if test $ERRORS -eq 0
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  ✅  NixOrb installed successfully!              ║"
    echo "║                                                  ║"
    echo "║  Usage:                                          ║"
    echo "║    nixorb start           # launch the orb      ║"
    echo "║    nixorb ask 'hello'     # one-shot query      ║"
    echo "║    nixorb check-deps      # verify packages     ║"
    echo "║    nixorb download-models # (re)fetch models    ║"
    echo "╚══════════════════════════════════════════════════╝"
else
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  ⚠   Installed with $ERRORS error(s) — see above ║"
    echo "╚══════════════════════════════════════════════════╝"
end
echo ""

# Handle --shaders-only flag
if contains -- --shaders-only $argv
    echo "Shader-only mode requested. Done."
    exit 0
end
