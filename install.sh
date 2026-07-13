#!/bin/bash
# NixOrb installer for Arch Linux + KDE Plasma 6
# Usage: ./install.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      NixOrb Installer for Arch Linux     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo ""

# Check if running on Arch
if ! command -v pacman &> /dev/null; then
    echo -e "${RED}Error: This installer is for Arch Linux only.${NC}"
    echo "For other distributions, install dependencies manually."
    exit 1
fi

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo -e "${GREEN}✓ NVIDIA GPU detected: $GPU_INFO${NC}"
    HAS_NVIDIA=true
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected — CPU inference will be slower${NC}"
    HAS_NVIDIA=false
fi

# ── System Dependencies ────────────────────────────────────────── #
echo ""
echo -e "${BLUE}→ Installing system dependencies…${NC}"

sudo pacman -S --needed --noconfirm \
    python python-pip python-virtualenv \
    qt6-base qt6-declarative qt6-wayland qt6-multimedia \
    portaudio wl-clipboard grim slurp \
    espeak-ng bubblewrap \
    git base-devel

echo -e "${GREEN}✓ System dependencies installed${NC}"

# ── Optional: Piper TTS ────────────────────────────────────────── #
echo ""
echo -e "${BLUE}→ Installing Piper TTS…${NC}"
if ! command -v piper &> /dev/null; then
    echo "Piper not found — attempting to install from AUR"
    if command -v yay &> /dev/null; then
        yay -S --needed --noconfirm piper-tts-bin
    elif command -v paru &> /dev/null; then
        paru -S --needed --noconfirm piper-tts-bin
    else
        echo -e "${YELLOW}⚠ No AUR helper found (install yay or paru)${NC}"
        echo "Piper TTS can be installed manually from AUR: piper-tts-bin"
    fi
else
    echo -e "${GREEN}✓ Piper already installed${NC}"
fi

# ── Ollama ─────────────────────────────────────────────────────── #
echo ""
echo -e "${BLUE}→ Installing Ollama…${NC}"
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama…"
    curl -fsSL https://ollama.com/install.sh | sh
    echo -e "${GREEN}✓ Ollama installed${NC}"
else
    echo -e "${GREEN}✓ Ollama already installed${NC}"
fi

# Start Ollama service
if ! systemctl --user is-active --quiet ollama; then
    echo "Starting Ollama service…"
    systemctl --user enable ollama
    systemctl --user start ollama
    sleep 2
fi

# ── Python Environment ─────────────────────────────────────────── #
echo ""
echo -e "${BLUE}→ Setting up Python environment…${NC}"

VENV_DIR="${HOME}/.local/share/nixorb/venv"
mkdir -p "$VENV_DIR"

if [ ! -f "$VENV_DIR/bin/python" ]; then
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install PyTorch with CUDA if NVIDIA GPU detected
if [ "$HAS_NVIDIA" = true ]; then
    echo -e "${BLUE}→ Installing PyTorch with CUDA 11.8…${NC}"
    pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 \
        --index-url https://download.pytorch.org/whl/cu118
else
    echo -e "${BLUE}→ Installing PyTorch (CPU-only)…${NC}"
    pip install torch torchaudio
fi

# Install NixOrb
echo -e "${BLUE}→ Installing NixOrb…${NC}"
pip install -e "."

# ── Pull default model ─────────────────────────────────────────── #
echo ""
echo -e "${BLUE}→ Pulling default LLM model (llama3.2)…${NC}"
ollama pull llama3.2

# ── Piper voice model ──────────────────────────────────────────── #
echo ""
echo -e "${BLUE}→ Downloading Piper voice model…${NC}"
VOICE_DIR="${HOME}/.local/share/piper/voices"
mkdir -p "$VOICE_DIR"

if [ ! -f "$VOICE_DIR/en_US-lessac-medium.onnx" ]; then
    cd "$VOICE_DIR"
    curl -L -o en_US-lessac-medium.onnx \
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    curl -L -o en_US-lessac-medium.onnx.json \
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    echo -e "${GREEN}✓ Voice model downloaded${NC}"
else
    echo -e "${GREEN}✓ Voice model already exists${NC}"
fi

# ── Desktop Entry ──────────────────────────────────────────────── #
echo ""
echo -e "${BLUE}→ Creating desktop entry…${NC}"

mkdir -p "${HOME}/.local/share/applications"
cat > "${HOME}/.local/share/applications/nixorb.desktop" << 'EOF'
[Desktop Entry]
Name=NixOrb
Comment=Floating AI Assistant
Exec=nixorb start
Icon=audio-input-microphone
Type=Application
Categories=Utility;Audio;
StartupNotify=false
EOF

echo -e "${GREEN}✓ Desktop entry created${NC}"

# ── KDE Shortcut ───────────────────────────────────────────────── #
echo ""
echo -e "${YELLOW}⚠ Important: Set up KDE shortcut:${NC}"
echo "   1. Open System Settings → Shortcuts → Custom Shortcuts"
echo "   2. Add a new global shortcut → Command/URL"
echo "   3. Set trigger: Meta+Space (or your preference)"
echo "   4. Set command: nixorb trigger"
echo "   5. Apply and save"
echo ""

# ── Done ───────────────────────────────────────────────────────── #
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     NixOrb installation complete!        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo "Start NixOrb:     nixorb start"
echo "Check status:     nixorb status"
echo "Edit config:      nixorb config"
echo "Check deps:       nixorb check"
echo ""
echo -e "${BLUE}Logs: ~/.local/share/nixorb/logs/nixorb.log${NC}"
echo ""
