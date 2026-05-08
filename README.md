# NixOrb 🌐

**Floating AI assistant orb for Arch Linux — KDE Plasma 6 Wayland — GTX 1080**

---

## What it is

NixOrb is a frameless, always-on-top AI assistant that lives on your screen as a glowing orb. Activate it with a global hotkey (or voice wake-word), speak your request, and it transcribes → reasons → speaks back — all with a particle-shader orb that pulses to the audio.

---

## Quick Start

### 1. Install system packages

```bash
sudo pacman -S --needed \
  python qt6-base qt6-declarative qt6-wayland qt6-multimedia \
  cuda cudnn portaudio wl-clipboard grim ffmpeg cmake ninja \
  base-devel git libnotify

yay -S --needed kglobalacceld piper-tts openwakeword
```

### 2. Python environment

```bash
git clone https://github.com/minerofthesoal/nixorb
cd nixorb

python -m venv .venv --system-site-packages
source .venv/bin/activate

# PyTorch with CUDA 11.8 (matches GTX 1080 drivers)
pip install torch==2.7.1+cu118 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118

pip install -e ".[dev]"
```

### 3. Compile QML shaders

```bash
cd assets/shaders
# Vertex passthrough
cat > orb_glow.vert << 'EOF'
#version 440
layout(location=0) in vec4 qt_Vertex;
layout(location=1) in vec2 qt_MultiTexCoord0;
layout(location=0) out vec2 qt_TexCoord0;
layout(std140, binding=0) uniform buf { mat4 qt_Matrix; float qt_Opacity; };
void main() { qt_TexCoord0 = qt_MultiTexCoord0; gl_Position = qt_Matrix * qt_Vertex; }
EOF
qsb --glsl "100es,120,150" --hlsl 50 --msl 12 orb_glow.vert -o orb_glow.vert.qsb
qsb --glsl "100es,120,150" --hlsl 50 --msl 12 orb_glow.frag -o orb_glow.frag.qsb
cd ../..
```

### 4. Configure

```bash
cp config/default.toml ~/.config/nixorb/config.toml
# Edit with your API keys and preferences
```

### 5. Run

```bash
# GUI mode (full orb)
nixorb start

# Headless daemon
nixorb start --headless

# One-shot text query
nixorb ask "What is my IP address?"

# Transcribe audio file
nixorb transcribe recording.wav

# Check dependencies
nixorb check-deps

# Export encrypted config
nixorb export-config --out backup.tar.gz.enc
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Qt Main Thread                           │
│  OrbWindow (QQuickView)  ·  SettingsWindow  ·  NixOrbTray      │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Qt signals ↕ asyncio via qasync
┌──────────────────────▼──────────────────────────────────────────┐
│                   asyncio Event Loop                            │
│  EventBus → handles all inter-subsystem messaging               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐      │
│  │ Whisper  │  │   LLM    │  │   TTS    │  │  Vision   │      │
│  │  Engine  │  │ Backend  │  │  Engine  │  │  Capture  │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬─────┘      │
└───────┼─────────────┼─────────────┼───────────────┼────────────┘
        │             │             │               │
┌───────▼─────────────▼─────────────▼───────────────▼────────────┐
│              ThreadPoolExecutor (blocking I/O)                  │
│  VRAM loads · sounddevice recording · llama.cpp inference       │
└─────────────────────────────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────┐
│         VRAMManager (GTX 1080)       │
│  whisper(2GB) ↔ local_llm(4GB)      │
│  Evict LOW priority before HIGH      │
│  nvidia-smi polling every 5s        │
└──────────────────────────────────────┘
```

---

## VRAM Budget (GTX 1080, 8 GB)

| Model                        | VRAM (INT8) | Priority |
|------------------------------|-------------|----------|
| Whisper Large v3             | ~2.0 GB     | LOW      |
| Local LLM (7B Q4_K_M)        | ~4.0 GB     | HIGH     |
| TTS (Piper, offline)         | ~0.1 GB     | MEDIUM   |
| Driver + KDE compositor      | ~0.5 GB     | reserved |
| **Safety buffer**            | **0.25 GB** | reserved |

**Paging flow:** hotkey pressed → Whisper loads (LLM evicted) → transcript ready → Whisper evicted → LLM loads → response streamed → LLM evicted → TTS speaks.

---

## 12 Advanced Features

| # | Feature | Implementation |
|---|---------|----------------|
| 1 | VRAM Paging | `core/vram_manager.py` — priority eviction, nvidia-smi polling |
| 2 | Screen Context | `vision/screen_capture.py` — `grim` → base64 → VLM |
| 3 | Vector Memory | `memory/vector_store.py` — ChromaDB persistent store |
| 4 | System Tray | `ui/tray_icon.py` — KDE QSystemTrayIcon |
| 5 | Offline Fallback | `llm/backends.py` — OfflineFallbackManager auto-switches |
| 6 | AUR Checker | `core/aur_checker.py` — pacman -Q at startup |
| 7 | Wake Word | `asr/wake_word.py` — OpenWakeWord ONNX always-on |
| 8 | Plugins | `plugins/loader.py` — hot-reload Python plugin folder |
| 9 | Clipboard | `action/clipboard.py` — wl-paste/wl-copy integration |
| 10 | Particle Shader | `assets/orb.qml` + `shaders/orb_glow.frag` — GLSL voronoi |
| 11 | Config Export | `utils/crypto.py` — PBKDF2 + Fernet encrypted tar.gz |
| 12 | Syntax Log | `ui/settings_window.py` — Pygments HTML in QTextEdit |

---

## Plugin API

Drop a `.py` file in `plugins/`. It must expose:

```python
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "What this tool does",
        "parameters": { "type": "object", "properties": {...} }
    }
}

def my_tool(param1: str, param2: int = 0) -> str:
    return "result"
```

The LLM will call your tool when appropriate. NixOrb dispatches it and feeds the result back into the conversation automatically.

---

## Security Model

- Commands require user confirmation unless allowlisted.
- `ALWAYS_DENY` list blocks destructive patterns unconditionally.
- Optional `bwrap` (bubblewrap) sandbox isolates execution from filesystem.
- NixOrb refuses to run as root.
- API keys stored in `~/.config/nixorb/config.toml` (chmod 600 recommended).

---

## Wayland Notes

- Uses `qt6-wayland` for native Wayland rendering.
- Orb window uses `Qt.Tool | Qt.FramelessWindowHint` — appears as overlay.
- Global hotkeys: KGlobalAccel D-Bus (Wayland-native) with pynput XWayland fallback.
- Screen capture via `grim` (Wayland screenshotting tool).
- Clipboard via `wl-paste` / `wl-copy`.

---

## License

MIT

(this was vibecoded)
