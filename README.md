# NixOrb 🌐

**Floating AI assistant orb for Arch Linux — KDE Plasma 6 Wayland — GTX 1080**

[![CI](https://github.com/minerofthesoal/nixorb/actions/workflows/ci.yml/badge.svg)](https://github.com/minerofthesoal/nixorb/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nixorb?color=blue)](https://pypi.org/project/nixorb/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![Arch Linux](https://img.shields.io/badge/Arch-Linux-1793D1?logo=arch-linux)](https://archlinux.org/)

---

## What it is

NixOrb is a frameless, always-on-top AI assistant that floats on your Wayland desktop as a glowing GLSL-shader orb. Press a global hotkey (or say a wake word), speak, and it transcribes → thinks → speaks back — the orb pulsing to the audio in real time.

**Key features at a glance:**

| Category | What it does |
|---|---|
| **Orb UI** | Frameless QML shader orb with particle system, drag-to-reposition |
| **ASR** | Whisper Large v3 INT8 (~2 GB VRAM) via faster-whisper |
| **LLM** | OpenAI API, Ollama, llama.cpp local, with offline auto-fallback |
| **TTS** | OpenAI TTS, HuggingFace models, offline Piper |
| **VRAM** | Priority-eviction manager keeps 8 GB GTX 1080 from OOM crashing |
| **Actions** | Sandboxed bash execution via bubblewrap; user confirmation gate |
| **Memory** | ChromaDB long-term vector memory, injected into every prompt |
| **Vision** | `grim` screen capture → VLM description on demand |
| **Wake word** | OpenWakeWord always-on detector (low CPU) |
| **Plugins** | Drop-in `.py` plugin folder, hot-reloadable |
| **Clipboard** | `wl-paste` / `wl-copy` Wayland integration |
| **Tray** | KDE Plasma 6 system tray icon |

---

## Installation

### Option 1 — Arch Linux (pacman) — recommended

```bash
# Download the .pkg.tar.zst from the GitHub Releases page, then:
sudo pacman -U nixorb-0.1.0-1-x86_64.pkg.tar.zst
```

### Option 2 — PyPI

```bash
# Install PyTorch with CUDA first (GTX 1080 → CUDA 11.8)
pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

pip install nixorb
```

### Option 3 — AppImage (portable)

```bash
chmod +x NixOrb-0.1.0-x86_64.AppImage
./NixOrb-0.1.0-x86_64.AppImage start
```

### Option 4 — Flatpak

```bash
flatpak install NixOrb-0.1.0.flatpak
flatpak run io.nixorb.NixOrb
```

### Option 5 — From source (development)

```bash
git clone https://github.com/minerofthesoal/nixorb.git
cd nixorb

# System packages
sudo pacman -S --needed python qt6-base qt6-declarative qt6-wayland \
  qt6-multimedia qt6-tools cuda cudnn portaudio wl-clipboard grim \
  ffmpeg base-devel
yay -S --needed kglobalacceld piper-tts openwakeword

# Python environment
python -m venv .venv --system-site-packages
source .venv/bin/activate
pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[dev]"

# Compile QML shaders (required once)
cd assets/shaders
qsb --glsl "100es,120,150" --hlsl 50 --msl 12 orb_glow.vert -o orb_glow.vert.qsb
qsb --glsl "100es,120,150" --hlsl 50 --msl 12 orb_glow.frag -o orb_glow.frag.qsb
cd ../..

# Run
nixorb start
```

---

## Quick start

```bash
nixorb start               # Launch GUI orb
nixorb start --headless    # Daemon only (no Qt), for SSH/scripting
nixorb ask "What is 2+2?"  # One-shot LLM query
nixorb tts "Hello world"   # Speak text
nixorb transcribe audio.wav
nixorb check-deps          # Verify Arch packages
nixorb list-devices        # List microphones
nixorb list-plugins        # Show loaded plugins
nixorb memory-search "python"
nixorb memory-clear --yes
nixorb export-config --out backup.tar.gz.enc
nixorb import-config backup.tar.gz.enc
nixorb version
```

---

## Configuration

`~/.config/nixorb/config.toml` is created on first run from the shipped defaults.

Key settings:

```toml
hotkey       = "Ctrl+Alt+Space"   # global activation hotkey
llm_backend  = "openai"           # openai | ollama | local
llm_model    = "gpt-4o-mini"
openai_api_key = "sk-..."

tts_backend  = "openai"           # openai | huggingface | piper
tts_voice    = "alloy"

local_model_path    = "/home/you/models/mistral-7b.Q4_K_M.gguf"
fallback_model_path = "/home/you/models/phi-2.Q4_K_M.gguf"

wake_word_enabled   = true
wake_word_model     = "hey_jarvis_v0.1"
screen_capture_enabled = true
require_action_confirmation = true
```

All settings are also editable via the GUI (right-click orb → Settings).

---

## VRAM budget — GTX 1080 (8 GB)

| Model | VRAM | Priority |
|---|---|---|
| Whisper Large v3 INT8 | ~2.1 GB | LOW — evicted first |
| Local LLM 7B Q4_K_M | ~4.0 GB | HIGH — evicted last |
| HuggingFace TTS | ~1.5 GB | MEDIUM |
| Piper TTS (offline) | ~0.1 GB | MEDIUM |
| Driver + KDE compositor | ~0.5 GB | reserved |
| Safety buffer | 0.25 GB | reserved |

**Paging flow:** hotkey → Whisper loads → transcript → Whisper evicted → LLM loads → response → LLM evicted → TTS speaks.

---

## Writing a plugin

Drop a `.py` file into `~/.local/share/nixorb/plugins/` (or the `plugins/` repo dir):

```python
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "What this does — the LLM reads this to decide when to call it.",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "The input"}
            },
            "required": ["input"],
        },
    },
}

def my_tool(input: str) -> str:
    return f"Result: {input.upper()}"
```

Async plugins work too — use `async def`. Reload in Settings → Plugins → Reload.

---

## Architecture

```
Qt Main Thread          asyncio Event Loop          Thread Pool
──────────────          ──────────────────          ───────────
OrbWindow (QML)    ←── EventBus (PriorityQueue) ←── Whisper inference
SettingsWindow     ←── LLM stream chunks        ←── llama.cpp generate
NixOrbTray         ←── TTS audio chunks         ←── VRAM load/unload
HotkeyManager          VRAMManager monitor           sounddevice record
                        PluginLoader
                        VectorMemory (ChromaDB)
```

---

## Security

- Hard-deny list blocks destructive commands unconditionally
- `REQUIRE_CONFIRM` list prompts user confirmation before sensitive ops
- Optional bubblewrap (`bwrap`) sandbox with `--unshare-net`
- NixOrb refuses to run as root
- API keys stored in `~/.config/nixorb/config.toml` (mode 600 recommended)
- Config export is PBKDF2 + Fernet encrypted

---

## Contributing

```bash
git clone https://github.com/minerofthesoal/nixorb.git
cd nixorb
pip install -e ".[dev]"
ruff check nixorb/
mypy nixorb/
pytest tests/ -v
```

---

## License

MIT © NixOrb Contributors
