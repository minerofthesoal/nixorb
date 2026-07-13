# NixOrb v2.0 🌐

**Floating AI assistant orb for Arch Linux — KDE Plasma 6 Wayland — Local-Only**

A complete ground-up remake of NixOrb with a focus on local AI, rock-solid Qt6 stability, and clean modular architecture.

---

## What's New in v2.0

- **Local-only AI** — Ollama backend, no API keys needed
- **Fixed Qt errors** — Proper qasync integration, QSocketNotifier fixes, accessibility bridge disabled
- **Simplified architecture** — Removed broken backends, focused on what works
- **Better VRAM management** — Priority eviction keeps GTX 1080 from OOM
- **Offline TTS** — Piper + espeak-ng fallback
- **Improved settings** — GUI settings editor with live updates

## Architecture Overview

![NixOrb Architecture Pipeline](docs/architecture_pipeline.png)

## Quick Start

### Install (Arch Linux)

```bash
git clone https://github.com/minerofthesoal/nixorb.git
cd nixorb
chmod +x install.sh
./install.sh
```

### Start NixOrb

```bash
nixorb start          # Launch the floating orb
nixorb status         # Check system status
nixorb ask "What is 2+2?"   # One-shot query
nixorb check          # Check dependencies
```

## Pipeline Flow

```
Trigger (Hotkey/WakeWord/Click)
  → Record Audio (sounddevice + VAD)
  → Transcribe (faster-whisper INT8, ~2.1GB VRAM)
  → Think (Ollama LLM stream, localhost:11434)
  → Speak (Piper TTS, offline)
  → Execute (<ACTION> sandboxed bash)
```

## VRAM Budget (GTX 1080 8GB)

| Component | VRAM | Priority |
|-----------|------|----------|
| Whisper Large v3 INT8 | ~2.1 GB | LOW — evicted first |
| Ollama LLM (loaded by Ollama) | ~4.0 GB | HIGH |
| Piper TTS | ~0.1 GB | MEDIUM |
| System + KDE | ~0.5 GB | reserved |
| Safety buffer | 0.25 GB | reserved |

## Configuration

Edit `~/.config/nixorb/config.toml` or use the GUI (right-click orb → Settings):

```toml
hotkey = "Ctrl+Alt+Space"
llm_model = "llama3.2"
ollama_host = "http://localhost:11434"
wake_word_enabled = true
tts_backend = "piper"
```

## Keyboard Shortcuts

| Action | Default | Notes |
|--------|---------|-------|
| Activate | `Ctrl+Alt+Space` | Global hotkey (via pynput/XWayland) |
| KDE Shortcut | Custom | Set in System Settings → Shortcuts |
| Orb click | Double-click | Direct activation |
| Opacity | Scroll wheel | While hovering over orb |
| Drag | Click+drag | Reposition the orb |

## Project Structure

```
nixorb/
├── core/           # Event bus, VRAM manager
├── asr/            # Whisper speech-to-text + wake word
├── llm/            # Ollama local LLM backend
├── tts/            # Piper text-to-speech
├── ui/             # Qt6 UI (orb, tray, settings, hotkey)
├── action/         # Sandboxed command execution + clipboard
├── memory/         # ChromaDB vector memory
├── plugins/        # Plugin loader
├── utils/          # Paths, logging, web search
├── vision/         # Screen capture
├── main.py         # Entry point
├── cli.py          # CLI interface
└── settings.py     # Configuration
```

## Dependencies

- **System**: `python 3.12+`, `qt6-base`, `qt6-declarative`, `qt6-wayland`, `portaudio`, `wl-clipboard`, `grim`, `piper-tts`, `ollama`
- **Python**: See `requirements.txt` / `pyproject.toml`

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT © NixOrb Contributors
