# NixOrb — Floating AI Assistant Orb
## Arch Linux · KDE Plasma 6 Wayland · GTX 1080 · Python 3.12

---

## Project Structure

```
nixorb/
├── README.md
├── ARCHITECTURE.md
├── pyproject.toml
├── setup.cfg
├── nixorb.desktop                  # XDG autostart entry
├── assets/
│   ├── orb.qml                     # QtQuick particle shader orb
│   ├── tray_icon.png
│   └── shaders/
│       ├── orb_glow.frag
│       └── particle.vert
├── config/
│   ├── default.toml                # Shipped defaults
│   └── schema.py                   # Pydantic v2 config schema
├── nixorb/
│   ├── __init__.py
│   ├── main.py                     # Entry point / daemon bootstrap
│   ├── cli.py                      # Typer CLI
│   ├── settings.py                 # Runtime config manager
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── event_bus.py            # Central asyncio EventBus
│   │   ├── thread_pool.py          # Managed ThreadPoolExecutor
│   │   ├── vram_manager.py         # GTX 1080 VRAM paging
│   │   └── aur_checker.py          # Startup dependency checker
│   │
│   ├── asr/
│   │   ├── __init__.py
│   │   ├── whisper_engine.py       # faster-whisper Large v3
│   │   └── wake_word.py            # OpenWakeWord listener
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── backend_base.py         # ABC for all LLM backends
│   │   ├── local_backend.py        # llama-cpp-python / Ollama
│   │   ├── openai_backend.py       # OpenAI / compatible APIs
│   │   └── offline_fallback.py     # Auto-fallback manager
│   │
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── tts_base.py             # ABC
│   │   ├── openai_tts.py           # OpenAI TTS
│   │   ├── hf_tts.py               # HuggingFace model TTS
│   │   └── piper_tts.py            # Offline Piper fallback
│   │
│   ├── vision/
│   │   ├── __init__.py
│   │   └── screen_capture.py       # grim + VLM integration
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── vector_store.py         # ChromaDB long-term memory
│   │   └── session_store.py        # In-session sliding window
│   │
│   ├── action/
│   │   ├── __init__.py
│   │   ├── executor.py             # Sandboxed bash execution
│   │   └── clipboard.py            # Wayland clipboard (wl-clipboard)
│   │
│   ├── plugins/
│   │   ├── __init__.py
│   │   ├── loader.py               # Dynamic plugin loader
│   │   └── builtin/
│   │       ├── systemd_plugin.py
│   │       └── kdeconnect_plugin.py
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── orb_window.py           # Frameless floating orb (PySide6)
│   │   ├── orb_qml_bridge.py       # Python ↔ QML bridge
│   │   ├── settings_window.py      # Settings GUI
│   │   ├── log_widget.py           # Syntax-highlighted log
│   │   ├── tray_icon.py            # KDE system tray
│   │   └── hotkey.py               # Wayland global hotkey (KGlobalAccel)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── audio.py                # PyAudio device management
│       ├── crypto.py               # Config encrypt/export
│       └── hypernix_client.py      # hypernix integration
│
├── plugins/                        # User-dropped plugin folder
│   └── example_plugin.py
└── tests/
    ├── test_vram_manager.py
    ├── test_executor.py
    └── test_event_bus.py
```

---

## Arch Linux Dependencies

### pacman packages
```bash
sudo pacman -S --needed \
  python python-pip python-virtualenv \
  qt6-base qt6-declarative qt6-wayland qt6-multimedia \
  cuda cudnn \
  portaudio \
  wl-clipboard grim \
  ffmpeg \
  cmake ninja \
  base-devel git \
  libnotify \
  python-pyqt6 python-pyqt6-webengine
```

### AUR packages (via yay)
```bash
yay -S --needed \
  python-pyside6 \
  openwakeword \
  piper-tts \
  kglobalacceld
```

### Python environment
```bash
python -m venv .venv --system-site-packages
source .venv/bin/activate

pip install \
  faster-whisper \
  torch==2.7.1+cu118 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118 \
  hypernix \
  chromadb \
  llama-cpp-python \
  openai \
  typer[all] \
  pydantic pydantic-settings \
  sounddevice soundfile \
  numpy \
  pyaudio \
  aiofiles \
  tomli tomli-w \
  cryptography \
  rich \
  pygments \
  openwakeword \
  huggingface-hub \
  transformers accelerate \
  bitsandbytes \
  pynput
```
