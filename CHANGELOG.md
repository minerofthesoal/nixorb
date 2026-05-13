## [0.01.0.04] — 2026-05-12

### Fixed
- ruff: all 46 CI lint errors resolved
- pacman PKGBUILD: removed llama-cpp-python (requires scikit_build_core)
- AppImage: removed invalid schema keys (comp, name) from AppImageBuilder.yml
- Flatpak: removed non-existent cuda SDK extension, use CPU fallback
- PyPI version scheme: switched to PEP-440 compatible format
- All builds should now pass

# Changelog

All notable changes to NixOrb are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [0.01.0.01] — 2026-05-11

### Added
- Full GUI floating orb (PySide6 + QML GLSL shader)
- faster-whisper Large v3 INT8 ASR with VAD gating
- HuggingFace, OpenAI, Ollama, and local llama.cpp LLM backends
- Default model: `torphix/stablelm-2-glados-v1` (GLaDOS personality)
- TTS: GLaDOS HF, OpenAI TTS, offline Piper
- `nixorb config` / `nixorb config-gui` commands
- `nixorb run` alias for `nixorb start`
- Web search via DuckDuckGo (no API key)
- Screen capture: CogFlorence-2.2-Large + Qwen3.5-4B VLM options
- Long-term vector memory via ChromaDB
- Plugin system with hot-reload (compile+exec for true reload)
- Built-in plugins: systemd, KDE Connect, weather, volume, notes, timer
- VRAM paging manager for GTX 1080 (8 GB)
- OpenWakeWord 0.4.0 compatibility fix
- XWayland auto-detection for pynput hotkeys
- `QSocketNotifier` warning fixed (poll-based shutdown)
- `KeyboardInterrupt` handled cleanly
- `emit_sync` in standalone config-gui no longer warns (loop guard)
- GitHub Actions: CI + manual-dispatch Release workflow
- Arch pacman PKGBUILD, Flatpak manifest, AppImage recipe
- Full test suite: 37 tests passing

### Fixed
- `pip install -e .` crash (`setuptools.backends.legacy` → hatchling)
- `source .venv/bin/activate` in fish → `activate.fish`
- `qsb` not in PATH → full path `/usr/lib/qt6/bin/qsb`
- piper-tts AUR corrupt package → pip install
- openwakeword not in AUR → pip install
- Plugin reload not picking up file changes → compile+exec strategy

## [0.1.0] — 2026-05-09

### Added
- Initial project structure
- Core event bus, VRAM manager, ASR engine
- Basic orb window, settings GUI, CLI
