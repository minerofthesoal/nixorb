## [2.0.1] — 2026-08-27

### Fixed — startup

- **`nixorb start` crashed immediately.** `main.py` and `cli.py` both imported
  `nixorb.llm.ollama_backend`, which was never written. Added it: streaming
  chat, health checks that name the fix ("Start it with: ollama serve"),
  tool-call capture, and a shared HTTP session.
- **The orb was an invisible 0×0 window.** `orb.qml` bound its width to
  `orbBridge.orbSize`, a property `OrbBridge` did not have; `undefined`
  collapsed to 0 and `QQuickView`'s default `SizeViewToRootObject` shrank the
  window to match. Added the property and made the window size authoritative.
  With the process running and nothing on screen, this looked exactly like a
  hang.
- `orb.qml` no longer imports `QtQuick.Shapes` or the `NixOrb` QML module —
  neither was used, and either could fail the whole QML load.
- ChromaDB was constructed synchronously on the event loop during startup;
  it now initializes off-thread, with telemetry disabled so an offline or
  firewalled machine does not stall on a posthog call.
- Startup failures now log what went wrong and where the log lives, instead
  of printing a bare traceback.

### Fixed — the assistant froze on activation

- **Command confirmation could never complete.** `ActionExecutor` waited on a
  future it created locally and handed to nobody, while `confirm_dialog` kept
  a separate registry the executor never read. Every command sat for the full
  60s timeout and was then reported as denied, with no dialog ever shown.
  Both halves now key off a shared `request_id` over ACTION_CONFIRMED /
  ACTION_DENIED.
- **A turn ran as an event-bus handler, deadlocking the bus.** Handlers are
  awaited inline by the dispatch loop, so the turn blocked delivery of the
  ACTION_REQUESTED it was itself waiting on. Orb state events never arrived
  either, so the orb stayed frozen. Turns now run as their own task.
- The event-bus dispatch task was created without keeping a reference and
  could be garbage-collected mid-flight, silently killing the bus.
- `queue.task_done()` now runs in a `finally`, so a raising handler cannot
  wedge shutdown's `queue.join()`.
- A handler that blocks dispatch for more than 5s now logs a warning naming
  it, rather than looking like a hang.

### Fixed — shutdown

- Quitting from the tray or with Ctrl-C called `app.quit()`, which tore down
  the qasync loop before cleanup ran ("no running event loop" on every step,
  leaked HTTP session, un-awaited coroutines). Shutdown is now a graceful
  unwind, and SIGINT/SIGTERM are handled.

### Fixed — errors that had no logs

- Wake word: `preload()` sat outside its `try`, so a missing `openwakeword`
  (an *optional* extra that was enabled by default) killed the task with an
  unhandled exception. Its capture loop also ran `sd.InputStream.read()`
  directly on the event loop, stalling Qt and the bus ~80ms at a time. It now
  runs in a worker thread, and `wake_word_enabled` defaults to `false`.
- Whisper hard-coded `device="cuda"`; a CPU-only machine (or a CUDA box with
  no cuDNN) got no ASR at all. It now falls back to CPU, and a missing
  `faster-whisper` reports the install command.
- `sounddevice`/`soundfile` were imported at module scope, so a missing
  PortAudio was a hard startup crash rather than a degraded feature.
- `settings.web_search_max_results` was read by `main.py` but not defined on
  `Settings` — every web-search turn raised AttributeError.
- `cryptography` is imported by `utils/crypto.py` but was never declared as a
  dependency.
- TTS failures were log-only; they now surface on the bus, so the orb can say
  why it went quiet instead of just not speaking.
- `write_clipboard` had no timeout while `read_clipboard` did.
- Plugin hot-reload re-ran stale bytecode: Python validates `__pycache__` on
  (mtime, size), so editing a plugin without changing its length reloaded the
  old code — precisely what "Reload Plugins" exists to avoid.
- Running as root aborted the whole app; it now disables command execution
  only, and says so.

### Changed

- **Plugins actually work now.** `PluginLoader.dispatch()` was called by
  `ToolDispatcher` but never existed, and `main.py` advertised tools to the
  model without ever running the calls that came back. Tool calls are now
  dispatched (sync and async plugins, off-thread, with a timeout) and fed
  back to the model, up to `MAX_TOOL_ROUNDS`.
- `require_action_confirmation` now gates **every** command, not just ones
  matching a substring list — `echo $(rm -rf ~)` sails past any such list, so
  it cannot be the security boundary. The hard denylist is a separate, harder
  check, and `is_high_risk()` is now only a UI hint.
- The bubblewrap sandbox moved to its own `sandbox_actions` setting
  (default off). It was previously tied to `require_action_confirmation`,
  which meant every approved command ran read-only with no network and failed
  for no visible reason.
- Mouse handling moved out of the QML `MouseArea`, which was swallowing drag
  and double-click, and made a *single* click start a conversation. Drag,
  double-click-to-activate, right-click menu and scroll-to-fade now all work.
- `install.sh` links `nixorb` into `~/.local/bin` (it previously installed
  into a venv that was never on `PATH`), warns if that is not on `PATH`, and
  no longer assumes Ollama ships a `--user` systemd unit.
- `nixorb status` now reports whether Ollama is actually reachable.
- Added a Troubleshooting table to the README.

### Tests

- The suite was written against the pre-v2 API and could not pass: 28 failing,
  19 collection errors. Now 125 passing, with `ruff` and `mypy` clean.
- Added `tests/test_pipeline_integration.py`, covering the whole turn against
  a stub Ollama, so the deadlocks above cannot come back silently.

## [0.01.0.05] — 2026-06-07

### Fixed
- Installed wheel asset lookup now resolves `share/nixorb/assets` so `nixorb start` can load the orb QML and tray icon outside a source checkout.
- Replaced shader-only orb backdrop with a pure-QML fallback so missing `.qsb` files no longer prevent the app from appearing.
- Startup now shows the tray/orb before heavyweight dependency and model initialization, then preloads the ASR model in the background.
- Whisper ASR now honors the configured model, falls back from CUDA to CPU, emits microphone level activity, and waits for initial speech instead of timing out after startup silence.
- Chroma memory now uses a local deterministic embedding function to avoid first-run network downloads in tests/offline installs.
- Added missing HuggingFace runtime dependencies and CPU fallback for HuggingFace TTS.

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
