## [0.3.3] — 2026-09-02

Merges the parallel Hugging Face work from `main` with the branch's native
Nemotron and voice-streaming work. Both sides had independently built HF
backends under different filenames; this keeps the union rather than one
side's half.

### Fixed

- **The default config could not load its own default model.** `main`'s base
  dependencies pinned `transformers>=4.44.0` while defaulting `asr_model` to
  `nvidia/nemotron-3.5-asr-streaming-0.6b`, whose architecture only exists
  from transformers 5.13. A clean install resolved happily and then died
  inside `from_pretrained` with an unrecognised-model-type error. The floor
  is now `>=5.13.0`, and `nixorb.hf.require_transformers()` reports the
  installed version and the upgrade command instead of failing obscurely.
- **The default Nemotron checkpoint ran without streaming.** `main` selected
  it through the generic `automatic-speech-recognition` pipeline, which
  re-transcribes a rolling buffer. The ASR factory now routes any Nemotron
  checkpoint to the native cache-aware backend, whatever `asr_backend` says
  — and routes `asr_backend="nemotron"` with a non-Nemotron model back to
  the generic engine instead of failing to load.

### Merged — kept from `main`

- `openai_compat_backend.py`: `llm_backend="openai"` against any
  OpenAI-compatible endpoint (OpenAI, vLLM, LM Studio, llama.cpp server).
- `hf_llm_backend.py` replaces the branch's thinner HF LLM backend: it also
  loads GGUF through llama-cpp-python, which is what makes a model fit
  beside ASR on 8 GB.
- `hf_asr_engine.py` replaces the branch's HF ASR engine: it approximates
  streaming for models with no native streaming API.
- Microphone resolution by name, monitor/loopback avoidance, and ambient
  noise-floor calibration — moved into `asr/base.py` so *every* engine gets
  them, rather than being duplicated in two of the three.
- `glados` and `openai` TTS backends, the builtin calculator / system-info /
  timer / weather plugins, the wake-word training scripts, `test_paths.py`,
  and main's richer settings documentation.

### Merged — kept from this branch

- Native Nemotron cache-aware streaming (`nemotron_asr.py`).
- `tts/speaker.py`: sentence-by-sentence playback, barge-in, and `<ACTION>`
  suppression during streaming.
- `hf.py`: shared device/dtype/token/cache plumbing and the transformers
  version guard.
- Follow-up listening, and a system prompt written for answers that are
  spoken rather than read.

### Changed

- One factory per stage, exporting both `build_*` (main's spelling) and
  `create_*` (the branch's), so no call site had to be rewritten twice.
- ASR backend names accept both `faster_whisper` and `faster-whisper`.
- `tts_hf_repo` is the canonical TTS model setting (main's name);
  `tts_hf_model` is still read as a fallback.

### Note

`HuggingFaceTTS` passes `trust_remote_code=True` when building a
`text-to-speech` pipeline, inherited from `main` — the default
`BreezeBlue/Breeze-TTS-2` is a voice-design model that does not load without
it. That executes Python from the model repo. `hf_trust_remote_code` (default
off) still governs the ASR and LLM paths.

## [0.3.2] — 2026-09-01

Version reset from the 2.0.x line to an honest pre-1.0 number.

### Added — any model on Hugging Face

Every stage is now pluggable, selected by `asr_backend`, `llm_backend` and
`tts_backend`. The previous "Hugging Face" modules could not work at all:
they read `Settings` fields (`tts_hf_repo`, `hf_token`, `openai_api_key`)
that v2 had removed, and nothing imported them. They are replaced with code
that runs.

- **ASR** — `faster-whisper` (default), or `huggingface` for any model with
  an `automatic-speech-recognition` tag (Whisper, Wav2Vec2, MMS, SeamlessM4T,
  Parakeet…). Language is only passed to models that accept it; sending
  `language=` to a CTC model is an error, not a no-op.
- **LLM** — `ollama` (default), or `huggingface` for any causal LM, streamed
  token by token through `TextIteratorStreamer` with the model's own chat
  template. Generation runs in a worker thread; on the event loop it would
  freeze the UI for the whole answer. Tool calls emitted in-band as
  `<tool_call>` JSON (Qwen, Hermes, …) are parsed and dispatched.
- **TTS** — `piper` (default), `huggingface` for any TTS model, or `espeak`.
  SpeechT5 gets explicit speaker-embedding handling since the pipeline
  cannot supply an x-vector on its own.
- `nixorb/hf.py` centralises device selection, dtype, token resolution and
  cache directory. A missing package now names the pip command that fixes it.
- `nixorb status` reports all three backends and whether each can load.

### Added — NVIDIA Nemotron 3.5 ASR

Native support for `nvidia/nemotron-3.5-asr-streaming-0.6b`, with its own
backend rather than the generic pipeline, because the point of the model is
cache-aware streaming: partial transcripts arrive while you are still
talking. 40 language-locales, punctuation and capitalisation, and `auto`
language detection whose `<xx-XX>` tag is stripped from the transcript and
logged.

- `asr_nemotron_lookahead` selects the right-context: 0/3/6/13 frames →
  80/320/560/1120 ms, validated against the checkpoint's own supported list
  rather than a hardcoded one.
- Setting `asr_model` to a Nemotron checkpoint selects the native backend
  even if `asr_backend` was left alone, so streaming is not lost silently.

### Added — it behaves like a voice assistant

- **Speaks while it thinks.** `nixorb/tts/speaker.py` synthesises each
  sentence as the model produces it. Measured on a simulated stream: first
  words at 0.30s against generation finishing at 0.91s.
- **Barge-in.** Triggering while it is talking stops playback and drops the
  queue instead of waiting its turn (`barge_in`).
- **Follow-ups.** After answering it listens again briefly, so a second
  question needs no second hotkey press (`follow_up_seconds`, capped at
  `MAX_FOLLOW_UPS` so a room with a television cannot loop forever).
- `<ACTION>` blocks are suppressed from speech *as they stream*, including
  when the tag is split across token boundaries — not read aloud and cleaned
  up afterwards.
- The default system prompt now asks for spoken answers: one or two
  sentences, answer first, no markdown.

### Fixed

- **transformers floor was too low for the default model.** The `[hf]` extra
  allowed `>=4.44` while the default ASR checkpoint needs `>=5.13`, so a
  valid install would resolve and then die inside `from_pretrained` with an
  unrecognised-architecture error. Both extras now require `>=5.13`, and
  `nixorb.hf.require_transformers()` reports the installed version and the
  upgrade command instead.
- **Python 3.12**: `requires-python` claimed 3.12 but nothing verified it.
  The suite now runs on 3.12; 3.13 added to the classifiers.
- A module-level `importorskip("transformers")` in the backend tests was
  skipping all 69 of them — including selection and pure-helper tests that
  never touch transformers — on any machine without it.
- `Speaker.start()` called `engine.stop()` on an idle engine; an engine
  entitled to treat `stop()` as terminal would then stay silent forever.
- Streaming chunk boundaries use the feature extractor's `win_length` (what
  the processor derives its chunk sizes from) rather than `n_fft`. They
  coincide on the released checkpoint; a fine-tune with a different window
  would have desynced every chunk.
- Startup logged "Ollama ready" and "Querying LLM: llama3.2" regardless of
  which backend was actually running.
- A runaway transcript was logged in full, turning one line into a screenful.

### Removed

`nixorb/llm/backends.py`, `nixorb/tts/glados_tts.py` and
`nixorb/tts/openai_tts.py`. All three were unreachable from the running app
and referenced settings that no longer exist; `openai_tts` also contradicted
the local-only design. `OfflineFallbackManager` moved to `nixorb/llm/factory.py`.

## [2.0.2] — 2026-08-27

### Changed
- **Redrew the orb.** The sphere is now a real radial-gradient render (Canvas)
  with an off-axis key light, limb darkening, a bounce light along the lower
  limb and a soft specular — instead of stacked flat circles with a linear
  gradient. It is painted once per state change; the bloom, pulse and audio
  ring animate as plain transforms, so nothing repaints at 60fps.
- **Smaller**: default `orb_size` 120 → 88.
- **Themed**: the palette is anchored on Claude's clay, so the orb reads as
  one family at rest — clay (idle), sage (listening), amber (thinking), lit
  clay (speaking), rust (error). The tray icon is now a shaded sphere in the
  same palette rather than a flat cyan dot, and tracks the state.
- Dropped the state caption: at 88px it collided with the audio ring and was
  illegible on a light desktop. The tray tooltip carries the state in words.

### Added
- **`nixorb trigger` actually works.** It was a stub that printed "not yet
  implemented", which meant the documented KDE global-shortcut setup did
  nothing. Added a Unix-socket control channel in `$XDG_RUNTIME_DIR`
  (`nixorb/core/ipc.py`); `nixorb trigger` activates the running orb,
  `nixorb quit` shuts it down, and `nixorb status` reports whether an
  instance is up. No D-Bus dependency, works under Wayland, X11 and a TTY.
- Starting a second instance is now refused — two orbs would fight over the
  microphone.

### Fixed
- **ChromaDB ran on the event loop mid-turn.** `build_context_block()` and
  `store()` do ONNX embedding inference synchronously; on the first query
  that froze the whole app for ~14 seconds (measured), taking the control
  socket down with it. Both now run off-thread.
- An IPC handler that raised closed the connection without a reply, leaving
  the client with an empty string; it now answers with the error.
- `IPCServer.start()` probed for a live peer with a blocking call from inside
  its own loop, so it always concluded "nobody home" and would have unlinked
  a live socket. The probe is async now.

### Changed — packaging
- Piper now comes from the AUR's **`piper-tts`** package rather than
  `piper-tts-bin`. Its binary is `piper-tts`, and NixOrb prefers that name:
  Arch's `piper` package is the gaming-mouse configuration tool, so testing
  for a bare `piper` finds unrelated software. Plain `piper` is still
  accepted as a fallback for pip installs and other distros.
- Voice lookup understands the AUR `piper-voices` layout
  (`<lang>/<locale>/<name>/<quality>/`), so an installed voice package is
  found instead of being re-downloaded.
- `install.sh` skips the HuggingFace voice download when the AUR package
  already provides the voice, and no longer aborts the install if Piper
  fails to build — it falls back to espeak-ng and says so.

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
