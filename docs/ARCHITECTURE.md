# NixOrb Architecture

## Threading Model

```
Main thread (Qt)          asyncio event loop         ThreadPoolExecutor
─────────────────         ──────────────────         ──────────────────
QApplication              EventBus dispatch          sounddevice record
OrbWindow (QML)    ←────  LLM chunk streaming   ←── Whisper inference
SettingsWindow     ←────  TTS audio chunks      ←── VRAM load/unload
NixOrbTray                VectorMemory queries       HF model loading
HotkeyManager(pynput)     Web search coroutines      executor blocking I/O
```

## Event Bus

All inter-component communication goes through `nixorb.core.event_bus.EventBus`.
No direct imports between subsystems at runtime.

Key events:
- `HOTKEY_TRIGGERED` / `WAKE_WORD_DETECTED` → start conversation turn
- `ORB_LISTENING` / `ORB_THINKING` / `ORB_SPEAKING` / `ORB_IDLE` → orb animation
- `LLM_CHUNK` → streamed text (forwarded to log widget)
- `TTS_AUDIO_CHUNK` → raw PCM (drives orb amplitude animation)
- `LOG` → forwarded to Settings log panel

## VRAM Paging (GTX 1080 8 GB)

```
Model           VRAM    Priority    Eviction order
──────────────  ──────  ────────    ──────────────
Whisper v3 INT8  ~2 GB  LOW         first
HF TTS           ~1.5GB MEDIUM      second
Local LLM        ~4 GB  HIGH        last
```

Flow: hotkey → evict LLM → load Whisper → transcribe →
      evict Whisper → load LLM → generate → evict LLM → TTS speaks

## QSocketNotifier Fix

`loop.add_signal_handler()` creates a Unix socketpair that Qt's event loop
doesn't own, producing the `QSocketNotifier: Can only be used with threads
started with QThread` warning.

Fix: use `app.aboutToQuit` signal + `signal.signal()` + 200ms asyncio poll.
No Unix socketpair, no Qt warning.
