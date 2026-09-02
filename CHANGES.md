# NixOrb changes — mic fixes, custom model support, orb redesign

## 1. Microphone — three stacked bugs, all fixed

Your log showed the wake word crashing and dictation reporting "no speech
detected" after 30s. Both traced back to real, distinct bugs:

- **`nixorb/asr/wake_word.py`** called `Model(wakeword_models=[...])`. The
  installed openwakeword renamed that constructor argument to
  `wakeword_model_paths`, and it expects file *paths*, not bare names — the
  old kwarg fell through into an internal `AudioFeatures` object, which is
  exactly the crash you saw.
- Even fixed, it would have stayed silent forever: openwakeword needs
  16-bit PCM integers, but the code fed it float32 audio normalized to
  [-1.0, 1.0] straight from `sounddevice`. Internally that gets cast with
  `.astype(np.int16)`, silently truncating every sample to 0.
- `"hey_nixorb"` was never a real trained model — openwakeword only ships
  `alexa`, `hey_jarvis`, `hey_mycroft`, `timer`, `weather`. It now falls
  back to `hey_jarvis` with a clear log line telling you how to train and
  point at your own model instead.
- Separately, `nixorb/asr/whisper_engine.py`'s hotkey-triggered dictation
  (what your log's 30-second "no speech detected" actually was) had a mic
  sensitivity slider in Settings wired to *nothing* — a hardcoded
  `SILENCE_THRESHOLD = 0.015` did all the work regardless of what you set.
  There was also no detection of PipeWire/PulseAudio monitor devices
  (loopbacks of your speaker output) masquerading as a microphone, which
  is one of the most common reasons "recording succeeds, hears nothing"
  happens on Linux.

**What changed:** new `nixorb/utils/audio.py` resolves and logs the actual
input device being used (rejecting monitor devices with a warning),
`mic_sensitivity` now really adjusts the VAD threshold, and both ASR
engines log the peak input level they actually saw when nothing was
detected — so if this still doesn't work on your hardware, the log will
tell you whether the mic was silent, too quiet, or the wrong device,
instead of just "no speech detected."

I couldn't test against your actual microphone/openwakeword-on-your-GPU in
this environment — I verified the fixes work correctly against the real
openwakeword package (constructor call, model resolution, int16
conversion all tested directly), but the true end-to-end test is on your
machine.

## 2. Custom HuggingFace models — LLM, ASR (incl. streaming), TTS

The actual reason your log showed a health check checking Ollama's model
list against a HuggingFace repo id: `main.py` always hardcoded
`OllamaBackend` and `PiperTTS`, completely ignoring `llm_backend` /
`tts_backend` in your settings. Changing those settings did nothing.

**New:**
- `nixorb/llm/factory.py` + `nixorb/llm/hf_llm_backend.py` — any local HF
  model, safetensors via transformers or GGUF via llama-cpp-python
  (auto-detected from the repo/filename), with generic `<tool_call>`-tag
  based tool calling so your existing plugins keep working.
- `nixorb/llm/openai_compat_backend.py` — any OpenAI-compatible endpoint
  (real OpenAI, or a local vLLM/TGI/LM Studio/llama.cpp server) — often
  the more practical way to serve a large custom HF model.
- `nixorb/asr/factory.py` + `nixorb/asr/hf_asr_engine.py` — any transformers
  ASR model/pipeline (distil-whisper, Parakeet, a fine-tune, anything),
  plus `asr_streaming` for chunk-and-retranscribe partial results while
  you're still talking.
- `nixorb/tts/tts_factory.py` is now actually wired into `main.py`, so
  `tts_backend = "glados"` / `"huggingface"` / `"openai"` all work.
- All three backends share their respective existing interface exactly
  (same event names, same `last_tool_calls`/`health_check`/`stream()`
  shape), so nothing else in the app needed to change.
- Fixed three missing `Settings` fields (`hf_token`, `openai_api_key`,
  `tts_hf_repo`) that existing-but-never-wired code already referenced —
  these would have thrown `AttributeError` the moment they were hit.
- Settings GUI (`nixorb/ui/settings_window.py`) now exposes backend
  choice + the new fields for ASR/LLM/TTS, and a real bug where selecting
  "espeak-ng" from the TTS dropdown would have crashed `build_tts` is
  fixed too (it's not a distinct engine — `PiperTTS` already falls back to
  espeak-ng on its own).

**New optional dependency extras** (`pyproject.toml`): `huggingface`,
`llama_cpp`, `openai` — install what you need, e.g.
`pip install '.[huggingface]'`.

## 3. Orb — redesigned from scratch, iOS Siri 2.0 style

`assets/orb.qml` is a full rewrite: dark glass base, additively-blended
iridescent colour blobs drifting around the rim, glass specular highlight,
state-driven palettes (cool blue/cyan idle, warm gold/pink/purple
thinking, full-spectrum speaking, red-rimmed error). I loaded this through
a real offscreen Qt renderer to check it (caught and fixed a property
name collision with Qt6's built-in `Item.palette`), and iterated on the
actual rendered output — renders are dark/glassy with concentrated rim
colour rather than a flat rainbow disc, matching the Dynamic Island
reference you sent, and error state stays visually distinct (red rim)
from the rest of the palette.

The old shader files under `assets/shaders/` were already dead code
(never wired into any QML) — left untouched, still unused.

## Testing

Ran the existing test suite throughout (107 passing; the only failures are
`test_memory.py`/parts of `test_core.py`, which fail in this sandbox
because chromadb's bundled embedding-model download is corrupted here —
unrelated to anything in this changeset). Added no new test dependencies
you don't already have. All modified/new Python files were syntax-checked
and the whole package byte-compiles cleanly.

## What I couldn't verify here (no real audio/GPU hardware in this sandbox)

- End-to-end mic capture and wake word detection on your actual GTX 1080 /
  microphone.
- A real HF model actually generating/transcribing/speaking end-to-end
  (the backends are correct against the real APIs, but I didn't have a
  multi-GB model to download and run in this environment).
- The orb rendered inside the real always-on-top floating window (I
  rendered it through Qt's offscreen platform, which exercises the same
  QML engine but not window compositing/transparency on your desktop).
