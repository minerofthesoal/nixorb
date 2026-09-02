"""Generic Hugging Face text-to-speech — any TTS model on the Hub.

    tts_backend  = "huggingface"
    tts_hf_model = "microsoft/speecht5_tts"   # or MMS-TTS, Bark, Parler, VITS…

Three loading strategies, tried in order, because the Hub does not have one
TTS interface:

  1. `text-to-speech` pipeline — MMS-TTS, VITS, Bark, most of the Hub.
  2. SpeechT5 — needs a speaker x-vector, which the pipeline cannot supply
     on its own, so it gets an explicit path.
  3. `text-to-audio` pipeline — the older task name some repos still declare.

All three end up returning (waveform, sampling_rate), which is played
through sounddevice like every other NixOrb audio path.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from nixorb import hf
from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

DEFAULT_MODEL = "microsoft/speecht5_tts"

# SpeechT5 cannot speak without a 512-dim speaker embedding. This is the
# dataset the model card uses; index 7306 is its usual female example voice.
XVECTOR_DATASET = "Matthijs/cmu-arctic-xvectors"
DEFAULT_SPEAKER_INDEX = 7306

# Guard against a model that runs away generating audio forever.
MAX_CHARS = 600


def _is_speecht5(model_id: str) -> bool:
    return "speecht5" in model_id.lower()


def load_speaker_embedding(source: str | None, settings: Any = None) -> Any:
    """Load a SpeechT5 speaker x-vector.

    `source` may be a path to a .npy file, an integer index into the
    cmu-arctic-xvectors dataset, or blank for the default voice.
    """
    torch = hf.require("torch")

    source = (source or "").strip()
    if source and Path(source).expanduser().is_file():
        vector = np.load(Path(source).expanduser())
        log.info("TTS: speaker embedding from %s", source)
        return torch.tensor(vector).reshape(1, -1).float()

    index = DEFAULT_SPEAKER_INDEX
    if source:
        try:
            index = int(source)
        except ValueError:
            log.warning(
                "TTS: speaker %r is neither a .npy path nor an index — "
                "using the default voice", source,
            )

    try:
        datasets = hf.require("datasets")
        kwargs = {}
        tok = hf.token(settings)
        if tok:
            kwargs["token"] = tok
        embeddings = datasets.load_dataset(
            XVECTOR_DATASET, split="validation", **kwargs
        )
        vector = embeddings[index]["xvector"]
        log.info("TTS: speaker embedding %s[%d]", XVECTOR_DATASET, index)
        return torch.tensor(vector).unsqueeze(0)
    except Exception as exc:
        raise RuntimeError(
            f"SpeechT5 needs a speaker embedding and none could be loaded "
            f"({exc}). Set tts_hf_speaker to a .npy file, or install the "
            f"'datasets' package: pip install 'nixorb[hf]'"
        ) from exc


class HuggingFaceTTS:
    """Text-to-speech through any Hugging Face model."""

    name = "huggingface"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings
        self._model_id = (
            getattr(settings, "tts_hf_repo", "")
            or getattr(settings, "tts_hf_model", "")
            or DEFAULT_MODEL
        ) if settings else DEFAULT_MODEL
        self._speaker = getattr(settings, "tts_hf_speaker", "") if settings else ""
        # Voice-design models (Breeze-TTS-2) take the voice as a natural
        # language instruction rather than a named preset.
        self._voice = getattr(settings, "tts_voice", "") if settings else ""
        self._volume = float(getattr(settings, "tts_volume", 1.0) or 1.0)
        self._speed = float(getattr(settings, "tts_speed", 1.0) or 1.0)
        self._engine: Any = None
        self._stopped = False

    def stop(self) -> None:
        """Cut playback off mid-sentence (barge-in)."""
        self._stopped = True
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    @property
    def available(self) -> bool:
        """True if transformers is importable — the model loads on first use."""
        try:
            hf.require("transformers")
            return True
        except hf.MissingDependency:
            return False

    # ── Loading ──────────────────────────────────────────────────── #

    def _load(self) -> dict[str, Any]:
        transformers = hf.require("transformers")
        device = hf.resolve_device(getattr(self._settings, "hf_device", "auto"))
        kwargs = hf.load_kwargs(self._settings, device)

        if _is_speecht5(self._model_id):
            return self._load_speecht5(transformers, device, kwargs)

        for task in ("text-to-speech", "text-to-audio"):
            try:
                log.info("TTS: loading %s as %s on %s",
                         self._model_id, task, device)
                pipe = transformers.pipeline(
                    task,
                    model=self._model_id,
                    device=0 if device == "cuda" else -1,
                    # Voice-design models ship custom code in the repo. This
                    # executes it — the same call main.py's default config
                    # already relies on for Breeze-TTS-2.
                    trust_remote_code=True,
                    **{k: v for k, v in kwargs.items()
                       if k != "trust_remote_code"},
                )
                return {"kind": "pipeline", "pipe": pipe}
            except Exception as exc:
                log.debug("TTS: %s does not load as %s (%s)",
                          self._model_id, task, exc)

        raise RuntimeError(
            f"'{self._model_id}' could not be loaded as a text-to-speech "
            f"model. Check the repo declares a text-to-speech or "
            f"text-to-audio pipeline_tag."
        )

    def _load_speecht5(
        self, transformers: Any, device: str, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        log.info("TTS: loading SpeechT5 %s on %s", self._model_id, device)
        processor = transformers.SpeechT5Processor.from_pretrained(
            self._model_id, **kwargs
        )
        model = transformers.SpeechT5ForTextToSpeech.from_pretrained(
            self._model_id, **kwargs
        ).to(device)
        vocoder = transformers.SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan", **kwargs
        ).to(device)
        speaker = load_speaker_embedding(self._speaker, self._settings).to(device)

        return {
            "kind": "speecht5",
            "processor": processor,
            "model": model,
            "vocoder": vocoder,
            "speaker": speaker,
            "device": device,
        }

    # ── Synthesis ────────────────────────────────────────────────── #

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (waveform, sampling_rate). Blocking; runs in a thread."""
        if self._engine is None:
            self._engine = self._load()

        engine = self._engine
        if engine["kind"] == "speecht5":
            return self._synthesize_speecht5(engine, text)

        # A voice-design model takes the voice description per call.
        if self._voice and _looks_like_voice_instruction(self._voice):
            try:
                return _waveform_from(
                    engine["pipe"](text, forward_params={"instruction": self._voice})
                )
            except (TypeError, ValueError) as exc:
                log.debug("TTS: %s ignores an instruction (%s)",
                          self._model_id, exc)

        return _waveform_from(engine["pipe"](text))

    @staticmethod
    def _synthesize_speecht5(engine: dict[str, Any], text: str) -> tuple[np.ndarray, int]:
        import torch

        inputs = engine["processor"](text=text, return_tensors="pt").to(
            engine["device"]
        )
        with torch.no_grad():
            speech = engine["model"].generate_speech(
                inputs["input_ids"], engine["speaker"], vocoder=engine["vocoder"]
            )
        return speech.cpu().numpy().astype(np.float32), 16000

    # ── Playback ─────────────────────────────────────────────────── #

    async def speak(self, text: str) -> None:
        """Synthesize and play. Never raises — TTS failing must not kill a turn."""
        import asyncio

        if not text or not text.strip():
            return

        text = text.strip()[:MAX_CHARS]
        self._stopped = False
        log.info("TTS: speaking '%s…' via %s", text[:60], self._model_id)
        await bus.emit(Event.TTS_START, data={"text": text[:200]}, source=self.name)

        try:
            audio, rate = await asyncio.to_thread(self._synthesize, text)
        except Exception as exc:
            msg = f"Hugging Face TTS failed ({self._model_id}): {exc}"
            log.error("TTS: %s", msg)
            await bus.emit(Event.TTS_ERROR, data={"error": msg}, source=self.name)
            await bus.emit(
                Event.LOG, data={"level": "warning", "msg": f"🔇 {msg}"},
                source=self.name,
            )
            return

        await asyncio.to_thread(self._play, audio, rate)
        await bus.emit(Event.TTS_DONE, source=self.name)

    def _play(self, audio: np.ndarray, rate: int) -> None:
        if self._stopped:
            return
        try:
            import sounddevice as sd

            samples = np.asarray(audio, dtype=np.float32).squeeze()
            peak = float(np.max(np.abs(samples))) if samples.size else 0.0
            if peak > 1.0:
                samples = samples / peak  # some vocoders return unnormalised audio
            sd.play(samples * self._volume, samplerate=int(rate))
            sd.wait()
        except Exception as exc:
            log.error("TTS: playback failed: %s", exc)

    async def synthesize_to_file(self, text: str, output_path: Path) -> bool:
        """Write speech to a WAV file instead of playing it."""
        import asyncio

        try:
            audio, rate = await asyncio.to_thread(self._synthesize, text)
            import soundfile as sf

            await asyncio.to_thread(
                sf.write, str(output_path), np.asarray(audio, dtype=np.float32), rate
            )
            return True
        except Exception as exc:
            log.error("TTS: synthesize_to_file failed: %s", exc)
            return False


def _looks_like_voice_instruction(voice: str) -> bool:
    """True for a prose voice description, not a Piper voice id.

    tts_voice is shared with Piper, where it holds things like
    "en_US-lessac-medium"; sending that to a voice-design model as an
    instruction produces nonsense.
    """
    return " " in voice.strip()


def _waveform_from(result: Any) -> tuple[np.ndarray, int]:
    """Normalise the several shapes HF TTS pipelines return."""
    if isinstance(result, list) and result:
        result = result[0]

    if isinstance(result, dict):
        for key in ("audio", "waveform", "array"):
            if key in result:
                rate = int(result.get("sampling_rate") or result.get("rate") or 16000)
                return np.asarray(result[key], dtype=np.float32).squeeze(), rate

    raise RuntimeError(f"Unrecognised TTS output: {type(result).__name__}")
