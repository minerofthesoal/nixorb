"""
nixorb/tts/glados_tts.py — GLaDOS-style TTS using torphix/stablelm-2-glados-v1.

The torphix/stablelm-2-glados-v1 model on HuggingFace is a StableLM model
fine-tuned with GLaDOS personality. For TTS we use the model's text-to-speech
pipeline if available, otherwise fall back to SpeechT5 with a custom voice.

If you want a true GLaDOS *voice*, install and configure piper-tts with
a GLaDOS voice model (several are available in the community).
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)


class GladosTTS:
    """
    GLaDOS-style TTS. Uses the pipeline from the HF model if it exposes
    a TTS head; otherwise falls back to SpeechT5 + vocoder with a
    neutral voice and slightly robotic pitch.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipeline = None
        self._loaded   = False

    def _load(self):
        try:
            from transformers import pipeline as hf_pipeline
            # Try TTS pipeline directly
            pipe = hf_pipeline(
                "text-to-speech",
                model=self._settings.tts_hf_repo,
                token=self._settings.hf_token or None,
                device=0 if _has_cuda() else -1,
            )
            log.info("GladosTTS: pipeline loaded from %s", self._settings.tts_hf_repo)
            return ("pipeline", pipe)
        except Exception as exc:
            log.warning("GladosTTS: TTS pipeline unavailable (%s), using SpeechT5", exc)
            return _load_speecht5(self._settings.hf_token)

    async def speak(self, text: str) -> None:
        await bus.emit(Event.TTS_START, source="GladosTTS")
        loop = asyncio.get_running_loop()
        try:
            if not self._loaded:
                self._pipeline = await loop.run_in_executor(None, self._load)
                self._loaded   = True

            pcm = await loop.run_in_executor(None, self._synthesise, text)
            if pcm is not None:
                await bus.emit(
                    Event.TTS_AUDIO_CHUNK, data={"pcm": pcm}, source="GladosTTS"
                )
                await loop.run_in_executor(None, _play, pcm)
        except Exception:
            log.exception("GladosTTS synthesis failed")
        finally:
            await bus.emit(Event.TTS_DONE, source="GladosTTS")

    def _synthesise(self, text: str) -> bytes | None:
        if self._pipeline is None:
            return None
        kind, pipe = self._pipeline
        if kind == "pipeline":
            out = pipe(text)
            arr = np.array(out["audio"], dtype=np.float32)
            if arr.ndim > 1:
                arr = arr[0]
            return (arr * 32767).astype(np.int16).tobytes()
        # SpeechT5 path
        return _synthesise_speecht5(pipe, text)


# ── Helpers ───────────────────────────────────────────────────────── #

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _load_speecht5(token: str):
    from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan
    from datasets import load_dataset
    import torch

    device = "cuda" if _has_cuda() else "cpu"
    proc  = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", token=token or None)
    model = SpeechT5ForTextToSpeech.from_pretrained(
        "microsoft/speecht5_tts", token=token or None
    ).to(device)
    voc   = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan", token=token or None
    ).to(device)
    embs  = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    # speaker 7306 = neutral/robotic sounding
    spk   = torch.tensor(embs[7306]["xvector"]).unsqueeze(0).to(device)
    return ("speecht5", (proc, model, voc, spk))


def _synthesise_speecht5(pipe, text: str) -> bytes | None:
    import torch
    proc, model, voc, spk = pipe
    inputs = proc(text=text, return_tensors="pt").to(spk.device)
    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"], spk, vocoder=voc)
    arr = speech.cpu().numpy()
    return (arr * 32767).astype(np.int16).tobytes()


def _play(pcm: bytes, sr: int = 16000) -> None:
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(arr, samplerate=sr, blocking=True)
    sd.wait()
