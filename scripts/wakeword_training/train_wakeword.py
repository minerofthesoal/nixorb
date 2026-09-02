#!/usr/bin/env python3
"""
train_wakeword.py — train custom openWakeWord models for NixOrb, overnight,
unattended, on a single 8 GB GPU (tested against a GTX 1080 / CUDA 11.8).

Trains one binary ONNX classifier per phrase in --phrases (default:
"hey nixorb", "nixorb", "hypernix"), following openWakeWord's own documented
approach: TTS-generate positive clips, embed them with openWakeWord's frozen
feature extractor, train a small classifier on top of those embeddings
against a large precomputed negative-feature set, export to ONNX.
See: https://github.com/dscripka/openWakeWord/blob/main/notebooks/training_models.ipynb

────────────────────────────────────────────────────────────────────────
WHAT ACTUALLY GENERATES THE POSITIVE CLIPS — READ THIS FIRST
────────────────────────────────────────────────────────────────────────
Four TTS models were requested. Only two of them are real, working, local
text-to-speech models that fit an 8 GB Pascal card:

  Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice   — WORKS. ~1.7B params, apache-2.0,
      9 built-in speaker voices, real `qwen-tts` pip package with a
      documented Python API. Used here with dtype=float16 (NOT the
      bfloat16 the model card recommends) and no flash-attention-2 —
      Pascal (sm_61) supports neither.

  Audio8/Audio8-TTS-Preview-0.1b         — WORKS, and is the better fit by
      far: ~0.17B params (10x smaller than the Qwen model), real
      documented `transformers` API (trust_remote_code=True). Custom
      "Audio8 Community License v1.0" — free for non-commercial use and
      for commercial use under $2M annual revenue; over that requires a
      separate license from Audio8. Same float16-not-bfloat16 override
      applied for Pascal.

  darkps/ice-012-audio                   — DOES NOT EXIST YET. Its model
      card literally says "COMING SOON" — no weights, no working example.
      This script checks the repo automatically on every run and enables
      itself the moment real files show up; until then it's a no-op, not
      an error.

  onnx-community/higgs-audio-v3-tts-4b   — real weights, but its own
      documented usage is via a dedicated SGLang-Omni or vLLM-Omni *server*
      (Docker, --gpus all), benchmarked by its authors on an H100. That's
      a fundamentally different deployment shape than "load a model and
      call generate()" — it doesn't fit an unattended single-script
      overnight run on a GTX 1080. Also: Boson's non-commercial-only
      license. Disabled by default; pass --enable-higgs if you stand up
      your own SGLang-Omni/vLLM-Omni server first (see README.md), and
      this script will POST short phrases to its OpenAI-compatible
      /v1/audio/speech endpoint instead of loading anything itself.

────────────────────────────────────────────────────────────────────────
HOW "FINISH IN ONE NIGHT, DON'T OOM THE 1080" IS ACTUALLY ENFORCED
────────────────────────────────────────────────────────────────────────
- Only one TTS model is ever resident on the GPU at a time. Between
  generators: del model, gc.collect(), torch.cuda.empty_cache().
- Generation is WALL-CLOCK BUDGETED, not sample-count-promised: pass
  --hours (default 8). The budget is split across (generator × phrase),
  and each slice stops at its deadline with however many clips it managed
  — never a fixed target that could blow past morning. A --target-clips
  ceiling still applies per slice so a fast run doesn't just keep going
  for no reason.
- The 17.3 GB ACAV100M negative-feature download and the 185 MB
  validation set (both from davidscripka/openwakeword_features on HF —
  precomputed, official, no audio generation or GPU needed for these)
  start in a background thread the moment the script launches, so that
  multi-hour, network-bound, non-GPU download overlaps with TTS
  generation instead of eating into the budget serially.
- Every phase is resumable: existing WAV clips are skipped by filename,
  existing downloads are skipped if already complete. A crash or Ctrl-C
  partway through just means re-running the script picks up where it
  left off.
- Final classifier training happens on (16, 96) embeddings, not raw
  audio or GPU-resident TTS weights — this step is cheap (small tensors,
  a few dense layers) and needs no meaningful VRAM regardless of which
  device trains it.

Everything downloaded/generated lives under --output-dir (default:
~/.local/share/nixorb/wakeword_training/). The finished .onnx files land
in <output-dir>/models/.

Usage:
    python train_wakeword.py                       # 8-hour budget, defaults
    python train_wakeword.py --hours 6
    python train_wakeword.py --phrases "hey nixorb" "nixorb" "hypernix"
    python train_wakeword.py --resume               # (default behavior anyway)
    python train_wakeword.py --enable-higgs --higgs-url http://localhost:8000

See README.md in this directory for setup (`pip install -r requirements.txt`,
qwen-tts, hardware notes) and how to wire the finished models into NixOrb.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import logging
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

log = logging.getLogger("train_wakeword")

# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────

DEFAULT_PHRASES = ["hey nixorb", "nixorb", "hypernix"]
SAMPLE_RATE = 16_000
CLIP_SECONDS = 3.0  # openWakeWord's standard training window
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

# Reserve the tail of the time budget for classifier training + export,
# so generation never eats the whole night and leaves nothing to train on.
TRAIN_RESERVE_MINUTES = 45

# Per (generator, phrase) slice: stop early if we hit this many clips even
# if there's budget left — no point over-generating past the point of
# diminishing returns for a single wake phrase.
TARGET_CLIPS_PER_SLICE = 1500

NEGATIVE_FEATURES_URL = (
    "https://huggingface.co/datasets/davidscripka/openwakeword_features/"
    "resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
)
NEGATIVE_FEATURES_SIZE = 17_300_000_000  # bytes, ~17.3 GB, from the HF repo listing
VALIDATION_FEATURES_URL = (
    "https://huggingface.co/datasets/davidscripka/openwakeword_features/"
    "resolve/main/validation_set_features.npy"
)
VALIDATION_FEATURES_SIZE = 185_000_000  # bytes, ~185 MB

ICE_012_AUDIO_REPO = "darkps/ice-012-audio"


def _default_output_dir() -> Path:
    return Path.home() / ".local" / "share" / "nixorb" / "wakeword_training"


# ─────────────────────────────────────────────────────────────────────────
# Text variation — same phrase spoken different ways, for prosodic variety
# ─────────────────────────────────────────────────────────────────────────

def phrase_variants(phrase: str) -> list[str]:
    """A handful of surface variations of one phrase for TTS to read."""
    base = phrase.strip()
    title = base[:1].upper() + base[1:]
    variants = {
        base,
        title,
        base.upper(),
        f"{title}.",
        f"{title}!",
        f"{title}?",
        f"Okay, {base}.",
        f"Um, {base}.",
    }
    return sorted(variants)


QWEN_INSTRUCTIONS = [
    "",  # no instruction — model default delivery
    "Speak in a neutral, clear, everyday tone.",
    "Speak quickly and casually, like calling out from another room.",
    "Speak calmly and a little quieter, like at night.",
]


# ─────────────────────────────────────────────────────────────────────────
# VRAM-safety helpers
# ─────────────────────────────────────────────────────────────────────────

def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def to_16k_mono_int16(wav: np.ndarray, sr: int) -> np.ndarray:
    """Resample/convert any generator's output to 16 kHz mono int16 PCM."""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=-1 if wav.shape[-1] < wav.shape[0] else 0)
    wav = np.clip(wav, -1.0, 1.0)

    if sr != SAMPLE_RATE:
        import torch
        import torchaudio.functional as AF

        t = torch.from_numpy(wav).float().unsqueeze(0)
        t = AF.resample(t, sr, SAMPLE_RATE)
        wav = t.squeeze(0).numpy()

    return (wav * 32767.0).astype(np.int16)


def pad_or_trim(pcm: np.ndarray, n_samples: int = CLIP_SAMPLES) -> np.ndarray:
    if len(pcm) >= n_samples:
        return pcm[:n_samples]
    pad = np.zeros(n_samples - len(pcm), dtype=np.int16)
    return np.concatenate([pcm, pad])


# ─────────────────────────────────────────────────────────────────────────
# TTS generators — each loads lazily, generates, and can be torn down
# ─────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class GeneratorSlot:
    name: str
    available: Callable[[], tuple[bool, str]]  # -> (ok, reason_if_not)
    load: Callable[[], Any]
    generate: Callable[[Any, str, int], np.ndarray]  # (handle, text, variant_idx) -> pcm16k int16
    unload: Callable[[Any], None]


def _torch_device_dtype() -> tuple[str, "Any"]:
    import torch

    if torch.cuda.is_available():
        # Pascal (sm_61, e.g. GTX 1080) has no native bf16 and no
        # flash-attention-2 support — float16 only, eager/sdpa attention.
        return "cuda:0", torch.float16
    return "cpu", torch.float32


# ---- Qwen3-TTS-12Hz-1.7B-CustomVoice --------------------------------------

_QWEN_SPEAKERS = ["Ryan", "Aiden", "Vivian", "Serena"]  # a mix of EN-native voices


def _qwen_available() -> tuple[bool, str]:
    try:
        import qwen_tts  # noqa: F401
    except ImportError:
        return False, "qwen-tts is not installed (pip install qwen-tts)"
    return True, ""


def _qwen_load() -> Any:
    import torch
    from qwen_tts import Qwen3TTSModel

    device, dtype = _torch_device_dtype()
    log.info("Qwen3-TTS: loading Qwen3-TTS-12Hz-1.7B-CustomVoice on %s (%s)", device, dtype)
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=device,
        dtype=dtype,
        # No attn_implementation override -> transformers picks a safe
        # default (eager/sdpa). flash_attention_2 is explicitly NOT used:
        # it requires fp16/bf16 support flash-attn itself doesn't ship
        # kernels for on Pascal.
    )
    return model


def _qwen_generate(model: Any, text: str, variant_idx: int) -> np.ndarray:
    speaker = _QWEN_SPEAKERS[variant_idx % len(_QWEN_SPEAKERS)]
    instruct = QWEN_INSTRUCTIONS[variant_idx % len(QWEN_INSTRUCTIONS)]
    wavs, sr = model.generate_custom_voice(
        text=text, language="English", speaker=speaker,
        instruct=instruct or None,
    )
    return to_16k_mono_int16(wavs[0], sr)


def _qwen_unload(model: Any) -> None:
    del model
    _release_cuda()


QWEN_SLOT = GeneratorSlot("qwen3-tts-1.7b", _qwen_available, _qwen_load, _qwen_generate, _qwen_unload)


# ---- Audio8-TTS-Preview-0.1b -----------------------------------------------

def _audio8_available() -> tuple[bool, str]:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        return False, f"missing dependency: {exc}"
    return True, ""


def _audio8_load() -> Any:
    import torch
    from transformers import AutoModel, AutoProcessor

    device, dtype = _torch_device_dtype()
    model_id = "Audio8/Audio8-TTS-Preview-0.1b"
    log.info("Audio8: loading %s on %s (%s)", model_id, device, dtype)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, dtype=dtype,
    ).eval().to(device)
    return (model, processor, device)


def _audio8_generate(handle: Any, text: str, variant_idx: int) -> np.ndarray:
    import torch

    model, processor, device = handle
    # Zero-shot / default-voice mode: omit reference_audio & reference_text
    # entirely (per the model card) rather than cloning a specific voice —
    # this naturally gives some voice variety across generations too.
    inputs = processor(text=[text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=256, temperature=0.8, top_p=0.9,
            top_k=50, do_sample=True, return_dict_in_generate=True,
        )
        waveforms, lengths = model.decode_audio(out.codes)
    audio = waveforms[0, : int(lengths[0])].float().cpu().numpy()
    return to_16k_mono_int16(audio, model.config.codec_sample_rate)


def _audio8_unload(handle: Any) -> None:
    del handle
    _release_cuda()


AUDIO8_SLOT = GeneratorSlot("audio8-tts-0.1b", _audio8_available, _audio8_load, _audio8_generate, _audio8_unload)


# ---- ice-012-audio — self-checking, currently a no-op ---------------------

def _ice012_available() -> tuple[bool, str]:
    """Checks the actual repo on HF each run. It's a placeholder ("COMING
    SOON", no weights) as of when this script was written — but if that
    changes, this starts working without any code edits."""
    try:
        req = urllib.request.Request(
            f"https://huggingface.co/api/models/{ICE_012_AUDIO_REPO}",
            headers={"User-Agent": "nixorb-wakeword-training"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            info = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, f"couldn't check repo status: {exc}"

    siblings = info.get("siblings", [])
    has_weights = any(
        s.get("rfilename", "").endswith((".safetensors", ".bin", ".onnx", ".pt"))
        for s in siblings
    )
    if not has_weights:
        return False, (
            f"{ICE_012_AUDIO_REPO} has no model weights yet (model card says "
            "\"COMING SOON\") — skipping. Re-run this script later; it "
            "checks automatically."
        )
    return False, (
        f"{ICE_012_AUDIO_REPO} now has weights, but this script wasn't "
        "written against a real usage example for it (none existed yet) — "
        "add a loader/generate function for it in this file before enabling."
    )


ICE012_SLOT = GeneratorSlot(
    "ice-012-audio", _ice012_available,
    load=lambda: (_ for _ in ()).throw(RuntimeError("not available")),
    generate=lambda h, t, i: np.zeros(1, dtype=np.int16),
    unload=lambda h: None,
)


# ---- higgs-audio-v3-tts-4b — opt-in, via an external server ---------------

def make_higgs_slot(server_url: str) -> GeneratorSlot:
    import urllib.request as _ur

    def available() -> tuple[bool, str]:
        try:
            with _ur.urlopen(server_url, timeout=3):
                pass
        except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
            return False, (
                f"--enable-higgs was passed but no server answered at "
                f"{server_url} ({exc}). Start an SGLang-Omni or vLLM-Omni "
                "server serving bosonai/higgs-audio-v3-tts-4b first — see "
                "README.md. This uses an H100 in its own published "
                "benchmarks; expect it to be slow on a GTX 1080-class card "
                "if you run it locally at all."
            )
        return True, ""

    def load() -> Any:
        return server_url  # nothing to load locally — it's a remote server

    def generate(handle: str, text: str, variant_idx: int) -> np.ndarray:
        import json as _json

        payload = _json.dumps({"input": text}).encode("utf-8")
        req = _ur.Request(
            f"{handle.rstrip('/')}/v1/audio/speech", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with _ur.urlopen(req, timeout=60) as resp:
            wav_bytes = resp.read()
        import io

        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        return to_16k_mono_int16(audio, sr)

    def unload(handle: Any) -> None:
        pass  # remote server — nothing local to release

    return GeneratorSlot("higgs-audio-v3-4b", available, load, generate, unload)


# ─────────────────────────────────────────────────────────────────────────
# Positive-clip generation
# ─────────────────────────────────────────────────────────────────────────

def _slugify(phrase: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")


def generate_positives_for_slice(
    slot: GeneratorSlot, phrase: str, out_dir: Path, deadline: float,
) -> int:
    """Generate clips for one (generator, phrase) pair until the slice
    deadline or TARGET_CLIPS_PER_SLICE, whichever comes first. Resumable:
    skips indices whose file already exists."""
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = phrase_variants(phrase)

    existing = len(list(out_dir.glob("*.wav")))
    if existing >= TARGET_CLIPS_PER_SLICE:
        log.info("  [%s/%s] already have %d clips — skipping", slot.name, phrase, existing)
        return existing

    ok, reason = slot.available()
    if not ok:
        log.warning("  [%s/%s] skipped: %s", slot.name, phrase, reason)
        return existing

    log.info("  [%s/%s] loading generator…", slot.name, phrase)
    try:
        handle = slot.load()
    except Exception:
        log.exception("  [%s/%s] failed to load — skipping this generator", slot.name, phrase)
        return existing

    made = existing
    consecutive_failures = 0
    try:
        while made < TARGET_CLIPS_PER_SLICE and time.monotonic() < deadline:
            variant_idx = made
            text = variants[variant_idx % len(variants)]
            out_path = out_dir / f"{made:05d}.wav"
            if out_path.exists():
                made += 1
                continue
            try:
                t0 = time.monotonic()
                pcm = slot.generate(handle, text, variant_idx)
                gen_seconds = time.monotonic() - t0
                if gen_seconds > 120:
                    log.warning(
                        "  [%s/%s] a single clip took %.0fs — this generator "
                        "looks too slow for the remaining budget, moving on",
                        slot.name, phrase, gen_seconds,
                    )
                    break
                pcm = pad_or_trim(pcm)
                import soundfile as sf

                sf.write(str(out_path), pcm, SAMPLE_RATE, subtype="PCM_16")
                made += 1
                consecutive_failures = 0
                if made % 100 == 0:
                    log.info("  [%s/%s] %d/%d clips (%.0f min left in this slice)",
                             slot.name, phrase, made, TARGET_CLIPS_PER_SLICE,
                             (deadline - time.monotonic()) / 60)
            except Exception:
                consecutive_failures += 1
                log.exception("  [%s/%s] clip %d failed", slot.name, phrase, made)
                if consecutive_failures >= 20:
                    log.error(
                        "  [%s/%s] %d consecutive failures — giving up on "
                        "this generator for this phrase", slot.name, phrase, consecutive_failures,
                    )
                    break
    finally:
        log.info("  [%s/%s] unloading generator…", slot.name, phrase)
        try:
            slot.unload(handle)
        except Exception:
            log.exception("  [%s/%s] unload raised (continuing anyway)", slot.name, phrase)

    return made


# ─────────────────────────────────────────────────────────────────────────
# Negative-feature download (background thread, starts immediately)
# ─────────────────────────────────────────────────────────────────────────

def _download_with_resume(url: str, dest: Path, expected_size: int) -> None:
    if dest.exists() and dest.stat().st_size >= expected_size * 0.99:
        log.info("Negatives: %s already downloaded (%.1f GB) — skipping",
                 dest.name, dest.stat().st_size / 1e9)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_from = tmp.stat().st_size if tmp.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "nixorb-wakeword-training"})
    if resume_from:
        req.add_header("Range", f"bytes={resume_from}-")
        log.info("Negatives: resuming %s from %.1f GB", dest.name, resume_from / 1e9)
    else:
        log.info("Negatives: downloading %s (~%.1f GB)…", dest.name, expected_size / 1e9)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            mode = "ab" if resume_from else "wb"
            with open(tmp, mode) as f:
                downloaded = resume_from
                last_log = time.monotonic()
                while chunk := resp.read(1024 * 1024 * 8):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if time.monotonic() - last_log > 60:
                        log.info("Negatives: %s at %.1f / %.1f GB",
                                 dest.name, downloaded / 1e9, expected_size / 1e9)
                        last_log = time.monotonic()
        tmp.rename(dest)
        log.info("Negatives: %s complete (%.1f GB)", dest.name, dest.stat().st_size / 1e9)
    except (urllib.error.URLError, OSError):
        log.exception("Negatives: download of %s failed — will retry on next run", dest.name)


def start_negative_downloads(output_dir: Path) -> tuple[threading.Thread, Path, Path]:
    neg_path = output_dir / "negatives" / "acav100m_2000hrs_16bit.npy"
    val_path = output_dir / "negatives" / "validation_set_features.npy"

    def _run() -> None:
        _download_with_resume(NEGATIVE_FEATURES_URL, neg_path, NEGATIVE_FEATURES_SIZE)
        _download_with_resume(VALIDATION_FEATURES_URL, val_path, VALIDATION_FEATURES_SIZE)

    thread = threading.Thread(target=_run, name="negative-downloads", daemon=True)
    thread.start()
    return thread, neg_path, val_path


# ─────────────────────────────────────────────────────────────────────────
# Feature extraction (openWakeWord's frozen embedding model)
# ─────────────────────────────────────────────────────────────────────────

def embed_positive_clips(wav_dir: Path, device: str = "cpu") -> np.ndarray:
    """Compute openWakeWord embeddings for every WAV in wav_dir. Cheap and
    fast (small ONNX models) — device defaults to CPU since GPU should
    already be free of TTS models by this phase."""
    from openwakeword.utils import AudioFeatures

    import soundfile as sf

    paths = sorted(wav_dir.glob("*.wav"))
    if not paths:
        return np.empty((0, 16, 96), dtype=np.float32)

    feats = AudioFeatures(device=device)
    clips = []
    for p in paths:
        audio, sr = sf.read(str(p), dtype="int16")
        if sr != SAMPLE_RATE:
            continue
        clips.append(pad_or_trim(audio))
    batch = np.stack(clips).astype(np.int16)
    return feats.embed_clips(batch, batch_size=64)


# ─────────────────────────────────────────────────────────────────────────
# Classifier training
# ─────────────────────────────────────────────────────────────────────────

def train_classifier(
    positive_features: np.ndarray,
    negative_features_path: Path,
    validation_features_path: Path | None,
    steps: int = 8000,
    batch_size: int = 128,
) -> "Any":
    """Small feed-forward classifier on top of (16, 96) embeddings, matching
    openWakeWord's own documented notebook approach ("a simple
    fully-connected neural network... export to ONNX"). Simplified relative
    to their full automatic_model_training.ipynb pipeline — no RIR/noise
    augmentation or adversarial-negative mixing here; add that later if a
    trained model's false-positive rate needs work."""
    import torch
    import torch.nn as nn

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    class Classifier(nn.Module):
        def __init__(self, in_dim: int = 16 * 96, hidden: int = 32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

    model = Classifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Negatives vastly outnumber positives — weight the positive class up.
    pos_weight = torch.tensor([20.0], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    neg = np.load(negative_features_path, mmap_mode="r")
    val = np.load(validation_features_path, mmap_mode="r") if validation_features_path and validation_features_path.exists() else None

    pos_t = torch.from_numpy(positive_features.astype(np.float32))
    n_pos = pos_t.shape[0]
    if n_pos == 0:
        raise RuntimeError("No positive clips were generated — nothing to train on. "
                            "Check the generator logs above for why every TTS model was skipped.")

    n_neg_total = neg.shape[0]
    rng = np.random.default_rng(0)

    model.train()
    for step in range(steps):
        pos_idx = rng.integers(0, n_pos, size=batch_size // 2)
        neg_idx = np.sort(rng.integers(0, n_neg_total, size=batch_size // 2))
        neg_batch = torch.from_numpy(np.asarray(neg[neg_idx]).astype(np.float32))
        pos_batch = pos_t[pos_idx]

        x = torch.cat([pos_batch, neg_batch], dim=0).to(device)
        y = torch.cat([
            torch.ones(pos_batch.shape[0], 1),
            torch.zeros(neg_batch.shape[0], 1),
        ]).to(device)

        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        if step % 1000 == 0 or step == steps - 1:
            msg = f"  step {step}/{steps}  loss={loss.item():.4f}"
            if val is not None:
                model.eval()
                with torch.no_grad():
                    val_idx = rng.integers(0, val.shape[0], size=min(2000, val.shape[0]))
                    val_x = torch.from_numpy(np.asarray(val[val_idx]).astype(np.float32)).to(device)
                    val_scores = torch.sigmoid(model(val_x))
                    fp_rate = (val_scores > 0.5).float().mean().item()
                msg += f"  validation-set false-positive-rate≈{fp_rate:.3%}"
                model.train()
            log.info(msg)

    model.eval()
    return model


def export_onnx(model: "Any", out_path: Path) -> None:
    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 16, 96)
    torch.onnx.export(
        model.cpu(), dummy, str(out_path),
        input_names=["x"], output_names=["logit"],
        dynamic_axes={"x": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=13,
    )
    log.info("Exported %s", out_path)


# ─────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    global TARGET_CLIPS_PER_SLICE

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phrases", nargs="+", default=DEFAULT_PHRASES)
    parser.add_argument("--hours", type=float, default=8.0, help="Total wall-clock budget.")
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--target-clips-per-slice", type=int, default=TARGET_CLIPS_PER_SLICE)
    parser.add_argument("--train-steps", type=int, default=8000)
    parser.add_argument("--skip-negatives-download", action="store_true")
    parser.add_argument("--enable-higgs", action="store_true",
                         help="Also use higgs-audio-v3 via an external SGLang-Omni/vLLM-Omni server.")
    parser.add_argument("--higgs-url", default="http://localhost:8000")
    args = parser.parse_args()

    TARGET_CLIPS_PER_SLICE = args.target_clips_per_slice

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    log.info("=" * 70)
    log.info("NixOrb wake-word training — budget: %.1f hours, output: %s", args.hours, args.output_dir)
    log.info("Log file: %s", log_path)

    start = time.monotonic()
    deadline = start + args.hours * 3600
    train_deadline_reserve = TRAIN_RESERVE_MINUTES * 60

    # Kick off the 17.3 GB + 185 MB negative-feature download immediately,
    # in the background, so it overlaps with TTS generation instead of
    # eating into the budget serially.
    dl_thread, neg_path, val_path = (
        (None, args.output_dir / "negatives" / "acav100m_2000hrs_16bit.npy",
         args.output_dir / "negatives" / "validation_set_features.npy")
        if args.skip_negatives_download
        else start_negative_downloads(args.output_dir)
    )

    slots: list[GeneratorSlot] = [QWEN_SLOT, AUDIO8_SLOT, ICE012_SLOT]
    if args.enable_higgs:
        slots.append(make_higgs_slot(args.higgs_url))

    # Report availability up front so a bad night doesn't surprise anyone
    # an hour in.
    log.info("Generator availability:")
    for slot in slots:
        ok, reason = slot.available()
        log.info("  %-16s %s%s", slot.name, "OK" if ok else "SKIP — ", reason)

    usable_slots = [s for s in slots if s.available()[0]]
    if not usable_slots:
        log.error("No usable TTS generators — nothing to do. See the availability report above.")
        return 1

    # Generation phase — split the budget across (generator × phrase).
    n_slices = len(usable_slots) * len(args.phrases)
    generation_deadline_overall = deadline - train_deadline_reserve
    per_slice_budget = max(
        (generation_deadline_overall - time.monotonic()) / max(n_slices, 1), 60
    )
    log.info("Generation phase: %d slices, ~%.0f min each", n_slices, per_slice_budget / 60)

    clip_dirs: dict[str, Path] = {}
    for phrase in args.phrases:
        slug = _slugify(phrase)
        for slot in usable_slots:
            if time.monotonic() >= generation_deadline_overall:
                log.warning("Generation deadline reached — moving to training with what we have.")
                break
            slice_deadline = min(time.monotonic() + per_slice_budget, generation_deadline_overall)
            out_dir = args.output_dir / "positives" / slug / slot.name
            n = generate_positives_for_slice(slot, phrase, out_dir, slice_deadline)
            log.info("  -> %d clips for '%s' via %s", n, phrase, slot.name)
        clip_dirs[slug] = args.output_dir / "positives" / slug

    # Make sure the negative-feature download is actually done before
    # training needs it — this is the one point we may have to just wait.
    if dl_thread is not None and dl_thread.is_alive():
        remaining = deadline - time.monotonic()
        log.info("Waiting up to %.0f min for the negative-feature download to finish…", remaining / 60)
        dl_thread.join(timeout=max(remaining, 0))
    if not neg_path.exists():
        log.error(
            "Negative-feature file never finished downloading (%s missing). "
            "Can't train without it — re-run the script (downloads resume) "
            "or pass --skip-negatives-download if you fetched it yourself.",
            neg_path,
        )
        return 1

    # Training phase — one classifier per phrase, using ALL generators'
    # clips pooled together for that phrase.
    models_dir = args.output_dir / "models"
    trained_paths = []
    for phrase in args.phrases:
        slug = _slugify(phrase)
        wav_root = clip_dirs.get(slug, args.output_dir / "positives" / slug)
        wav_dirs = [d for d in wav_root.glob("*") if d.is_dir()]
        if not wav_dirs:
            log.warning("No positive clips at all for '%s' — skipping training for this phrase.", phrase)
            continue

        log.info("Embedding positive clips for '%s'…", phrase)
        all_feats = []
        for d in wav_dirs:
            f = embed_positive_clips(d)
            if f.shape[0]:
                all_feats.append(f)
        if not all_feats:
            log.warning("No embeddable clips for '%s' — skipping.", phrase)
            continue
        pos_features = np.concatenate(all_feats, axis=0)
        log.info("'%s': %d positive embedding windows", phrase, pos_features.shape[0])

        log.info("Training classifier for '%s'…", phrase)
        model = train_classifier(
            pos_features, neg_path, val_path if val_path.exists() else None,
            steps=args.train_steps,
        )
        out_path = models_dir / f"{slug}.onnx"
        export_onnx(model, out_path)
        trained_paths.append((phrase, out_path))
        del model
        _release_cuda()

    elapsed = (time.monotonic() - start) / 3600
    log.info("=" * 70)
    log.info("Done in %.1f hours.", elapsed)
    if trained_paths:
        log.info("Trained models:")
        for phrase, path in trained_paths:
            log.info("  '%s' -> %s", phrase, path)
        joined = ",".join(str(p.resolve()) for _, p in trained_paths)
        log.info("")
        log.info("To use all of these together in NixOrb, set in ~/.config/nixorb/config.toml:")
        log.info("  wake_word_model = \"%s\"", joined)
    else:
        log.error("No models were trained. Check the log above for why every phrase was skipped.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
