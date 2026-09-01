"""tests/test_backends.py — backend selection for ASR, TTS and LLM.

NixOrb v0.3 can drive any ASR, TTS or causal LM on the Hugging Face Hub, plus
NVIDIA Nemotron 3.5 natively. These cover the selection logic and the pure
helpers; the model-loading paths need the models themselves and are exercised
are exercised where transformers is available.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nixorb.settings import Settings

# ── ASR selection ────────────────────────────────────────────────── #

def test_default_asr_is_faster_whisper():
    from nixorb.asr.factory import create_asr
    from nixorb.asr.whisper_engine import WhisperEngine

    assert isinstance(create_asr(Settings()), WhisperEngine)


def test_huggingface_asr_backend():
    from nixorb.asr.factory import create_asr
    from nixorb.asr.hf_asr import HFASREngine

    engine = create_asr(
        Settings(asr_backend="huggingface", asr_model="openai/whisper-small")
    )
    assert isinstance(engine, HFASREngine)
    assert engine.name == "huggingface"


def test_nemotron_asr_backend():
    from nixorb.asr.factory import create_asr
    from nixorb.asr.nemotron_asr import NemotronASREngine

    engine = create_asr(
        Settings(
            asr_backend="nemotron",
            asr_model="nvidia/nemotron-3.5-asr-streaming-0.6b",
        )
    )
    assert isinstance(engine, NemotronASREngine)
    assert engine.supports_streaming is True


def test_nemotron_model_id_selects_the_native_backend():
    """Pointing asr_model at Nemotron must not silently lose streaming."""
    from nixorb.asr.factory import create_asr
    from nixorb.asr.nemotron_asr import NemotronASREngine

    engine = create_asr(
        Settings(
            asr_backend="huggingface",
            asr_model="nvidia/nemotron-3.5-asr-streaming-0.6b",
        )
    )
    assert isinstance(engine, NemotronASREngine)


@pytest.mark.parametrize(
    "alias,expected",
    [("whisper", "faster-whisper"), ("hf", "huggingface"),
     ("transformers", "huggingface"), ("nvidia", "nemotron"),
     ("NEMOTRON", "nemotron"), (None, "faster-whisper")],
)
def test_asr_backend_aliases(alias, expected):
    from nixorb.asr.factory import normalise_backend

    assert normalise_backend(alias) == expected


def test_unknown_asr_backend_falls_back_loudly(caplog):
    from nixorb.asr.factory import create_asr
    from nixorb.asr.whisper_engine import WhisperEngine

    engine = create_asr(Settings(asr_backend="nonsense"))
    assert isinstance(engine, WhisperEngine)
    assert "nonsense" in caplog.text


# ── LLM selection ────────────────────────────────────────────────── #

def test_default_llm_is_ollama():
    from nixorb.llm.factory import create_llm
    from nixorb.llm.ollama_backend import OllamaBackend

    assert isinstance(create_llm(Settings()), OllamaBackend)


def test_huggingface_llm_backend():
    from nixorb.llm.factory import create_llm
    from nixorb.llm.hf_backend import HuggingFaceBackend

    llm = create_llm(Settings(llm_backend="huggingface",
                              llm_hf_model="Qwen/Qwen2.5-3B-Instruct"))
    assert isinstance(llm, HuggingFaceBackend)
    assert llm.model == "Qwen/Qwen2.5-3B-Instruct"


def test_unknown_llm_backend_falls_back_loudly(caplog):
    from nixorb.llm.factory import create_llm
    from nixorb.llm.ollama_backend import OllamaBackend

    assert isinstance(create_llm(Settings(llm_backend="gpt5")), OllamaBackend)
    assert "gpt5" in caplog.text


async def test_hf_llm_health_check_without_transformers():
    """A missing dependency must name the install command, not stack-trace."""
    from nixorb import hf
    from nixorb.llm.hf_backend import HuggingFaceBackend

    llm = HuggingFaceBackend(Settings(llm_backend="huggingface"))
    with patch.object(hf, "require",
                      side_effect=hf.MissingDependency("'torch' is not installed")):
        health = await llm.health_check()

    assert health["ok"] is False
    assert "torch" in health["error"]


# ── TTS selection ────────────────────────────────────────────────── #

def test_default_tts_is_piper():
    from nixorb.tts.piper_tts import PiperTTS
    from nixorb.tts.tts_factory import create_tts

    assert isinstance(create_tts(Settings()), PiperTTS)


def test_huggingface_tts_backend():
    from nixorb.tts.hf_tts import HuggingFaceTTS
    from nixorb.tts.tts_factory import create_tts

    with patch.object(HuggingFaceTTS, "available", True):
        engine = create_tts(
            Settings(tts_backend="huggingface", tts_hf_model="facebook/mms-tts-eng")
        )
    assert isinstance(engine, HuggingFaceTTS)
    assert engine._model_id == "facebook/mms-tts-eng"


def test_huggingface_tts_falls_back_when_transformers_is_missing(caplog):
    """Selecting an unavailable backend must not leave the orb mute."""
    from nixorb.tts.hf_tts import HuggingFaceTTS
    from nixorb.tts.piper_tts import PiperTTS
    from nixorb.tts.tts_factory import create_tts

    with patch.object(HuggingFaceTTS, "available", False):
        engine = create_tts(Settings(tts_backend="huggingface"))

    assert isinstance(engine, PiperTTS)
    assert "nixorb[hf]" in caplog.text


def test_espeak_backend_skips_piper():
    from nixorb.tts.tts_factory import create_tts

    engine = create_tts(Settings(tts_backend="espeak"))
    assert engine._piper_available is False


# ── Nemotron helpers ─────────────────────────────────────────────── #

@pytest.mark.parametrize(
    "given,expected",
    [("", "auto"), (None, "auto"), ("auto", "auto"), ("AUTO", "auto"),
     ("en-US", "en-US"), ("de", "de"), ("  fr-FR  ", "fr-FR")],
)
def test_normalise_language(given, expected):
    from nixorb.asr.nemotron_asr import normalise_language

    assert normalise_language(given) == expected


@pytest.mark.parametrize("supported", [0, 3, 6, 13])
def test_supported_lookaheads_pass_through(supported):
    from nixorb.asr.nemotron_asr import resolve_lookahead

    assert resolve_lookahead(supported) == supported


@pytest.mark.parametrize(
    "given,expected",
    [(1, 0), (5, 6), (99, 13), ("garbage", 3), (None, 3)],
)
def test_unsupported_lookahead_is_clamped(given, expected):
    """The processor rejects anything outside the trained set."""
    from nixorb.asr.nemotron_asr import resolve_lookahead

    assert resolve_lookahead(given) == expected


def test_every_lookahead_has_a_documented_latency():
    from nixorb.asr.nemotron_asr import LOOKAHEAD_LATENCY_MS

    assert LOOKAHEAD_LATENCY_MS == {0: 80, 3: 320, 6: 560, 13: 1120}


@pytest.mark.parametrize(
    "text,expected_text,expected_lang",
    [
        ("Hello there. <en-US>", "Hello there.", "en-US"),
        ("Bonjour. <fr>", "Bonjour.", "fr"),
        ("No tag at all.", "No tag at all.", None),
        ("Tag <en-US> mid sentence stays.", "Tag <en-US> mid sentence stays.", None),
    ],
)
def test_split_language_tag(text, expected_text, expected_lang):
    """In auto mode the detected locale rides after the terminal punctuation."""
    from nixorb.asr.nemotron_asr import split_language_tag

    assert split_language_tag(text) == (expected_text, expected_lang)


def test_chunk_window_back_shifts_by_half_the_window():
    """Without the shift, each chunk restarts the STFT and the seams stutter."""
    from nixorb.asr.nemotron_asr import chunk_window

    start, end = chunk_window(mel_frame_idx=10, hop_length=160, window=400,
                              samples_per_chunk=1280)
    assert start == 10 * 160 - 200
    assert end == start + 1280


def test_feature_window_prefers_win_length_over_n_fft():
    """The processor derives its chunk sizes from win_length, so we must too."""
    from nixorb.asr.nemotron_asr import feature_window

    class Both:
        hop_length, win_length, n_fft = 160, 400, 512

    class OnlyFft:
        hop_length, n_fft = 160, 512

    assert feature_window(Both()) == 400
    assert feature_window(OnlyFft()) == 512


def test_chunk_windows_advance_without_overlap():
    """Cache-aware streaming is strictly non-overlapping; that is the point."""
    from nixorb.asr.nemotron_asr import chunk_window

    hop, window, per_chunk = 160, 400, 1280
    mel_per_chunk = per_chunk // hop

    windows = []
    idx = 8
    for _ in range(4):
        windows.append(chunk_window(idx, hop, window, per_chunk))
        idx += mel_per_chunk

    for (_, prev_end), (next_start, _) in zip(windows, windows[1:], strict=False):
        assert next_start == prev_end


def test_nemotron_default_model_is_the_streaming_checkpoint():
    from nixorb.asr.nemotron_asr import DEFAULT_MODEL

    assert DEFAULT_MODEL == "nvidia/nemotron-3.5-asr-streaming-0.6b"


def test_nemotron_reports_configured_latency():
    from nixorb.asr.nemotron_asr import NemotronASREngine

    engine = NemotronASREngine(
        Settings(asr_backend="nemotron", asr_nemotron_lookahead=13)
    )
    assert engine.latency_ms == 1120


# ── Live audio buffer ────────────────────────────────────────────── #

def test_live_audio_slices_across_appends():
    import numpy as np

    from nixorb.asr.nemotron_asr import _LiveAudio

    live = _LiveAudio()
    live.append(np.arange(5, dtype=np.float32))
    live.append(np.arange(5, 10, dtype=np.float32))

    chunk = live.take(2, 8, timeout=0.5)
    assert list(chunk) == [2, 3, 4, 5, 6, 7]


def test_live_audio_returns_none_once_closed_and_short():
    import numpy as np

    from nixorb.asr.nemotron_asr import _LiveAudio

    live = _LiveAudio()
    live.append(np.zeros(4, dtype=np.float32))
    live.close()

    assert live.take(0, 100, timeout=0.5) is None
    assert live.closed is True


def test_live_audio_waits_for_a_late_append():
    import threading
    import time

    import numpy as np

    from nixorb.asr.nemotron_asr import _LiveAudio

    live = _LiveAudio()

    def _feed():
        time.sleep(0.05)
        live.append(np.ones(10, dtype=np.float32))

    threading.Thread(target=_feed, daemon=True).start()
    chunk = live.take(0, 10, timeout=2.0)
    assert chunk is not None and len(chunk) == 10


# ── HF helpers ───────────────────────────────────────────────────── #

def test_missing_dependency_names_the_install_command():
    from nixorb import hf

    with pytest.raises(hf.MissingDependency, match=r"nixorb\[hf\]"):
        hf.require("a_module_that_does_not_exist")


def test_resolve_device_honours_cpu():
    from nixorb import hf

    assert hf.resolve_device("cpu") == "cpu"


def test_resolve_device_rejects_nonsense(caplog):
    from nixorb import hf

    assert hf.resolve_device("quantum") in ("cpu", "cuda")
    assert "quantum" in caplog.text


def test_token_prefers_settings_then_env(monkeypatch):
    from nixorb import hf

    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert hf.token(Settings(hf_token="from-config")) == "from-config"
    assert hf.token(Settings()) == "from-env"

    monkeypatch.delenv("HF_TOKEN")
    assert hf.token(Settings()) is None


def test_load_kwargs_omits_trust_remote_code_by_default():
    """It executes code from the model repo, so it must be opt-in."""
    from nixorb import hf

    assert "trust_remote_code" not in hf.load_kwargs(Settings())
    assert hf.load_kwargs(Settings(hf_trust_remote_code=True))["trust_remote_code"]


def test_load_kwargs_passes_cache_dir(tmp_path):
    from nixorb import hf

    kwargs = hf.load_kwargs(Settings(hf_cache_dir=str(tmp_path)))
    assert kwargs["cache_dir"] == str(tmp_path)


# ── HF ASR result shapes ─────────────────────────────────────────── #

@pytest.mark.parametrize(
    "result,expected",
    [
        ("plain text", "plain text"),
        ({"text": " hello "}, "hello"),
        ([{"text": "one"}, {"text": "two"}], "one two"),
        ({"text": ""}, ""),
    ],
)
def test_hf_asr_extracts_text_from_pipeline_output(result, expected):
    from nixorb.asr.hf_asr import _extract_text

    assert _extract_text(result) == expected


def test_hf_asr_only_sends_language_to_multilingual_models():
    """Passing language= to a CTC model is an error, not a no-op."""
    from nixorb.asr.hf_asr import HFASREngine

    whisper = HFASREngine(Settings(asr_model="openai/whisper-small",
                                   asr_language="fr"))
    wav2vec = HFASREngine(Settings(asr_model="facebook/wav2vec2-base-960h",
                                   asr_language="fr"))
    blank = HFASREngine(Settings(asr_model="openai/whisper-small",
                                 asr_language=""))

    assert whisper._wants_language() is True
    assert wav2vec._wants_language() is False
    assert blank._wants_language() is False


# ── HF TTS output shapes ─────────────────────────────────────────── #

@pytest.mark.parametrize(
    "result",
    [
        {"audio": [0.1, 0.2], "sampling_rate": 22050},
        [{"audio": [0.1, 0.2], "sampling_rate": 22050}],
        {"waveform": [0.1, 0.2], "rate": 22050},
    ],
)
def test_hf_tts_normalises_pipeline_output(result):
    from nixorb.tts.hf_tts import _waveform_from

    audio, rate = _waveform_from(result)
    assert rate == 22050
    assert len(audio) == 2


def test_hf_tts_rejects_unrecognised_output():
    from nixorb.tts.hf_tts import _waveform_from

    with pytest.raises(RuntimeError, match="Unrecognised"):
        _waveform_from(object())


async def test_hf_tts_failure_is_reported_on_the_bus(started_bus):
    """A model that cannot load must say so, not leave the orb silently mute."""
    from nixorb.core.event_bus import Event
    from nixorb.tts.hf_tts import HuggingFaceTTS

    engine = HuggingFaceTTS(Settings(tts_backend="huggingface"))
    engine._load = MagicMock(side_effect=RuntimeError("no such model"))

    seen = []

    async def _record(payload):
        seen.append((payload.event, payload.data))

    started_bus.subscribe(Event.TTS_ERROR, _record)
    started_bus.subscribe(Event.LOG, _record)

    await engine.speak("hello")
    await started_bus._queue.join()

    assert Event.TTS_ERROR in [e for e, _ in seen]
    assert any("no such model" in str(d) for _, d in seen)


# ── LLM tool calls ───────────────────────────────────────────────── #

def test_hf_backend_parses_in_band_tool_calls():
    """Qwen-style models emit calls as JSON in tags, not a structured field."""
    from nixorb.llm.hf_backend import _parse_tool_calls

    text = (
        'Sure.\n<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "Perth"}}\n</tool_call>'
    )
    assert _parse_tool_calls(text) == [
        {"name": "get_weather", "arguments": {"city": "Perth"}}
    ]


def test_hf_backend_ignores_malformed_tool_calls():
    from nixorb.llm.hf_backend import _parse_tool_calls

    assert _parse_tool_calls("<tool_call>not json</tool_call>") == []
    assert _parse_tool_calls("no tags here") == []


# ── Nemotron against the real transformers classes ───────────────── #
# These skip unless transformers ships Nemotron3_5Asr (>= 5.13). They check
# the API this backend is written against actually looks the way it does in
# the model card, so a transformers upgrade cannot break streaming silently.

def _has_nemotron() -> bool:
    try:
        import transformers
    except ImportError:
        return False
    return hasattr(transformers, "AutoModelForRNNT")


nemotron_only = pytest.mark.skipif(
    not _has_nemotron(), reason="needs transformers >= 5.13"
)


@nemotron_only
def test_automodel_for_rnnt_resolves_nemotron():
    from transformers.models.auto.modeling_auto import MODEL_FOR_RNNT_MAPPING_NAMES

    assert MODEL_FOR_RNNT_MAPPING_NAMES["nemotron3_5_asr"] == "Nemotron3_5AsrForRNNT"


@nemotron_only
def test_processor_exposes_every_streaming_attribute_we_use():
    """The streaming loop is written against these six names."""
    from transformers.models.nemotron3_5_asr.processing_nemotron3_5_asr import (
        Nemotron3_5AsrProcessor,
    )

    for attr in (
        "num_samples_first_audio_chunk",
        "num_mel_frames_first_audio_chunk",
        "num_samples_per_audio_chunk",
        "num_mel_frames_per_audio_chunk",
        "set_num_lookahead_tokens",
        "streaming_latency_ms",
    ):
        assert hasattr(Nemotron3_5AsrProcessor, attr), attr


@nemotron_only
def test_processor_carries_the_supported_lookahead_list():
    """Read off the instance, not the class — _apply_lookahead relies on it."""
    import inspect

    from transformers.models.nemotron3_5_asr.processing_nemotron3_5_asr import (
        Nemotron3_5AsrProcessor,
    )

    params = inspect.signature(Nemotron3_5AsrProcessor.__init__).parameters
    assert "supported_num_lookahead_tokens" in params
    assert "default_num_lookahead_tokens" in params


@nemotron_only
def test_processor_accepts_the_call_arguments_we_pass():
    import inspect

    from transformers.models.nemotron3_5_asr.processing_nemotron3_5_asr import (
        Nemotron3_5AsrProcessor,
    )

    params = inspect.signature(Nemotron3_5AsrProcessor.__call__).parameters
    for name in ("audio", "sampling_rate", "is_streaming",
                 "is_first_audio_chunk", "language"):
        assert name in params, name


@nemotron_only
def test_processor_injects_lookahead_into_generate_inputs():
    """We rely on this: generate() must see the same right-context as the
    chunk sizes, and the processor is what puts it there."""
    import inspect

    from transformers.models.nemotron3_5_asr import processing_nemotron3_5_asr as mod

    source = inspect.getsource(mod.Nemotron3_5AsrProcessor.__call__)
    assert 'inputs["num_lookahead_tokens"]' in source


def test_apply_lookahead_respects_what_the_checkpoint_supports(caplog):
    """A fine-tune may support fewer right-contexts; set_num_lookahead_tokens
    raises on anything outside its own list."""
    from nixorb.asr.nemotron_asr import NemotronASREngine

    engine = NemotronASREngine(
        Settings(asr_backend="nemotron", asr_nemotron_lookahead=13)
    )
    processor = MagicMock()
    processor.supported_num_lookahead_tokens = [0, 3]
    processor.streaming_latency_ms = 320

    engine._apply_lookahead(processor)

    processor.set_num_lookahead_tokens.assert_called_once_with(3)
    assert engine._lookahead == 3
    assert "not supported" in caplog.text


def test_apply_lookahead_survives_a_rejecting_processor():
    from nixorb.asr.nemotron_asr import NemotronASREngine

    engine = NemotronASREngine(Settings(asr_backend="nemotron"))
    processor = MagicMock()
    processor.supported_num_lookahead_tokens = []
    processor.set_num_lookahead_tokens.side_effect = ValueError("nope")
    processor.default_num_lookahead_tokens = 6

    engine._apply_lookahead(processor)
    assert engine._lookahead == 6


def test_latency_never_raises_for_an_unexpected_lookahead():
    from nixorb.asr.nemotron_asr import NemotronASREngine

    engine = NemotronASREngine(Settings(asr_backend="nemotron"))
    engine._lookahead = 7  # chosen by the checkpoint, not by us
    assert engine.latency_ms > 0
