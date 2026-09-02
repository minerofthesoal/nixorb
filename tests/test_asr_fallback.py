"""What happens when a torch-backed ASR engine cannot load at all.

Seen in the wild: transformers>=5 imports torchaudio at module scope from
`audio_utils`, so a torch install whose pieces come from different builds
makes `AutoProcessor.from_pretrained` raise

    ImportError: libcudart.so.12: cannot open shared object file

before a single sample is read. That escaped `preload()`, killed the turn,
and did so identically on every trigger afterwards — the assistant was
deaf for the rest of the session.
"""
from __future__ import annotations

import pytest

from nixorb.asr.base import (
    FALLBACK_WHISPER_MODEL,
    ASREngine,
    build_whisper_fallback,
    whisper_fallback_settings,
)
from nixorb.hf import describe_native_import_error
from nixorb.settings import Settings

LIBCUDART = "libcudart.so.12: cannot open shared object file: No such file or directory"


class TestDescribeNativeImportError:
    def test_recognises_a_missing_cuda_runtime(self):
        advice = describe_native_import_error(ImportError(LIBCUDART))
        assert advice is not None
        assert "libcudart.so.12" in advice
        assert "torch" in advice and "torchaudio" in advice
        assert "download.pytorch.org" in advice

    def test_recognises_an_abi_mismatch(self):
        advice = describe_native_import_error(
            ImportError("/lib/libtorch_cpu.so: undefined symbol: _ZN3c105ErrorC1E")
        )
        assert advice is not None
        assert "undefined symbol" not in advice  # it explains, not echoes

    @pytest.mark.parametrize(
        "exc",
        [
            ImportError("No module named 'transformers'"),
            ValueError("unrecognized model type"),
            OSError("model.safetensors not found"),
        ],
    )
    def test_stays_quiet_about_unrelated_failures(self, exc):
        # Guessing "reinstall torch" at an unrelated error sends people the
        # wrong way, so it must return None rather than a plausible story.
        assert describe_native_import_error(exc) is None


class TestWhisperFallbackSettings:
    def test_replaces_a_model_whisper_cannot_load(self):
        original = Settings(asr_model="nvidia/nemotron-3.5-asr-streaming-0.6b")
        swapped = whisper_fallback_settings(original)
        assert swapped.asr_model == FALLBACK_WHISPER_MODEL
        # The caller's settings are shared; mutating them would silently
        # reconfigure the real engine too.
        assert original.asr_model == "nvidia/nemotron-3.5-asr-streaming-0.6b"

    def test_keeps_a_model_that_is_already_a_whisper_size(self):
        original = Settings(asr_model="large-v3")
        assert whisper_fallback_settings(original).asr_model == "large-v3"

    def test_carries_the_rest_of_the_configuration_over(self):
        original = Settings(asr_language="de", mic_sensitivity=0.8)
        swapped = whisper_fallback_settings(original)
        assert swapped.asr_language == "de"
        assert swapped.mic_sensitivity == 0.8


class _Broken(ASREngine):
    """An engine whose model cannot be loaded."""

    name = "broken"
    whisper_fallback = True

    def __init__(self, settings, exc):
        super().__init__(settings)
        self._exc = exc
        self.load_attempts = 0

    def _load(self):
        self.load_attempts += 1
        raise self._exc

    def _transcribe(self, audio):  # pragma: no cover - never reached
        raise AssertionError("the broken engine should never transcribe")


class _NoFallback(_Broken):
    name = "no-fallback"
    whisper_fallback = False


class _Stub:
    """A stand-in for faster-whisper that does load."""

    name = "faster-whisper"
    supports_streaming = False

    def __init__(self):
        self.preloaded = False
        self.unloaded = False
        self.stopped = False
        self.is_loaded = False

    @property
    def active_name(self):
        return self.name

    async def preload(self):
        self.preloaded = True
        self.is_loaded = True

    async def unload(self):
        self.unloaded = True
        self.is_loaded = False

    def stop_recording(self):
        self.stopped = True

    async def record_and_transcribe(self):
        return "heard by the stand-in"

    async def stream_transcribe(self):
        yield "partial"
        yield "heard by the stand-in"


@pytest.fixture
def stub(monkeypatch):
    made = _Stub()
    monkeypatch.setattr(
        "nixorb.asr.base.build_whisper_fallback",
        lambda settings, name, exc: made,
    )
    return made


class TestPreloadFallback:
    async def test_a_broken_engine_hands_over_instead_of_raising(self, stub):
        engine = _Broken(Settings(), ImportError(LIBCUDART))
        await engine.preload()
        assert stub.preloaded
        assert engine.is_loaded
        assert engine.active_name == "faster-whisper"

    async def test_the_turn_still_produces_a_transcript(self, stub):
        engine = _Broken(Settings(), ImportError(LIBCUDART))
        assert await engine.record_and_transcribe() == "heard by the stand-in"

    async def test_streaming_goes_through_the_stand_in(self, stub):
        engine = _Broken(Settings(), ImportError(LIBCUDART))
        assert [p async for p in engine.stream_transcribe()] == [
            "partial", "heard by the stand-in",
        ]

    async def test_it_does_not_retry_the_broken_load_every_turn(self, stub):
        engine = _Broken(Settings(), ImportError(LIBCUDART))
        await engine.record_and_transcribe()
        await engine.record_and_transcribe()
        assert engine.load_attempts == 1

    async def test_stop_and_unload_reach_the_stand_in(self, stub):
        engine = _Broken(Settings(), ImportError(LIBCUDART))
        await engine.preload()
        engine.stop_recording()
        assert stub.stopped
        await engine.unload()
        assert stub.unloaded
        assert not engine.is_loaded

    async def test_engines_that_opt_out_still_raise(self, stub):
        # faster-whisper itself must not recurse into a fallback, and a
        # genuinely misconfigured engine should be loud.
        engine = _NoFallback(Settings(), ImportError(LIBCUDART))
        with pytest.raises(ImportError):
            await engine.preload()

    async def test_the_original_error_wins_when_whisper_fails_too(self, monkeypatch):
        class _AlsoBroken(_Stub):
            async def preload(self):
                raise RuntimeError("faster-whisper is not installed")

        monkeypatch.setattr(
            "nixorb.asr.base.build_whisper_fallback",
            lambda settings, name, exc: _AlsoBroken(),
        )
        engine = _Broken(Settings(), ImportError(LIBCUDART))
        with pytest.raises(ImportError, match="libcudart"):
            await engine.preload()
        assert engine._delegate is None


class TestBuildWhisperFallback:
    def test_returns_none_when_faster_whisper_cannot_be_imported(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "nixorb.asr.whisper_engine":
                raise ImportError("no faster_whisper here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        assert build_whisper_fallback(
            Settings(), "nemotron", ImportError(LIBCUDART)
        ) is None

    def test_builds_a_whisper_engine_with_a_usable_model(self):
        engine = build_whisper_fallback(
            Settings(asr_model="nvidia/nemotron-3.5-asr-streaming-0.6b"),
            "nemotron",
            ImportError(LIBCUDART),
        )
        assert engine is not None
        assert engine.name == "faster-whisper"
        assert engine._model_name == FALLBACK_WHISPER_MODEL


class TestHFEngineStreaming:
    def test_it_has_the_method_it_advertises(self):
        # It sets supports_streaming = True, and main calls
        # stream_transcribe() on that basis — the method has to exist.
        from nixorb.asr.hf_asr_engine import HFASREngine

        assert HFASREngine.supports_streaming
        assert hasattr(HFASREngine, "stream_transcribe")


class TestTheRealNemotronEngine:
    """The engine that actually failed, wired the way the orb wires it."""

    def _engine(self, monkeypatch):
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "AutoModelForRNNT"):
            pytest.skip("transformers is too old for the native backend")

        from nixorb.asr.factory import build_asr

        # The failure the user hit: transformers>=5 imports torchaudio from
        # audio_utils at module scope, so AutoProcessor raises the linker
        # error before a single sample is read.
        def explode(*args, **kwargs):
            raise ImportError(LIBCUDART)

        monkeypatch.setattr(
            transformers.AutoProcessor, "from_pretrained", explode
        )
        engine = build_asr(Settings())
        assert engine.name == "nemotron", "the default config must route here"
        return engine

    async def test_the_default_config_survives_a_broken_torchaudio(
        self, monkeypatch, stub
    ):
        engine = self._engine(monkeypatch)
        # Before: this raised out of preload() and main logged "Turn failed".
        assert await engine.record_and_transcribe() == "heard by the stand-in"
        assert engine.active_name == "faster-whisper"

    async def test_streaming_survives_it_too(self, monkeypatch, stub):
        engine = self._engine(monkeypatch)
        assert engine.supports_streaming
        assert [p async for p in engine.stream_transcribe()] == [
            "partial", "heard by the stand-in",
        ]
