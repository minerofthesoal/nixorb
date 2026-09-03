"""Decoding one clip's tokens, and reporting a failure as a failure.

From the field, on a machine where Nemotron finally loaded:

    ASR: nemotron failed: expected string or bytes-like object, got 'list'
    No speech detected

Two bugs in one line. `generate(...).sequences` is batched — shape
(batch, tokens) — and `PreTrainedTokenizer.decode` routes a 2-D input to
its batched branch, returning a *list* of strings; the regex that strips
special tokens got the list. Then the crash was reported to the user as
silence, which is the one thing it was not.
"""
from __future__ import annotations

import numpy as np

from nixorb.asr.base import ASREngine
from nixorb.asr.nemotron_asr import NemotronASREngine, token_budget
from nixorb.settings import Settings


class _Tokeniser:
    """Decodes the way transformers really does, batched branch included."""

    def decode(self, token_ids, **kwargs):
        ids = np.asarray(token_ids).tolist()
        if ids and isinstance(ids[0], list):
            # This is the branch that produced a list where the caller
            # expected a string.
            return [self._one(row) for row in ids]
        return self._one(ids)

    def batch_decode(self, token_ids, **kwargs):
        ids = np.asarray(token_ids).tolist()
        if not (ids and isinstance(ids[0], list)):
            ids = [ids]
        return [self._one(row) for row in ids]

    @staticmethod
    def _one(ids):
        # What the model really emits in auto mode: a leftover special
        # token, the transcript, then the detected locale appended after
        # the terminal punctuation.
        return "<|startoftranscript|>hello there <en-US>" if ids else ""


class _DecodeOnly(_Tokeniser):
    """An older processor with no batch_decode at all."""

    batch_decode = None


class TestDecodeSequences:
    def test_a_batched_tensor_yields_a_string_not_a_list(self):
        sequences = np.array([[5, 6, 7]])
        out = NemotronASREngine._decode_sequences(_Tokeniser(), sequences)
        assert isinstance(out, str)
        assert out == "<|startoftranscript|>hello there <en-US>"

    def test_it_still_works_without_batch_decode(self):
        sequences = np.array([[5, 6, 7]])
        out = NemotronASREngine._decode_sequences(_DecodeOnly(), sequences)
        assert isinstance(out, str)
        assert "hello there" in out

    def test_an_unbatched_sequence_is_fine_too(self):
        out = NemotronASREngine._decode_sequences(_Tokeniser(), np.array([5, 6, 7]))
        assert out == "<|startoftranscript|>hello there <en-US>"

    def test_an_empty_batch_is_empty_text_not_a_crash(self):
        class _Empty:
            def batch_decode(self, token_ids, **kwargs):
                return []

        assert NemotronASREngine._decode_sequences(_Empty(), np.array([[]])) == ""

    def test_nested_lists_are_unwrapped(self):
        class _Nested:
            def batch_decode(self, token_ids, **kwargs):
                return [["deeply nested"]]

        assert NemotronASREngine._decode_sequences(
            _Nested(), np.array([[1]])
        ) == "deeply nested"


class TestTokenBudget:
    def test_it_scales_with_the_audio(self):
        assert token_budget(1.0) < token_budget(10.0) < token_budget(30.0)

    def test_short_clips_still_get_room(self):
        # A budget of 3 tokens for a 0.1s clip would truncate "yes".
        assert token_budget(0.1) >= 64

    def test_a_long_clip_stays_bounded(self):
        # Generous, but not a runaway decode.
        assert token_budget(30.0) < 2000


class _Processor(_Tokeniser):
    """Just enough processor for _transcribe to run end to end."""

    class feature_extractor:
        sampling_rate = 16000

    def __init__(self):
        self.tokenizer = self
        self.calls = []

    def __call__(self, audio, **kwargs):
        class _Inputs(dict):
            def to(self, *args, **kwargs):
                return self

        return _Inputs(input_features=np.zeros((1, 4, 80)))


class _Model:
    device = "cpu"
    dtype = "float32"

    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs

        class _Out:
            sequences = np.array([[1, 2, 3]])

        return _Out()


class TestTranscribeEndToEnd:
    def test_a_real_transcribe_returns_text_not_a_crash(self):
        engine = NemotronASREngine(Settings())
        model = _Model()
        engine._model = {"model": model, "processor": _Processor()}

        text = engine._transcribe(np.zeros(16000, dtype=np.float32))

        assert text == "hello there"
        assert engine.detected_language == "en-US"

    def test_it_bounds_the_generation_instead_of_warning(self):
        # transformers warns on every turn when generate() falls back to a
        # model-agnostic max_length of 640.
        engine = NemotronASREngine(Settings())
        model = _Model()
        engine._model = {"model": model, "processor": _Processor()}

        engine._transcribe(np.zeros(16000 * 5, dtype=np.float32))

        assert "max_new_tokens" in model.generate_kwargs
        assert model.generate_kwargs["max_new_tokens"] == token_budget(5.0)


class _Failing(ASREngine):
    name = "failing"

    def _load(self):
        return object()

    def _transcribe(self, audio):
        raise RuntimeError("expected string or bytes-like object, got 'list'")


class _Silent(ASREngine):
    name = "silent"

    def _load(self):
        return object()

    def _transcribe(self, audio):
        return ""


class TestLastError:
    """A crash and a silent room both return None. They are not the same."""

    async def test_a_crash_is_recorded(self, started_bus, monkeypatch):
        engine = _Failing(Settings())
        engine._model = object()
        monkeypatch.setattr(
            "nixorb.asr.base.record_with_vad",
            lambda *a, **k: np.zeros(16000, dtype=np.float32),
        )
        assert await engine.record_and_transcribe() is None
        assert "got 'list'" in engine.last_error

    async def test_silence_leaves_it_empty(self, started_bus, monkeypatch):
        engine = _Silent(Settings())
        engine._model = object()
        monkeypatch.setattr(
            "nixorb.asr.base.record_with_vad",
            lambda *a, **k: np.zeros(16000, dtype=np.float32),
        )
        assert await engine.record_and_transcribe() is None
        assert engine.last_error == ""

    async def test_it_is_cleared_between_turns(self, started_bus, monkeypatch):
        engine = _Failing(Settings())
        engine._model = object()
        monkeypatch.setattr(
            "nixorb.asr.base.record_with_vad",
            lambda *a, **k: np.zeros(16000, dtype=np.float32),
        )
        await engine.record_and_transcribe()
        assert engine.last_error

        # A later good turn must not still look like a failure.
        engine._transcribe = lambda audio: "fine now"  # type: ignore[method-assign]
        assert await engine.record_and_transcribe() == "fine now"
        assert engine.last_error == ""
