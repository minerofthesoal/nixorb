"""tests/test_speaker.py — streaming speech, barge-in, action suppression.

What makes a voice assistant feel responsive is not raw model speed, it is
that it starts talking before it has finished thinking. These cover that,
plus being able to talk over it.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from nixorb.tts.speaker import (
    MAX_BUFFER_CHARS,
    MIN_SENTENCE_CHARS,
    Speaker,
    split_sentences,
)


class FakeTTS:
    """Records what was spoken, with a realistic playback delay."""

    name = "fake"

    def __init__(self, delay: float = 0.02) -> None:
        self.said: list[str] = []
        self.times: list[float] = []
        self.stopped = False
        self._delay = delay
        self._t0 = time.monotonic()

    async def speak(self, text: str) -> None:
        if self.stopped:
            return
        self.said.append(text)
        self.times.append(time.monotonic() - self._t0)
        await asyncio.sleep(self._delay)

    def stop(self) -> None:
        self.stopped = True


# ── Sentence splitting ───────────────────────────────────────────── #

def test_splits_on_terminal_punctuation():
    sentences, rest = split_sentences("Hello there. How are you? Still going")
    assert sentences == ["Hello there.", "How are you?"]
    assert rest.strip() == "Still going"


def test_holds_back_a_fragment_too_short_to_speak():
    """Synthesising "Ok." on its own sounds worse than waiting one token."""
    sentences, rest = split_sentences("Ok. ")
    assert sentences == []
    assert rest == "Ok. "


def test_flushes_a_model_that_never_punctuates():
    """A run-on answer must still make noise rather than buffering forever."""
    text = "word " * 120
    sentences, rest = split_sentences(text)
    assert sentences, "nothing was flushed"
    assert len(sentences[0]) <= MAX_BUFFER_CHARS
    assert rest


def test_does_not_split_mid_word():
    text = "supercalifragilistic " * 40
    sentences, _ = split_sentences(text)
    assert not sentences[0].endswith("supercalifragilis")


def test_ellipsis_and_paragraph_breaks_end_a_sentence():
    sentences, _ = split_sentences("Let me think about that…  next")
    assert sentences == ["Let me think about that…"]

    sentences, _ = split_sentences("First paragraph here\n\nsecond")
    assert sentences and "First paragraph" in sentences[0]


def test_min_sentence_constant_is_sane():
    assert 0 < MIN_SENTENCE_CHARS < MAX_BUFFER_CHARS


# ── Streaming ────────────────────────────────────────────────────── #

async def test_speaks_before_generation_finishes():
    """The whole point: the reply starts while the model is still writing."""
    tts = FakeTTS(delay=0.03)
    speaker = Speaker(tts, streaming=True)
    await speaker.start()

    reply = ("The capital of France is Paris. It has about two million "
             "people. Would you like the weather there?")
    for word in reply.split(" "):
        await speaker.feed(word + " ")
        await asyncio.sleep(0.005)

    assert tts.said, "nothing was spoken while the model was still generating"
    generation_end = time.monotonic() - tts._t0
    await speaker.finish()

    assert tts.times[0] < generation_end
    assert len(tts.said) == 3


async def test_finish_speaks_the_unterminated_tail():
    """A reply with no final full stop must still be spoken."""
    tts = FakeTTS()
    speaker = Speaker(tts, streaming=True)
    await speaker.start()
    await speaker.feed("no punctuation at the end of this one")
    await speaker.finish()

    assert tts.said == ["no punctuation at the end of this one"]


async def test_non_streaming_mode_uses_say():
    tts = FakeTTS()
    speaker = Speaker(tts, streaming=False)
    await speaker.start()
    await speaker.feed("this should not be spoken yet")
    await speaker.finish()
    assert tts.said == []

    await speaker.say("the whole answer at once")
    assert tts.said == ["the whole answer at once"]


# ── Barge-in ─────────────────────────────────────────────────────── #

async def test_barge_in_stops_playback_and_drops_the_queue():
    tts = FakeTTS(delay=0.05)
    speaker = Speaker(tts, streaming=True)
    await speaker.start()

    for n in ("One sentence here. ", "Two sentences here. ",
              "Three sentences here. ", "Four sentences here. "):
        await speaker.feed(n)

    await asyncio.sleep(0.06)
    await speaker.stop()
    spoken_at_stop = len(tts.said)

    await asyncio.sleep(0.2)
    assert tts.stopped is True
    assert spoken_at_stop < 4, "stop() did not interrupt anything"
    assert len(tts.said) == spoken_at_stop, "kept speaking after stop()"


async def test_feeding_after_stop_is_ignored():
    tts = FakeTTS()
    speaker = Speaker(tts, streaming=True)
    await speaker.start()
    await speaker.stop()
    await speaker.feed("this arrived too late. ")
    await asyncio.sleep(0.05)
    assert tts.said == []


async def test_start_does_not_stop_an_idle_engine():
    """start() clears the decks; an engine may treat stop() as terminal."""
    tts = FakeTTS()
    speaker = Speaker(tts, streaming=True)
    await speaker.start()
    assert tts.stopped is False

    await speaker.feed("a complete sentence to speak. ")
    await speaker.finish()
    assert tts.said == ["a complete sentence to speak."]


async def test_speaking_reports_state():
    tts = FakeTTS(delay=0.05)
    speaker = Speaker(tts, streaming=True)
    assert speaker.speaking is False

    await speaker.start()
    await speaker.feed("something long enough to say. ")
    assert speaker.speaking is True

    await speaker.finish()
    assert speaker.speaking is False


async def test_a_failing_engine_does_not_kill_the_stream():
    """One bad sentence must not silence the rest of the answer."""
    class Flaky(FakeTTS):
        async def speak(self, text):
            if "boom" in text:
                raise RuntimeError("synth failed")
            await super().speak(text)

    tts = Flaky()
    speaker = Speaker(tts, streaming=True)
    await speaker.start()
    await speaker.feed("first sentence here. ")
    await speaker.feed("this one goes boom now. ")
    await speaker.feed("third sentence here. ")
    await speaker.finish()

    assert "first sentence here." in tts.said
    assert "third sentence here." in tts.said


# ── ACTION suppression ───────────────────────────────────────────── #

async def test_action_blocks_are_never_read_aloud():
    tts = FakeTTS()
    speaker = Speaker(tts, streaming=True)
    await speaker.start()

    text = "Sure, checking now. <ACTION>df -h</ACTION> You have plenty of space."
    for ch in text:
        await speaker.feed(ch)
    await speaker.finish()

    spoken = " ".join(tts.said)
    assert "df -h" not in spoken
    assert "<ACTION>" not in spoken and "</ACTION>" not in spoken
    assert "checking now" in spoken
    assert "plenty of space" in spoken


async def test_action_split_across_chunk_boundaries():
    """The tag rarely arrives in one token."""
    tts = FakeTTS()
    speaker = Speaker(tts, streaming=True)
    await speaker.start()

    for chunk in ["Running it. ", "<ACT", "ION>rm ", "-rf /tmp/x</ACT",
                  "ION> All done now."]:
        await speaker.feed(chunk)
    await speaker.finish()

    spoken = " ".join(tts.said)
    assert "rm" not in spoken and "ACTION" not in spoken
    assert "Running it." in spoken
    assert "All done now." in spoken


@pytest.mark.parametrize("tag_text", ["<ACTION>ls</ACTION>", "<ACTION>\nls -la\n</ACTION>"])
async def test_a_reply_that_is_only_a_command_speaks_nothing(tag_text):
    tts = FakeTTS()
    speaker = Speaker(tts, streaming=True)
    await speaker.start()
    for ch in tag_text:
        await speaker.feed(ch)
    await speaker.finish()
    assert tts.said == []
