"""Speech helpers: text for speaking, transcripts back to ids, spoken confirmations,
audio resampling and Piper. No whisper here: the round trip is test_speech_roundtrip.py."""

from __future__ import annotations

import numpy as np
import pytest

from copilot.speech.confirm import parse_confirmation
from copilot.speech.stt import to_mono_16k
from copilot.speech.text import normalize_transcript, speakable
from copilot.speech.tts import VOICES_DIR, PiperSpeaker, SilentSpeaker, make_speaker


@pytest.mark.parametrize(("heard", "meant"), [
    ("Where is order five five one two?", "Where is order 5512?"),
    ("Where is order fifty five twelve?", "Where is order 5512?"),
    ("Order seventy seven eighty one.", "Order 7781."),
    ("Order double five one two.", "Order 5512."),
    ("Order five thousand five hundred twelve.", "Order 5512."),
    ("Customer one one three nine can't log in.", "Customer 1139 can't log in."),
    ("Ticket oh four two.", "Ticket 42."),
    ("The number is five five one two.", "The number is 5512."),
    # whisper's own spellings of one spoken id (seen in V0)
    ("Order 55 12 has not arrived.", "Order 5512 has not arrived."),
    ("Order 12.03 is late.", "Order 1203 is late."),
    ("Ticket number 1.999 please.", "Ticket number 1999 please."),
    ("Order 778-1.", "Order 7781."),
    ("Order 6-0-2-0 was damaged.", "Order 6020 was damaged."),
    ("Order #5512.", "Order 5512."),
    ("Open a ticket for order five five one two, the book arrived damaged.",
     "Open a ticket for order 5512, the book arrived damaged."),
    # left alone
    ("Create one for it.", "Create one for it."),
    ("I have 2 tickets and $12.03 left.", "I have 2 tickets and $12.03 left."),
    ("Make it high priority.", "Make it high priority."),
    ("Hi there!", "Hi there!"),
])
def test_normalize_transcript(heard: str, meant: str) -> None:
    assert normalize_transcript(heard) == meant


@pytest.mark.parametrize(("reply", "spoken"), [
    ("Order 5512 was delivered on 2026-01-24 by **SwiftShip**. The total was $48.47. There are no open tickets.",
     "Order 5 5 1 2 was delivered on January 24, 2026 by SwiftShip. The total was $48.47."),
    ("| a | b |\n|---|---|\n| 1 | 2 |\nTicket 1481 is open.", "Ticket 1 4 8 1 is open."),
    ("Her email is maria.gonzalez@example.com.", "Her email is an address at example.com."),
    ("That is m***@example.com, see https://x.y/z.", "That is an address at example.com, see a link on screen."),
    ("## Result\n- one\n- two", "Result. one. two."),
    ("First. Second. Third.", "First. Second."),  # two sentences at most
    ("Opened ticket 1481 for Prisha Virk.", "Opened ticket 1 4 8 1 for Prisha Virk."),
    ("Open a high-priority damaged item ticket for Prisha Virk (customer 1139) about order 5512: “Book”. Shall I open it?",
     "Open a high-priority damaged item ticket for Prisha Virk about order 5 5 1 2: “Book”. Shall I open it?"),
])
def test_speakable(reply: str, spoken: str) -> None:
    assert speakable(reply) == spoken


@pytest.mark.parametrize(("said", "kind", "changes"), [
    ("Yes.", "approve", {}), ("yeah go ahead", "approve", {}), ("Sure, do it.", "approve", {}),
    ("Sounds good", "approve", {}), ("Yes please.", "approve", {}), ("Okay.", "approve", {}),
    ("No.", "reject", {}), ("No thanks", "reject", {}), ("Cancel that.", "reject", {}), ("Never mind.", "reject", {}),
    ("Don't open it.", "reject", {}),
    ("Make it urgent.", "edit", {"priority": "urgent"}), ("Make it high priority", "edit", {"priority": "high"}),
    ("Yes, but make it urgent.", "edit", {"priority": "urgent"}), ("Set the priority to low.", "edit", {"priority": "low"}),
    ("Change it to a refund ticket.", "edit", {"category": "refund"}), ("It's a payment issue.", "edit", {"category": "payment"}),
    ("It's for order 7888.", "edit", {"order_id": 7888}),
    ("Change the subject to Book never arrived.", "edit", {"subject": "Book never arrived"}),
    ("Actually, where is order 7888?", "new_request", {}), ("No, where is order 7888 instead?", "new_request", {}),
    ("Find the customer Travis Casey", "new_request", {}),
    ("Hmm.", "unclear", {}), ("What?", "unclear", {}), ("", "unclear", {}), ("Wait.", "unclear", {}),
    ("Yes, and where is order 7888?", "unclear", {}),
])
def test_parse_confirmation(said: str, kind: str, changes: dict) -> None:
    answer = parse_confirmation(said)
    assert (answer.kind, answer.changes) == (kind, changes)


def test_to_mono_16k() -> None:
    t = np.linspace(0, 1, 48_000, endpoint=False)
    stereo = (np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1) * 32767 * 0.5).astype(np.int16)
    mono = to_mono_16k((48_000, stereo))
    assert mono.dtype == np.float32 and len(mono) == 16_000
    assert 0.45 < np.abs(mono).max() < 0.55  # same loudness, int16 scaled to [-1, 1]
    assert len(to_mono_16k((16_000, np.zeros(800, dtype=np.float32)))) == 800


@pytest.mark.skipif(not (VOICES_DIR / "en_US-lessac-medium.onnx").exists(), reason="Piper voice not downloaded")
def test_piper_speaks() -> None:
    speaker = PiperSpeaker()
    speech = speaker.speak("Opened ticket 1 4 8 1. Anything else?")
    assert speech is not None and speech.samples.dtype == np.int16 and speech.sample_rate == 22_050
    assert 2.0 < speech.audio_seconds < 6.0 and speech.seconds < 3.0
    assert len(list(speaker.stream("One. Two. Three."))) == 3  # a clip per sentence


def test_silent_speakers() -> None:
    assert isinstance(make_speaker("browser"), SilentSpeaker) and make_speaker("none").speak("hi") is None
    with pytest.raises(FileNotFoundError, match="download_voices"):
        PiperSpeaker("no-such-voice")


@pytest.mark.skipif(not (VOICES_DIR / "en_US-lessac-medium.onnx").exists(), reason="Piper voice not downloaded")
def test_endpointer_keeps_one_word_answers_and_ignores_clicks() -> None:
    """A bug found in the phone work: "Virk." (~0.25 s voiced) was dropped as too short."""
    from copilot.speech.vad import Endpointer

    def heard(samples: np.ndarray, rate: int) -> bool:
        ep, found = Endpointer(end_silence=0.7), False
        audio = np.concatenate([samples.astype(np.int16), np.zeros(int(1.2 * rate), dtype=np.int16)])
        for i in range(0, len(audio), rate // 50):
            found |= ep.feed((rate, audio[i:i + rate // 50])) is not None
        return found

    speaker = PiperSpeaker()
    for word in ("Virk.", "Yes.", "No."):
        speech = speaker.speak(word)
        assert speech is not None and heard(speech.samples, speech.sample_rate), word
    click = (np.random.default_rng(0).normal(0, 8000, 320)).astype(np.int16)  # 40 ms at 8 kHz
    assert not heard(click, 8000)
