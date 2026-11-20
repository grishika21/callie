"""Speech round trip (slow: loads whisper): Piper speaks the support sentences in
eval/speech_sentences.yaml, whisper small.en transcribes them, and after normalizing
every order/ticket/customer id and keyword must come back. Run with:
    TALKTABLE_LIVE_TESTS=1 .venv/bin/python -m pytest tests/test_speech_roundtrip.py
"""

from __future__ import annotations

import pytest

from copilot.speech.tts import VOICES_DIR
from eval.benchmark_speech import run

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(not (VOICES_DIR / "en_US-lessac-medium.onnx").exists(), reason="no Piper voice")]


def test_ids_and_keywords_survive_the_round_trip() -> None:
    [row] = run(["small.en"], noise=None)
    assert row["misses"] == []
    assert row["wer"] < 0.05 and row["median_s"] < 3.0
