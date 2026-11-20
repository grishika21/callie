"""Text to speech. Default: Piper with the en_US-lessac-medium voice, on the CPU.

Measured in V0: ~0.09 s to synthesize a 3-second sentence (~39x real time), so a
two-sentence reply is ready in about 0.2 s. `speak` returns one clip; `stream` yields a
clip per sentence, so playback can start after the first one.

Fallbacks: "browser" leaves speaking to the browser's own speech synthesis (the UI
does it), "none" stays silent. Both return no audio.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from talktable.config import DATA_DIR

logger = logging.getLogger(__name__)

VOICES_DIR = DATA_DIR / "voices"
DEFAULT_VOICE = "en_US-lessac-medium"


@dataclass(frozen=True)
class Speech:
    sample_rate: int
    samples: np.ndarray  # int16 mono
    seconds: float  # time to synthesize

    @property
    def audio_seconds(self) -> float:
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0


class Speaker(Protocol):
    def speak(self, text: str) -> Speech | None: ...


@lru_cache(maxsize=2)
def _voice(path: str) -> Any:
    from piper import PiperVoice

    started = time.perf_counter()
    voice = PiperVoice.load(path)
    logger.info("piper voice %s loaded in %.1f s", Path(path).stem, time.perf_counter() - started)
    return voice


class PiperSpeaker:
    def __init__(self, voice: str = DEFAULT_VOICE, voices_dir: Path = VOICES_DIR) -> None:
        self.path = voices_dir / f"{voice}.onnx"
        if not self.path.exists():
            raise FileNotFoundError(
                f"Piper voice {self.path} is missing. Download it with:\n"
                f"  python -m piper.download_voices {voice} --download-dir {voices_dir}")

    def load(self) -> None:
        _voice(str(self.path))

    def stream(self, text: str) -> Iterator[Speech]:
        voice = _voice(str(self.path))
        for sentence in (s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s):
            started = time.perf_counter()
            chunks = list(voice.synthesize(sentence))
            if not chunks:
                continue
            samples = np.concatenate([c.audio_int16_array for c in chunks])
            yield Speech(chunks[0].sample_rate, samples, time.perf_counter() - started)

    def speak(self, text: str) -> Speech | None:
        parts = list(self.stream(text))
        if not parts:
            return None
        gap = np.zeros(int(parts[0].sample_rate * 0.25), dtype=np.int16)  # a short pause between sentences
        samples = np.concatenate([x for p in parts for x in (p.samples, gap)][:-1])
        return Speech(parts[0].sample_rate, samples, sum(p.seconds for p in parts))


class SilentSpeaker:
    """'browser' (the page speaks the text itself) or 'none'."""

    def speak(self, text: str) -> Speech | None:
        return None


def make_speaker(engine: str, voice: str = DEFAULT_VOICE) -> Speaker:
    if engine == "piper":
        return PiperSpeaker(voice)
    if engine in ("browser", "none"):
        return SilentSpeaker()
    raise ValueError(f"unknown text-to-speech engine {engine!r} (use: piper, browser, none)")
