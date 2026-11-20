"""Hands-free mode: find where an utterance ends in the microphone stream.

The browser sends a chunk every half second. Speech is louder than the background
(an energy threshold above a running noise floor); an utterance ends after
`end_silence` seconds of quiet. Very short sounds (a cough, a click) are ignored and
very long ones are cut, so one utterance is always a sensible thing to transcribe.
Whisper's own voice activity filter then trims what is left.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from callie.speech.stt import WHISPER_RATE, to_mono_16k


class Endpointer:
    # min_speech: a one-word answer ("Virk.", "Yes.") has only ~0.25 s of voiced audio; 0.15 s
    # keeps those while a click (~40 ms) is still ignored (measured in the phone work).
    def __init__(self, *, end_silence: float = 0.8, min_speech: float = 0.15, max_utterance: float = 20.0,
                 floor: float = 0.006, ratio: float = 3.0, pre_roll: float = 0.3) -> None:
        self.end_silence, self.min_speech, self.max_utterance = end_silence, min_speech, max_utterance
        self.floor, self.ratio = floor, ratio
        self._pre_roll = int(pre_roll * WHISPER_RATE)
        self._noise: deque[float] = deque(maxlen=40)  # recent quiet frames' loudness
        self.reset()

    def reset(self) -> None:
        self._speech: list[np.ndarray] = []
        self._recent = np.zeros(0, dtype=np.float32)  # audio just before speech starts
        self._voiced = 0.0
        self._quiet = 0.0

    @property
    def speaking_for(self) -> float:
        """Seconds of speech in the utterance being collected (0 when there is none)."""
        return self._voiced if self._speech else 0.0

    @property
    def threshold(self) -> float:
        noise = float(np.median(self._noise)) if self._noise else 0.0
        return max(self.floor, noise * self.ratio)

    def feed(self, chunk: tuple[int, np.ndarray]) -> np.ndarray | None:
        """Add a chunk; returns a finished utterance (16 kHz float32) or None."""
        audio = to_mono_16k(chunk)
        if not len(audio):
            return None
        frame = int(0.03 * WHISPER_RATE)  # 30 ms frames
        utterance = None
        for start in range(0, len(audio), frame):
            piece = audio[start:start + frame]
            loud = float(np.sqrt(np.mean(piece**2)))
            seconds = len(piece) / WHISPER_RATE
            if loud > self.threshold:
                if not self._speech:
                    self._speech.append(self._recent[-self._pre_roll:])
                self._speech.append(piece)
                self._voiced += seconds
                self._quiet = 0.0
            elif self._speech:
                self._speech.append(piece)
                self._quiet += seconds
            else:
                self._noise.append(loud)
                self._recent = np.concatenate([self._recent, piece])[-self._pre_roll:]
            heard = sum(len(p) for p in self._speech) / WHISPER_RATE
            if self._speech and (self._quiet >= self.end_silence or heard >= self.max_utterance):
                if self._voiced >= self.min_speech:
                    utterance = np.concatenate(self._speech)
                self.reset()
        return utterance
