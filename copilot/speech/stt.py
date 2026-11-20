"""Speech to text. Default: faster-whisper `small.en` on the CPU (int8), with voice
activity detection and a prompt of the shop's own words (carriers, ticket categories).

Measured in V0 (17 support sentences spoken by Piper): small.en heard every order id and
keyword, at ~1.3 s per 3-second clip; base.en ~0.5 s but missed 1 id in 10. The domain
prompt roughly halves base.en's word errors.

Audio comes in as the browser sends it, (sample_rate, samples) in any rate, int or
float, mono or stereo. It is resampled to 16 kHz with NumPy, so no ffmpeg is needed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

WHISPER_RATE = 16_000
DOMAIN_PROMPT = ("Bookstore support call. Order, ticket and customer numbers. Carriers: SwiftShip, ParcelPro, "
                 "GlobalPost, NorthStar Freight, QuickDrop, OceanAir Logistics. Ticket categories: delivery, payment, "
                 "damaged item, refund, account. Priority: low, normal, high, urgent. TalkTable.")


@dataclass(frozen=True)
class Transcript:
    text: str
    seconds: float  # time to transcribe
    audio_seconds: float


class Transcriber(Protocol):
    # prompt: words to expect (whisper's initial prompt); None = the transcriber's default
    def transcribe(self, audio: tuple[int, np.ndarray], prompt: str | None = None) -> Transcript: ...


def to_mono_16k(audio: tuple[int, np.ndarray]) -> np.ndarray:
    """Any (rate, samples) -> float32 mono at 16 kHz in [-1, 1] (linear resampling)."""
    rate, samples = audio
    data = np.asarray(samples)
    if np.issubdtype(data.dtype, np.integer):  # scale before mixing channels, while the type is known
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1) if data.shape[1] <= 8 else data.mean(axis=0)
    if rate != WHISPER_RATE and len(data):
        n = max(1, round(len(data) * WHISPER_RATE / rate))
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)
    return np.clip(data, -1.0, 1.0)


def load_audio(path: str | Path) -> tuple[int, np.ndarray]:
    """Any audio file the browser uploads (WAV, WebM/Opus, Ogg, MP3...) as 16 kHz mono
    float32, decoded with PyAV (bundled with faster-whisper), so no ffmpeg is needed."""
    import av

    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError(f"{path}: no audio track")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=WHISPER_RATE)
        parts = [frame.to_ndarray().reshape(-1) for packet in container.demux(stream) for decoded in packet.decode()
                 for frame in resampler.resample(decoded)]
        parts += [frame.to_ndarray().reshape(-1) for frame in resampler.resample(None)]  # flush
    data = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)
    return WHISPER_RATE, data


@lru_cache(maxsize=2)
def _whisper(model: str, compute_type: str, threads: int) -> Any:
    from faster_whisper import WhisperModel

    started = time.perf_counter()
    loaded = WhisperModel(model, device="cpu", compute_type=compute_type, cpu_threads=threads)
    logger.info("whisper %s loaded in %.1f s", model, time.perf_counter() - started)
    return loaded


class WhisperTranscriber:
    def __init__(self, model: str = "small.en", *, beam_size: int = 5, compute_type: str = "int8", threads: int = 4,
                 prompt: str | None = DOMAIN_PROMPT) -> None:
        self.model_name, self.beam_size, self.prompt = model, beam_size, prompt
        self._args = (model, compute_type, threads)

    def load(self) -> None:
        _whisper(*self._args)

    def transcribe(self, audio: tuple[int, np.ndarray], prompt: str | None = None) -> Transcript:
        data = to_mono_16k(audio)
        started = time.perf_counter()
        segments, _info = _whisper(*self._args).transcribe(
            data, language="en", beam_size=self.beam_size, vad_filter=True, initial_prompt=prompt or self.prompt,
            condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segments).strip()
        return Transcript(text, time.perf_counter() - started, len(data) / WHISPER_RATE)


class GroqTranscriber:
    """Groq's hosted Whisper. Fast where the CPU is slow (the free Hugging Face Space);
    if Groq fails or is rate-limited, the local `fallback` transcribes instead."""

    URL = "https://api.groq.com/openai/v1/audio/transcriptions"

    def __init__(self, api_key: str, model: str = "whisper-large-v3-turbo", *, fallback: Transcriber | None = None,
                 prompt: str | None = DOMAIN_PROMPT, timeout: float = 15.0) -> None:
        self._key, self.model, self.fallback, self.prompt, self.timeout = api_key, model, fallback, prompt, timeout

    def load(self) -> None:
        if self.fallback is not None and hasattr(self.fallback, "load"):
            self.fallback.load()

    def transcribe(self, audio: tuple[int, np.ndarray], prompt: str | None = None) -> Transcript:
        import io
        import wave

        import httpx

        data = to_mono_16k(audio)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(WHISPER_RATE)
            handle.writeframes((data * 32767).astype(np.int16).tobytes())
        started = time.perf_counter()
        try:
            response = httpx.post(self.URL, headers={"Authorization": f"Bearer {self._key}"}, timeout=self.timeout,
                                  files={"file": ("speech.wav", buffer.getvalue(), "audio/wav")},
                                  data={"model": self.model, "language": "en", "temperature": "0",
                                        **({"prompt": prompt or self.prompt} if prompt or self.prompt else {})})
            response.raise_for_status()
            text = str(response.json().get("text", "")).strip()
            return Transcript(text, time.perf_counter() - started, len(data) / WHISPER_RATE)
        except Exception as exc:
            if self.fallback is None:
                raise
            logger.warning("Groq transcription failed (%s); using the local model", type(exc).__name__)
            return self.fallback.transcribe(audio, prompt)


def make_transcriber(engine: str, model: str, *, groq_key: str | None = None,
                     groq_model: str = "whisper-large-v3-turbo") -> Transcriber:
    if engine == "whisper":
        return WhisperTranscriber(model)
    if engine == "groq":
        if not groq_key:
            raise ValueError("Groq speech-to-text needs GROQ_API_KEY")
        return GroqTranscriber(groq_key, groq_model, fallback=WhisperTranscriber(model))
    raise ValueError(f"unknown speech-to-text engine {engine!r} (use: whisper, groq)")
