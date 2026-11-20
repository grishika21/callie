"""Phone audio: Twilio Media Streams carry G.711 μ-law, 8 kHz, mono, in 20 ms frames.

The codec is NumPy (Python's `audioop` is gone in 3.13); tests check it bit for bit
against `audioop` while 3.11 still has it. Resampling uses SciPy's polyphase filter, so
Piper's 22.05 kHz voice is low-passed properly before it is squeezed into 8 kHz.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

PHONE_RATE = 8_000
FRAME_BYTES = 160  # 20 ms of μ-law at 8 kHz
SILENCE_BYTE = 0xFF  # μ-law for zero
_BIAS = 0x84


def ulaw_decode(data: bytes) -> np.ndarray:
    """μ-law bytes -> int16 samples."""
    u = ~np.frombuffer(data, dtype=np.uint8)
    sign, exponent, mantissa = u & 0x80, (u >> 4) & 0x07, u & 0x0F
    magnitude = ((mantissa.astype(np.int32) << 3) + _BIAS) << exponent.astype(np.int32)
    magnitude -= _BIAS
    return np.where(sign != 0, -magnitude, magnitude).astype(np.int16)


_SEG_END = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF])


def ulaw_encode(samples: np.ndarray) -> bytes:
    """int16 samples -> μ-law bytes (the classic G.711 coder, on 14-bit magnitudes)."""
    x = samples.astype(np.int32) >> 2  # 16 -> 14 bits (an arithmetic shift, like audioop)
    negative = x < 0
    x = np.minimum(np.where(negative, -x, x), 8159) + (_BIAS >> 2)
    seg = np.searchsorted(_SEG_END, x)  # the first segment whose end is >= x
    uval = np.where(seg >= 8, 0x7F, (np.minimum(seg, 7) << 4) | ((x >> (np.minimum(seg, 7) + 1)) & 0x0F))
    return (uval ^ np.where(negative, 0x7F, 0xFF)).astype(np.uint8).tobytes()


def resample(samples: np.ndarray, rate: int, target: int) -> np.ndarray:
    """Polyphase resampling (with its anti-aliasing filter); keeps the dtype family."""
    if rate == target or not len(samples):
        return samples
    ratio = Fraction(target, rate).limit_denominator(1000)
    out = resample_poly(samples.astype(np.float32), ratio.numerator, ratio.denominator)
    if np.issubdtype(samples.dtype, np.integer):
        return np.clip(np.round(out), -32768, 32767).astype(np.int16)
    return out.astype(np.float32)


def to_phone(samples: np.ndarray, rate: int) -> bytes:
    """Any int16 audio (e.g. Piper at 22.05 kHz) -> 8 kHz μ-law bytes."""
    return ulaw_encode(resample(samples.astype(np.int16), rate, PHONE_RATE))


def from_phone(data: bytes) -> tuple[int, np.ndarray]:
    """8 kHz μ-law bytes -> (8000, int16 samples), ready for the endpointer and whisper."""
    return PHONE_RATE, ulaw_decode(data)


def frames(data: bytes) -> Iterator[str]:
    """μ-law bytes as base64 20 ms frames, the way Twilio's `media` messages carry them."""
    for start in range(0, len(data), FRAME_BYTES):
        chunk = data[start:start + FRAME_BYTES]
        if len(chunk) < FRAME_BYTES:
            chunk += bytes([SILENCE_BYTE]) * (FRAME_BYTES - len(chunk))
        yield base64.b64encode(chunk).decode()


def phone_line(samples: np.ndarray, rate: int, *, noise_snr_db: float | None = None,
               rng: np.random.Generator | None = None) -> tuple[int, np.ndarray]:
    """What a caller's voice sounds like after the phone network: band-limited to 8 kHz,
    μ-law companded, optionally with line noise. Used by the phone benchmarks."""
    eight = resample(samples.astype(np.int16), rate, PHONE_RATE).astype(np.float32)
    if noise_snr_db is not None:
        rng = rng or np.random.default_rng(0)
        noise = rng.normal(0, 1, len(eight)).astype(np.float32)
        noise *= np.sqrt(np.mean(eight**2) / (10 ** (noise_snr_db / 10)) / max(np.mean(noise**2), 1e-9))
        eight = eight + noise
    return from_phone(ulaw_encode(np.clip(eight, -32768, 32767).astype(np.int16)))
