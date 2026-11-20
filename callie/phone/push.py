"""Push-to-talk calls, for hosts that can't stream audio both ways (Streamlit Community Cloud).

The visitor records a whole answer and presses stop; the recording becomes one utterance for
the same CallSession a Twilio call drives (8 kHz μ-law, the same verification, agent and
guards). When the turn is done, everything the call said comes back as one clip to play.
Nothing listens while the page plays it, so a reply's mark fires as soon as it is queued.

No web framework in here: the Streamlit page (streamlit_demo/streamlit_app.py) is a thin
layer over `PushToTalkLine`, and the tests drive this module directly.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import secrets
import time
import wave
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from callie.phone.audio import PHONE_RATE, resample, ulaw_decode, ulaw_encode
from callie.phone.call import CallSession, PhoneDeps, Timing
from callie.speech.stt import to_mono_16k

NEVER = 10**9  # seconds: a timer that doesn't fire

ENDINGS = {
    "goodbye": "The call ended. Thanks for trying it!",
    "transferred": "Transferred to a person (in this demo, nobody is on the line).",
    "not verified": "The call ended: the details didn't match. Try order 5512, last name Virk.",
    "time limit": "The call reached its time limit.",
    "caller hung up": "You hung up.",
}

LISTENING = {"order": "Your order number, please", "name": "Your last name, please",
             "spell": "Spell your last name, please", "help": "Verified. How can Callie help?"}


def push_to_talk_timing(max_seconds: float) -> Timing:
    """No "One moment" (the page shows that it's thinking) and no "Are you still there?" (nobody
    would hear it until the next turn). An abandoned call ends at the time limit."""
    return Timing(silence_nudge=NEVER, silence_end=NEVER, max_seconds=max_seconds, filler_after=NEVER,
                  still_looking_every=NEVER)


class ClipIO:
    """CallIO that keeps the call's audio until the page asks for it."""

    def __init__(self) -> None:
        self.session: CallSession | None = None
        self.clips: list[np.ndarray] = []
        self.hung_up = self.transferred = False

    async def play(self, mulaw: bytes, mark: str) -> None:
        self.clips.append(ulaw_decode(mulaw))
        if self.session is not None:  # "played": the page plays it after the turn, and nothing listens meanwhile
            session = self.session
            asyncio.get_running_loop().call_soon(lambda: asyncio.ensure_future(session.on_mark(mark)))

    async def clear(self) -> None:
        """Barge-in can't happen here (nothing plays during a turn), so there's nothing to stop."""

    async def hang_up(self) -> None:
        self.hung_up = True

    async def transfer(self) -> None:
        self.transferred = True

    def take(self) -> tuple[int, np.ndarray] | None:
        """Everything said since the last take, as one 8 kHz clip with short gaps."""
        if not self.clips:
            return None
        gap = np.zeros(int(0.25 * PHONE_RATE), dtype=np.int16)
        audio = np.concatenate([part for clip in self.clips for part in (clip, gap)])
        self.clips = []
        return PHONE_RATE, audio


def decode_recording(data: bytes) -> tuple[int, np.ndarray]:
    """The browser's recording (WAV) as (rate, samples)."""
    with wave.open(io.BytesIO(data)) as handle:
        rate, width, channels = handle.getframerate(), handle.getsampwidth(), handle.getnchannels()
        raw = handle.readframes(handle.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")
    samples = np.frombuffer(raw, dtype=dtype)
    if width == 1:  # 8-bit WAV is unsigned
        samples = (samples.astype(np.int16) - 128) * 256
    return rate, samples.reshape(-1, channels) if channels > 1 else samples


def to_phone_line(audio: tuple[int, np.ndarray]) -> bytes:
    """Any recording -> 8 kHz μ-law, what a phone call would carry."""
    mono = to_mono_16k(audio)
    return ulaw_encode(resample((mono * 32767).astype(np.int16), 16_000, PHONE_RATE))


def wav_bytes(audio: tuple[int, np.ndarray]) -> bytes:
    rate, samples = audio
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return buffer.getvalue()


@dataclass
class Reply:
    audio: tuple[int, np.ndarray] | None  # what the call said, to play now
    heard_speech: bool = True  # False: the recording was silent, nothing was sent to Whisper


@dataclass
class PushToTalkCall:
    session: CallSession
    io: ClipIO
    visitor: str
    started: float = field(default_factory=time.monotonic)

    @property
    def ended(self) -> bool:
        return self.session.state == "ended"

    @property
    def status(self) -> str:
        if self.ended:
            reason = self.session.end_reason or ""
            return next((text for start, text in ENDINGS.items() if reason.startswith(start)), "The call ended.")
        return LISTENING.get(self.session.state, "")

    async def speak(self, recording: tuple[int, np.ndarray]) -> Reply:
        heard = await self.session.on_utterance(to_phone_line(recording))
        await self.settle()
        return Reply(self.io.take(), heard_speech=heard)

    async def type(self, text: str) -> Reply:
        await self.session.on_text(text)
        await self.settle()
        return Reply(self.io.take())

    async def key(self, digit: str) -> Reply:
        await self.session.on_dtmf(digit)
        await asyncio.sleep(0)  # let the keypad's turn start
        await self.settle()
        return Reply(self.io.take())

    async def settle(self, timeout: float = 120.0) -> None:
        """Wait for the call's work to finish (a keypad turn runs as a task)."""
        deadline = time.monotonic() + timeout
        while not (self.session.idle or self.ended) and time.monotonic() < deadline:
            await asyncio.sleep(0.05)


class PushToTalkLine:
    """The demo's phone line: calls keyed by browser session, with the same limits as the
    browser call (calls at once, calls per visitor per hour, call length)."""

    def __init__(self, deps: PhoneDeps, *, timing: Timing | None = None, secret: str | None = None) -> None:
        self.deps, self.settings = deps, deps.settings
        self.timing = timing or push_to_talk_timing(self.settings.phone_max_minutes * 60)
        self.calls: dict[str, PushToTalkCall] = {}
        self._recent: dict[str, deque[float]] = defaultdict(deque)
        self._secret = (secret or secrets.token_hex(16)).encode()

    def visitor(self, address: str) -> str:
        """Visitors are told apart by an HMAC of their address; the address isn't kept."""
        return hmac.new(self._secret, address.encode(), hashlib.sha256).hexdigest()[:16]

    def get(self, tab: str) -> PushToTalkCall | None:
        return self.calls.get(tab)

    def active(self) -> int:
        return sum(not call.ended for call in self.calls.values())

    async def start(self, tab: str, address: str) -> tuple[PushToTalkCall | None, str]:
        old = self.calls.get(tab)
        if old is not None and not old.ended:
            return old, ""
        self.calls.pop(tab, None)
        visitor = self.visitor(address)
        recent = self._recent[visitor]
        while recent and recent[0] < time.monotonic() - 3600:
            recent.popleft()
        if self.active() >= self.settings.phone_max_calls:
            return None, "All lines are busy right now. Please try again in a few minutes."
        if len(recent) >= self.settings.phone_calls_per_number_hour:
            return None, "You've made several calls in the last hour. Please try again later."
        recent.append(time.monotonic())
        clip_io = ClipIO()
        session = CallSession(f"WEB{secrets.token_hex(6)}", visitor, "web", clip_io, self.deps, self.timing)
        clip_io.session = session
        call = self.calls[tab] = PushToTalkCall(session, clip_io, visitor)
        await session.start()
        return call, ""

    async def hang_up(self, tab: str) -> None:
        call = self.calls.pop(tab, None)
        if call is not None:
            await call.session.end("caller hung up", hang_up=False)

    def sweep(self) -> None:
        """Forget calls that ended more than a while ago (the page shows how a call ended once)."""
        for tab, call in list(self.calls.items()):
            if call.ended and time.monotonic() - call.started > self.timing.max_seconds + 600:
                self.calls.pop(tab, None)
