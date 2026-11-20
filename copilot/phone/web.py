"""The public demo's "Call the bookstore" page: the browser is the phone.

The microphone's audio is squeezed into 8 kHz μ-law and fed to the same CallSession that a
Twilio call drives, so a browser call behaves exactly like a phone call: greeting,
verification (order number + last name, spelled if needed), help, confirmation, goodbye.
Replies come back as audio the page plays, one after another. An on-screen keypad stands
in for the phone's buttons. Unlike a phone line, the browser's own speaker can echo into
its microphone, so the page doesn't listen while a reply plays (no talking over it).

Limits for a public link: calls at once (shared with phone calls), calls per visitor per
hour, and the call's length. Every handler runs on the server's event loop, which also
holds the phone runtime (its TalkTable stdio session).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import html
import logging
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import gradio as gr
import numpy as np

from copilot.phone.audio import PHONE_RATE, resample, ulaw_decode, ulaw_encode
from copilot.phone.call import CallSession, PhoneDeps, Timing
from copilot.ui import _step_message

logger = logging.getLogger(__name__)

_PRIVATE: dict[str, Any] = {"api_visibility": "private"}
KEYS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "0", "#"]

INTRO = """\
### Call the bookstore's AI phone assistant
Press **Record** below (allow the microphone), then just talk, as on a phone call. It answers by voice.

**Demo customer:** order number **5512**, last name **Virk** (spell it if asked: V, I, R, K).
Then try: *"Where is my order?"* · *"The book arrived damaged, please open a ticket."* (say *yes* or press **1**)
· *"Do you have any books by Erik Gross?"* · *"What's the status of order 7888?"* (someone else's) · *"Goodbye."*

The keypad works like a phone's: type the order number then **#**, **1** = yes, **2** = no, **0** = a person.
"""

HOW_IT_WORKS = """\
- **Same engine as the real phone line** (Twilio Media Streams): your voice is turned into 8 kHz phone audio,
  transcribed by Whisper, and handled by a call script plus an AI agent.
- **Verification first**, without any AI: the order number and the last name must match the Support Desk's
  records. Three tries; it never says which part was wrong.
- **Only your own data:** after verification, every request to the Support Desk carries a token for that one
  customer. Someone else's order is simply "not found". Book questions go to a TalkTable that can only see the
  book catalog.
- **Tickets need your "yes"** (or keypad 1). They're written with channel `phone` and created by `phone-agent`.
- **Limits:** calls at once, calls per visitor per hour, call length. The data is fictional and resets when
  the demo restarts.

Source and design notes: [github.com/grishika21/talktable](https://github.com/grishika21/talktable)
"""


class WebIO:
    """CallIO for a browser call: replies queue up and are handed to the page one at a time;
    a reply's mark fires when it would have finished playing."""

    def __init__(self) -> None:
        self.session: CallSession | None = None
        self.queue: list[tuple[np.ndarray, str]] = []
        self.audible_until = 0.0
        self.hung_up = self.transferred = False

    async def play(self, mulaw: bytes, mark: str) -> None:
        self.queue.append((ulaw_decode(mulaw), mark))

    async def clear(self) -> None:
        self.queue.clear()

    async def hang_up(self) -> None:
        self.hung_up = True

    async def transfer(self) -> None:
        self.transferred = True

    def take(self) -> tuple[int, np.ndarray] | None:
        """The next audio for the page, once what it played before has finished."""
        now = time.monotonic()
        if not self.queue or now < self.audible_until:
            return None
        clips, self.queue = self.queue, []
        gap = np.zeros(int(0.25 * PHONE_RATE), dtype=np.int16)
        audio = np.concatenate([x for pcm, _ in clips for x in (pcm, gap)])
        seconds = len(audio) / PHONE_RATE
        self.audible_until = now + seconds + 0.4  # + the page's polling delay
        loop = asyncio.get_running_loop()
        for _, mark in clips:  # the call learns the audio has played
            loop.call_later(seconds + 0.4, lambda m=mark: asyncio.ensure_future(self._marked(m)))
        return PHONE_RATE, audio

    async def _marked(self, mark: str) -> None:
        if self.session is not None:
            await self.session.on_mark(mark)


@dataclass
class WebCall:
    session: CallSession
    io: WebIO
    visitor: str
    started: float = field(default_factory=time.monotonic)


def _escape(text: str) -> str:
    return html.escape(text, quote=False)


def transcript(session: CallSession) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if session.spoken:
        messages.append({"role": "assistant", "content": _escape(session.spoken[0])})
    for turn in session.turns:
        messages.append({"role": "user", "content": _escape(turn.heard)})
        messages.extend(_step_message(step) for step in turn.steps)
        messages.append({"role": "assistant", "content": _escape(turn.reply)})
    return messages


def status_html(text: str, live: bool = False) -> str:
    dot = "#3fb950" if live else "#8b949e"
    return f'<div class="cp-status"><b style="color:{dot}">●</b> {_escape(text)}</div>'


class PhoneDemo:
    """Browser calls, shared by every tab, on the server's event loop."""

    def __init__(self, deps: PhoneDeps, *, timing: Timing | None = None, other_calls: Callable[[], int] = lambda: 0,
                 secret: str | None = None) -> None:
        settings = deps.settings
        self.deps, self.settings = deps, settings
        self.timing = timing or Timing(max_seconds=settings.phone_max_minutes * 60)
        self.other_calls = other_calls  # phone calls in progress (they share the limit)
        self.calls: dict[str, WebCall] = {}
        self._recent: dict[str, deque[float]] = defaultdict(deque)
        self._secret = (secret or secrets.token_hex(16)).encode()

    def visitor(self, request: gr.Request) -> str:
        forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        address = forwarded or (request.client.host if request.client else "?")
        return hmac.new(self._secret, address.encode(), hashlib.sha256).hexdigest()[:16]

    # --- handlers --------------------------------------------------------------------------------

    async def start(self, request: gr.Request) -> tuple[Any, Any, Any]:
        tab = request.session_hash or ""
        if tab in self.calls:
            return gr.skip(), gr.skip(), gr.Timer(active=True)
        visitor = self.visitor(request)
        recent = self._recent[visitor]
        while recent and recent[0] < time.monotonic() - 3600:
            recent.popleft()
        if len(self.calls) + self.other_calls() >= self.settings.phone_max_calls:
            return status_html("All lines are busy right now. Please try again in a few minutes."), [], gr.skip()
        if len(recent) >= self.settings.phone_calls_per_number_hour:
            return status_html("You've made several calls in the last hour. Please try again later."), [], gr.skip()
        recent.append(time.monotonic())
        io = WebIO()
        session = CallSession(f"WEB{secrets.token_hex(6)}", visitor, "web", io, self.deps, self.timing)
        io.session = session
        self.calls[tab] = WebCall(session, io, visitor)
        await session.start()
        return status_html("Connected. Listen to the greeting, then talk.", live=True), transcript(session), \
            gr.Timer(active=True)

    async def hear(self, chunk: tuple[int, np.ndarray] | None, request: gr.Request) -> None:
        call = self.calls.get(request.session_hash or "")
        if chunk is None or call is None or call.session.state == "ended":
            return
        if time.monotonic() < call.io.audible_until or call.io.queue:
            return  # a reply is playing: the browser's speaker would echo into its microphone
        rate, samples = chunk
        data = np.asarray(samples)
        if data.ndim == 2:
            data = data.mean(axis=1)
        pcm = resample(np.clip(data, -32768, 32767).astype(np.int16), rate, PHONE_RATE)
        await call.session.on_audio(ulaw_encode(pcm))

    async def key(self, digit: str, request: gr.Request) -> None:
        call = self.calls.get(request.session_hash or "")
        if call is not None and call.session.state != "ended":
            await call.session.on_dtmf(digit)

    async def poll(self, request: gr.Request) -> tuple[Any, Any, Any, Any]:
        """Every half second during a call: new audio, the transcript, the status."""
        tab = request.session_hash or ""
        call = self.calls.get(tab)
        if call is None:
            return gr.skip(), gr.skip(), gr.skip(), gr.Timer(active=False)
        audio = call.io.take()
        session = call.session
        if session.state == "ended" and audio is None and not call.io.queue:
            self.calls.pop(tab, None)
            ending = {"goodbye": "The call ended. Thanks for trying it!", "transferred": "Transferred to a person "
                      "(in this demo, nobody is on the line).", "not verified": "The call ended: the details "
                      "didn't match. Try order 5512, last name Virk."}.get(session.end_reason or "", "The call ended.")
            return gr.skip(), transcript(session), status_html(ending), gr.Timer(active=False)
        state = {"order": "Listening: your order number", "name": "Listening: your last name",
                 "spell": "Listening: spell your last name", "help": "Verified. Listening"}.get(session.state, "")
        busy = "Thinking…" if not session.idle and session.state == "help" else state
        return audio if audio is not None else gr.skip(), transcript(session), status_html(busy, live=True), \
            gr.skip()

    async def hang_up(self, request: gr.Request) -> tuple[Any, Any]:
        call = self.calls.pop(request.session_hash or "", None)
        if call is not None:
            await call.session.end("caller hung up", hang_up=False)
        return status_html("Call ended. Press Record to call again."), gr.Timer(active=False)

    async def close(self, request: gr.Request) -> None:
        await self.hang_up(request)


def _keypad(demo: PhoneDemo, digit: str) -> Callable[..., Any]:
    async def press(request: gr.Request) -> None:  # Gradio passes the request by this annotation
        await demo.key(digit, request)

    return press


def build_demo(demo: PhoneDemo) -> gr.Blocks:
    with gr.Blocks(title="TalkTable Copilot: call the bookstore", fill_height=True) as app:
        gr.HTML('<div class="tt-brand"><div class="tt-logo">T</div><h1>Talk<span>Table</span> Copilot</h1></div>')
        gr.Markdown(INTRO)
        status = gr.HTML(status_html("Press Record to call."))
        with gr.Row(equal_height=True):
            phone = gr.Audio(sources=["microphone"], type="numpy", streaming=True, label="Your phone (Record = call, "
                             "Stop = hang up)", scale=3)
            voice = gr.Audio(label="The bookstore", autoplay=True, interactive=False, scale=2)
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(show_label=False, height=380, latex_delimiters=[], elem_id="cp-chat",
                                  placeholder="The call's transcript appears here, with every tool the agent uses.")
            with gr.Column(scale=1, min_width=210):
                buttons = []
                for row in range(4):
                    with gr.Row():
                        for digit in KEYS[row * 3:row * 3 + 3]:
                            buttons.append((gr.Button(digit, size="sm", min_width=50), digit))
        with gr.Accordion("How it works", open=False):
            gr.Markdown(HOW_IT_WORKS)
        timer = gr.Timer(0.5, active=False)

        phone.start_recording(demo.start, None, [status, chat, timer], **_PRIVATE)
        phone.stream(demo.hear, [phone], None, stream_every=0.5, time_limit=None, concurrency_limit=None,
                     show_progress="hidden", **_PRIVATE)
        phone.stop_recording(demo.hang_up, None, [status, timer], **_PRIVATE)
        timer.tick(demo.poll, None, [voice, chat, status, timer], show_progress="hidden", **_PRIVATE)
        for button, digit in buttons:
            button.click(_keypad(demo, digit), None, None, **_PRIVATE)
        app.unload(demo.close)
    return app
