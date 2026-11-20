"""One phone call, from greeting to goodbye. Transport-agnostic: the Twilio gateway feeds
it Media Streams events, the simulator feeds it scripted audio, both through `CallIO`.

    greeting -> order number (spoken, or keypad + #) -> last name (spoken, then spelled)
             -> verified by the Support Desk -> help (the caller agent) -> goodbye

* Verification is a fixed script, no LLM. Three tries per call (counted by the desk); the
  caller never hears which part was wrong; after that, a transfer or a polite goodbye.
* The caller can talk over a reply (barge-in): playback stops and their words count.
* Keypad: digits for the order number, 1 / 2 to confirm or decline a ticket, 0 for a person.
* "One moment" when a turn takes a while, "Still looking" while TalkTable works,
  "Are you still there?" after a silence, and a hard limit on the call's length.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from copilot.audit import AuditLog
from copilot.config import CopilotSettings
from copilot.phone.agent import GATEWAY_ACCOUNT, CallActions, PhoneRuntime
from copilot.phone.audio import PHONE_RATE, from_phone, to_phone
import re

from copilot.phone.verify import extract_last_name, extract_order_number, extract_spelling
from copilot.session import CopilotSession
from copilot.speech.stt import Transcriber
from copilot.speech.text import speakable
from copilot.speech.tts import Speaker
from copilot.speech.vad import Endpointer

logger = logging.getLogger(__name__)

State = Literal["greeting", "order", "name", "spell", "help", "ended"]

GREETING = ("Thanks for calling the bookstore. I'm an automated assistant, and this call is handled by A I and "
            "logged. To look up your order, what's your order number? You can also type it on the keypad, then "
            "press pound.")
ASK_NAME = "Thanks. And what's the last name on the account?"
ASK_SPELLING = "I couldn't match that. Could you spell your last name for me, letter by letter?"
RETRY = "I still couldn't match those details. Let's try once more. What's your order number?"
NO_ORDER = "Sorry, I didn't catch an order number. Please say it again, or type it on the keypad, then press pound."
FAILED_TRANSFER = "I'm sorry, I couldn't verify your details. Let me connect you with our support team."
FAILED_GOODBYE = ("I'm sorry, I couldn't verify your details. Please check the order number in your confirmation "
                  "email and call us again. Goodbye.")
WELCOME = "Thanks, {name}. You're verified. How can I help you today?"
ONE_MOMENT, STILL_LOOKING = "One moment.", "Still looking, thanks for waiting."
# What whisper should expect at each step. A general hint (carriers, ticket categories) makes
# it "hear" those words in spelled letters: the browser test got "Ticket categories." for
# "V, I, R, K". P0's 24/24 spelled names used the spelling hint.
HEARING = {
    "order": "The caller says their order number, for example: My order number is 5512.",
    "name": "The caller says their last name, for example: My last name is Virk.",
    "spell": "The caller spells their last name letter by letter, for example: V, I, R, K.",
}
STILL_THERE, SILENT_GOODBYE = "Are you still there?", "I'll end the call now. Thanks for calling. Goodbye."
TOO_LONG = "We've reached the time limit for this call. Thanks for calling, goodbye."
TRANSFERRING = "Let me connect you with someone from our support team. Please hold."
GOODBYE = "Thanks for calling. Goodbye!"
_GOODBYE = re.compile(r"\b(?:good ?bye|bye|that'?s all|that is all|nothing else|no,? that'?s it|have a (?:good|nice) "
                      r"(?:day|one))\b", re.IGNORECASE)


_PERSON = re.compile(r"\b(?:talk|speak|connect|transfer|put me through|get me)\b.*\b(?:person|human|agent|"
                     r"representative|operator|someone|somebody|staff)\b|\b(?:real|live|actual) (?:person|human)\b",
                     re.IGNORECASE)


def wants_a_person(text: str) -> bool:
    """"Can I talk to a real person?", "Put me through to an agent." (the standard phone-menu request)."""
    return bool(_PERSON.search(text))


def is_goodbye(text: str) -> bool:
    """A caller wrapping up ("That's all, thank you. Goodbye!"), not a question."""
    return bool(_GOODBYE.search(text)) and "?" not in text and len(text.split()) <= 10
NO_ONE = "No one can take the call right now. I can open a ticket so our team calls you back. Would you like that?"


class CallIO(Protocol):
    async def play(self, mulaw: bytes, mark: str) -> None: ...  # queue audio, then a mark when it has played
    async def clear(self) -> None: ...  # stop what is playing (barge-in)
    async def hang_up(self) -> None: ...
    async def transfer(self) -> None: ...


@dataclass
class PhoneDeps:
    runtime: PhoneRuntime
    transcriber: Transcriber
    speaker: Speaker
    settings: CopilotSettings
    audit: AuditLog | None = None
    can_transfer: bool = False


@dataclass
class Timing:
    silence_nudge: float = 12.0  # seconds of silence before "Are you still there?"
    silence_end: float = 12.0  # then this long before hanging up
    max_seconds: float = 600.0
    filler_after: float = 1.5
    still_looking_every: float = 8.0
    keypad_pause: float = 3.0  # digits without # are taken after this pause
    barge_in_speech: float = 0.45  # seconds of caller speech that stop our playback


@dataclass
class Turn:
    """One caller utterance and what the call did with it (for the simulator and tests)."""

    heard: str
    reply: str
    state: State
    seconds: float  # from the end of the caller's speech to the first audio (maybe "One moment")
    steps: list = field(default_factory=list)
    reply_seconds: float | None = None  # ...to the actual answer


class CallSession:
    def __init__(self, call_id: str, caller: str, caller_label: str, io: CallIO, deps: PhoneDeps,
                 timing: Timing | None = None) -> None:
        self.call_id, self.caller, self.caller_label = call_id, caller, caller_label
        self.io, self.deps, self.timing = io, deps, timing or Timing()
        self.state: State = "greeting"
        self.endpointer = Endpointer(end_silence=0.7)
        self.session: CopilotSession | None = None
        self.actions = CallActions()
        self.customer_id: int | None = None
        self.turns: list[Turn] = []
        self.spoken: list[str] = []  # everything the call said, in order
        self.end_reason: str | None = None
        self._order_id: int | None = None
        self._busy = False
        self._playing: str | None = None  # the mark of the audio playing now
        self._playing_until = 0.0
        self._marks = 0
        self._digits, self._last_digit = "", 0.0
        self._started = self._activity = time.monotonic()
        self._nudged = False
        self._tasks: set[asyncio.Task] = set()
        self._done = asyncio.Event()

    # --- events from the transport --------------------------------------------------------

    async def start(self) -> None:
        self._audit("call", "started")
        await self.say(GREETING)
        self.state = "order"
        self._spawn(self._watchdog())

    async def on_audio(self, mulaw: bytes) -> None:
        if self.state == "ended" or self._busy:
            return  # one thing at a time: speech during a turn isn't collected
        rate, pcm = from_phone(mulaw)
        utterance = self.endpointer.feed((rate, pcm))
        if self._playing and self.endpointer.speaking_for >= self.timing.barge_in_speech:
            await self._stop_playing()  # the caller talks over us: listen instead
        if self.endpointer.speaking_for:
            self._activity = time.monotonic()
        # an answer that ends as our audio ends counts, even before Twilio's mark arrives
        if utterance is not None and (not self._playing or time.monotonic() > self._playing_until):
            self._playing = None
            self._spawn(self._handle(utterance, time.monotonic()))

    async def on_dtmf(self, digit: str) -> None:
        if self.state == "ended" or self._busy:
            return
        self._activity = self._last_digit = time.monotonic()
        if digit == "0" and self.state not in ("greeting",):
            self._spawn(self._transfer_or_offer("transferred (keypad 0)"))
        elif self.state == "order":
            if digit == "#":
                self._spawn(self._take_keypad_order())
            elif digit.isdigit():
                self._digits += digit
        elif self.state == "help" and self.session is not None and self.session.pending is not None and digit in "12":
            self._spawn(self._handle_text("yes" if digit == "1" else "no", time.monotonic()))

    async def on_mark(self, name: str) -> None:
        if name == self._playing:
            self._playing = None
            self._activity = time.monotonic()  # the caller's turn starts now
            self.endpointer.reset()

    async def on_stop(self) -> None:
        await self.end("caller hung up", hang_up=False)

    async def wait(self) -> str | None:
        await self._done.wait()
        return self.end_reason

    @property
    def idle(self) -> bool:
        return not self._busy and self._playing is None and not self._tasks_running()

    # --- speaking ---------------------------------------------------------------------------

    async def say(self, text: str) -> None:
        if self.state == "ended":
            return
        speech = await asyncio.to_thread(self.deps.speaker.speak, speakable(text, max_sentences=3))
        self.spoken.append(text)
        if speech is None:
            return
        mulaw = await asyncio.to_thread(to_phone, speech.samples, speech.sample_rate)
        self._marks += 1
        self._playing = mark = f"say-{self._marks}"
        self._playing_until = time.monotonic() + len(mulaw) / PHONE_RATE
        await self.io.play(mulaw, mark)

    async def _stop_playing(self) -> None:
        self._playing = None
        await self.io.clear()

    async def _until_quiet(self, extra: float = 1.0) -> None:
        while self._playing and time.monotonic() < self._playing_until + extra:
            await asyncio.sleep(0.05)

    # --- turns --------------------------------------------------------------------------------

    async def _handle(self, utterance: np.ndarray, ended_at: float) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            heard = await asyncio.to_thread(self.deps.transcriber.transcribe, (16_000, utterance),
                                            HEARING.get(self.state))
            logger.info("call %s heard %r (%.1f s)", self.call_id[-6:], heard.text, heard.seconds)
            if heard.text.strip():
                await self._dispatch(heard.text.strip(), ended_at)
        except Exception:
            logger.exception("call %s: turn failed", self.call_id[-6:])
            await self.say("Sorry, something went wrong on my side. Could you say that again?")
        finally:
            self._busy = False
            self.endpointer.reset()

    async def _handle_text(self, text: str, ended_at: float) -> None:
        """A keypad answer (1 / 2) handled like speech."""
        if self._busy:
            return
        self._busy = True
        try:
            await self._dispatch(text, ended_at)
        finally:
            self._busy = False

    async def _dispatch(self, text: str, ended_at: float) -> None:
        self._nudged = False
        before = len(self.spoken)
        if self.state == "order":
            await self._take_order(extract_order_number(text))
        elif self.state == "name":
            await self._verify(extract_last_name(text))
        elif self.state == "spell":
            await self._verify(extract_spelling(text))
        elif self.state == "help" and wants_a_person(text) and self.deps.can_transfer:
            self.turns.append(Turn(text, TRANSFERRING, self.state, time.monotonic() - ended_at))
            await self._transfer_or_offer("transferred")  # no model needed; without a number the agent offers a callback
            return
        elif self.state == "help" and is_goodbye(text) and (self.session is None or self.session.pending is None):
            await self.say(GOODBYE)  # a plain goodbye needs no model
            self.turns.append(Turn(text, GOODBYE, self.state, time.monotonic() - ended_at))
            await self._until_quiet()
            await self.end("goodbye")
            return
        elif self.state == "help":
            await self._agent_turn(text, ended_at)
            return
        self.turns.append(Turn(text, " ".join(self.spoken[before:]), self.state, time.monotonic() - ended_at))

    async def _take_keypad_order(self) -> None:
        digits, self._digits = self._digits, ""
        self._busy = True
        try:
            started = time.monotonic()
            await self._take_order(int(digits) if digits.isdigit() and len(digits) >= 2 else None)
            self.turns.append(Turn(f"(keypad {digits}#)", self.spoken[-1], self.state, time.monotonic() - started))
        finally:
            self._busy = False

    async def _take_order(self, order_id: int | None) -> None:
        if order_id is None:
            await self.say(NO_ORDER)
            return
        self._order_id, self.state = order_id, "name"
        await self.say(ASK_NAME)

    async def _verify(self, last_name: str) -> None:
        assert self._order_id is not None
        try:
            result = await self.deps.runtime.verify(self.call_id, self.caller, self._order_id, last_name,
                                                    spelled=self.state == "spell")
        except PermissionError:  # the desk's tries per call / failures per number ran out
            result = {"verified": False, "tries_left": 0}
        if result.get("verified"):
            self.customer_id = result["customer_id"]
            agent = await self.deps.runtime.agent_for_call(self.call_id, self.caller, self.customer_id,
                                                           result["first_name"], self.actions,
                                                           can_transfer=self.deps.can_transfer,
                                                           verified_order=self._order_id)
            self.session = CopilotSession(agent, self.deps.audit)
            self.state = "help"
            self._audit("verify", "ok")
            await self.say(WELCOME.format(name=result["first_name"]))
            return
        self._audit("verify", "failed")
        if result.get("tries_left", 0) <= 0:
            await self._give_up()
        elif self.state == "name":
            self.state = "spell"
            await self.say(ASK_SPELLING)
        else:
            self.state, self._order_id = "order", None
            await self.say(RETRY)

    async def _give_up(self) -> None:
        if self.deps.can_transfer:
            await self.say(FAILED_TRANSFER)
            await self._until_quiet()
            await self.io.transfer()
            await self.end("not verified, transferred", hang_up=False)
        else:
            await self.say(FAILED_GOODBYE)
            await self._until_quiet()
            await self.end("not verified")

    async def _agent_turn(self, text: str, ended_at: float) -> None:
        assert self.session is not None
        work = asyncio.ensure_future(self.session.hear(text))
        first_audio: float | None = None
        filler, last_still = False, time.monotonic()
        while not work.done():
            await asyncio.wait({work}, timeout=0.25)
            if work.done():
                break
            waited = time.monotonic() - ended_at
            running = self.session.agent.running_tool(self.session.thread_id)
            if not filler and waited > self.timing.filler_after:
                filler = True
                first_audio = first_audio or time.monotonic()
                await self.say(ONE_MOMENT)
            elif running and running[0] == "ask" and time.monotonic() - last_still > self.timing.still_looking_every:
                last_still = time.monotonic()
                await self.say(STILL_LOOKING)
        result = work.result()
        answered = time.monotonic()
        await self._until_quiet(extra=0.3)  # let a filler finish
        first_audio = first_audio or time.monotonic()
        await self.say(result.reply)
        self.turns.append(Turn(text, result.reply, self.state, first_audio - ended_at, result.steps,
                               reply_seconds=answered - ended_at))
        if self.actions.end:
            await self._until_quiet()
            await self.end("goodbye")
        elif self.actions.transfer:
            await self._until_quiet()
            await self.io.transfer()
            await self.end("transferred", hang_up=False)

    async def _transfer_or_offer(self, reason: str) -> None:
        if self.deps.can_transfer:
            await self.say(TRANSFERRING)
            await self._until_quiet()
            await self.io.transfer()
            await self.end(reason, hang_up=False)
        else:
            await self.say(NO_ONE)

    # --- housekeeping ------------------------------------------------------------------------

    async def _watchdog(self) -> None:
        while self.state != "ended":
            await asyncio.sleep(0.25)
            now = time.monotonic()
            if now - self._started > self.timing.max_seconds:
                await self.say(TOO_LONG)
                await self._until_quiet()
                await self.end("time limit")
                return
            if self.state == "order" and self._digits and now - self._last_digit > self.timing.keypad_pause:
                await self._take_keypad_order()
                continue
            if not self.idle_for_watchdog:
                continue
            quiet = now - self._activity
            if not self._nudged and quiet > self.timing.silence_nudge:
                self._nudged = True
                await self.say(STILL_THERE)
            elif self._nudged and quiet > self.timing.silence_end:
                await self.say(SILENT_GOODBYE)
                await self._until_quiet()
                await self.end("silence")
                return

    @property
    def idle_for_watchdog(self) -> bool:
        return not self._busy and self._playing is None and not self.endpointer.speaking_for

    def _tasks_running(self) -> bool:
        return any(not t.done() for t in self._tasks if t.get_coro().__name__ != "_watchdog")  # type: ignore[union-attr]

    def _spawn(self, coro) -> None:  # type: ignore[no-untyped-def]
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def end(self, reason: str, *, hang_up: bool = True) -> None:
        if self.state == "ended":
            return
        self.state, self.end_reason = "ended", reason
        self._audit("call", "ended", detail=reason)
        if hang_up:
            with contextlib.suppress(Exception):
                await self.io.hang_up()
        current = asyncio.current_task()
        for task in list(self._tasks):
            if task is not current:
                task.cancel()
        self._done.set()

    def _audit(self, tool: str, outcome: str, detail: str | None = None) -> None:
        if self.deps.audit is None:
            return
        self.deps.audit.write(user=GATEWAY_ACCOUNT, role="phone", tool=tool, source="phone", outcome=outcome,
                              detail=detail, latency_ms=(time.monotonic() - self._started) * 1000,
                              context={"call": self.call_id, "caller": self.caller_label,
                                       "customer_id": self.customer_id})
