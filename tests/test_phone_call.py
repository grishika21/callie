"""The phone call engine (callie/phone/call.py) with a fake transport and fake speech,
the scripted model, and the real Support Desk (HTTP + tokens, in-memory store: customer
1 Maria Gonzalez owns orders 5512 and 7781; customer 2 Sam Lee owns 6020)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import numpy as np
import pytest

from callie.auth import TokenIssuer
from callie.phone import call as callmod
from callie.phone.agent import PhoneRuntime
from callie.phone.audio import FRAME_BYTES, SILENCE_BYTE, ulaw_encode
from callie.phone.call import CallSession, PhoneDeps, Timing
from callie.speech.stt import Transcript
from callie.speech.tts import Speech
from tests.agent_fakes import ScriptedChatModel, call, say
from tests.desk_fakes import SECRET, Desk, running_desk
from tests.test_callie_agent import ask

pytestmark = pytest.mark.anyio


class FakeIO:
    def __init__(self) -> None:
        self.session: CallSession | None = None
        self.played: list[str] = []
        self.cleared = self.hung_up = self.transferred = 0

    async def play(self, mulaw: bytes, mark: str) -> None:
        self.played.append(mark)
        asyncio.get_running_loop().call_later(0.02, lambda: asyncio.ensure_future(self.session.on_mark(mark)))  # type: ignore[union-attr]

    async def clear(self) -> None:
        self.cleared += 1

    async def hang_up(self) -> None:
        self.hung_up += 1

    async def transfer(self) -> None:
        self.transferred += 1


class Ears:
    """Transcribes each utterance as the next scripted line (and notes the hint it got)."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.prompts: list[str | None] = []

    def transcribe(self, audio: tuple[int, np.ndarray], prompt: str | None = None) -> Transcript:
        self.prompts.append(prompt)
        return Transcript(self.lines.pop(0) if self.lines else "", 0.0, len(audio[1]) / audio[0])


class Mouth:
    def speak(self, text: str) -> Speech:
        return Speech(22_050, np.zeros(441, dtype=np.int16), 0.0)  # 20 ms


@pytest.fixture(scope="module")
def desk(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Desk]:
    with running_desk(tmp_path_factory.mktemp("call-desk")) as server:
        yield server


@pytest.fixture
def model() -> ScriptedChatModel:
    return ScriptedChatModel()


@pytest.fixture
async def runtime(desk: Desk, model: ScriptedChatModel) -> AsyncIterator[PhoneRuntime]:
    desk.reset()
    rt = PhoneRuntime(desk.settings, users=desk.users, model_factory=lambda: model, talktable_tools=[ask],
                      issuer=TokenIssuer(SECRET, desk.settings.desk_url))
    await rt.start()
    yield rt
    await rt.close()


_calls = iter(range(10**6))


async def start_call(runtime: PhoneRuntime, desk: Desk, *, transfer: bool = False,
                     timing: Timing | None = None) -> tuple[CallSession, FakeIO, Ears]:
    io, ears = FakeIO(), Ears()
    deps = PhoneDeps(runtime, ears, Mouth(), desk.settings, desk.audit, can_transfer=transfer)
    n = next(_calls)
    session = CallSession(f"CA{n}", f"hash-{n}", "…42", io, deps, timing)
    io.session = session
    await session.start()
    await settle(session)
    return session, io, ears


async def settle(session: CallSession, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    await asyncio.sleep(0.05)
    while not session.idle and session.state != "ended":
        assert time.monotonic() < deadline, "the call did not settle"
        await asyncio.sleep(0.02)


async def speak(session: CallSession, ears: Ears, text: str) -> None:
    """The caller says `text`: 0.6 s of voice, then 1 s of silence, as 20 ms μ-law frames."""
    ears.lines.append(text)
    rng = np.random.default_rng(0)
    voice = ulaw_encode((rng.normal(0, 6000, 4800)).astype(np.int16))
    audio = voice + bytes([SILENCE_BYTE]) * 8000
    for i in range(0, len(audio), FRAME_BYTES):
        await session.on_audio(audio[i:i + FRAME_BYTES])
    await settle(session)


async def verified(runtime: PhoneRuntime, desk: Desk, **kwargs: Any) -> tuple[CallSession, FakeIO, Ears]:
    session, io, ears = await start_call(runtime, desk, **kwargs)
    await speak(session, ears, "My order number is five five one two.")
    await speak(session, ears, "My last name is Gonzalez.")
    assert session.state == "help"
    return session, io, ears


# --- verification ----------------------------------------------------------------------------


async def test_greeting_and_spoken_verification(runtime: PhoneRuntime, desk: Desk) -> None:
    session, io, ears = await start_call(runtime, desk)
    assert session.spoken == [callmod.GREETING] and session.state == "order"
    await speak(session, ears, "My order number is five five one two.")
    assert session.state == "name" and session.spoken[-1] == callmod.ASK_NAME
    await speak(session, ears, "It's Gonzalez.")
    assert session.state == "help" and session.customer_id == 1
    assert session.spoken[-1] == "Thanks, Maria. You're verified. How can I help you today?"
    events = [(e["tool"], e["outcome"]) for e in desk.audit.read() if e["source"] == "phone"]
    assert events == [("call", "started"), ("verify", "ok")]


async def test_keypad_order_number(runtime: PhoneRuntime, desk: Desk) -> None:
    session, io, ears = await start_call(runtime, desk)
    for digit in "5512#":
        await session.on_dtmf(digit)
    await settle(session)
    assert session.state == "name" and session.turns[-1].heard == "(keypad 5512#)"


async def test_spelling_after_a_misheard_name(runtime: PhoneRuntime, desk: Desk) -> None:
    session, io, ears = await start_call(runtime, desk)
    await speak(session, ears, "5512")
    await speak(session, ears, "My last name is Gonzalves.")  # misheard
    assert session.state == "spell" and session.spoken[-1] == callmod.ASK_SPELLING
    await speak(session, ears, "G, O, N, Z, A, L, E, Z.")
    assert session.state == "help"
    assert ears.prompts == [callmod.HEARING["order"], callmod.HEARING["name"], callmod.HEARING["spell"]]


async def test_three_failures_end_the_call(runtime: PhoneRuntime, desk: Desk) -> None:
    session, io, ears = await start_call(runtime, desk)
    await speak(session, ears, "5512")
    await speak(session, ears, "Lee")
    await speak(session, ears, "L, E, E")
    assert session.state == "order" and session.spoken[-1] == callmod.RETRY
    await speak(session, ears, "6020")
    await speak(session, ears, "Gonzalez")  # Maria's name with Sam's order: no
    await session.wait()
    assert session.end_reason == "not verified" and io.hung_up == 1 and session.customer_id is None
    assert callmod.FAILED_GOODBYE in session.spoken


async def test_three_failures_with_a_transfer_number(runtime: PhoneRuntime, desk: Desk) -> None:
    session, io, ears = await start_call(runtime, desk, transfer=True)
    for line in ("5512", "Lee", "L, E, E", "5512", "Lee"):
        await speak(session, ears, line)
    await session.wait()
    assert io.transferred == 1 and io.hung_up == 0 and session.end_reason == "not verified, transferred"


# --- helping a verified caller ----------------------------------------------------------------------


async def test_own_order(runtime: PhoneRuntime, desk: Desk, model: ScriptedChatModel) -> None:
    session, io, ears = await verified(runtime, desk)
    model.script = [call("my_order", order_id=5512), say("Your order 5512 is on its way with SwiftShip.")]
    await speak(session, ears, "Where is my order?")
    turn = session.turns[-1]
    assert turn.reply == "Your order 5512 is on its way with SwiftShip." and turn.steps[0].ok


async def test_someone_elses_order_is_not_found(runtime: PhoneRuntime, desk: Desk, model: ScriptedChatModel) -> None:
    session, io, ears = await verified(runtime, desk)
    model.script = [call("my_order", order_id=6020), say("I couldn't find order 6020 on your account.")]
    await speak(session, ears, "What about order six oh two oh?")
    step = session.turns[-1].steps[0]
    assert step.ok is False and "couldn't find order 6020 on your account" in str(step.result)


async def test_ticket_confirmed_on_the_keypad(runtime: PhoneRuntime, desk: Desk, model: ScriptedChatModel) -> None:
    session, io, ears = await verified(runtime, desk)
    model.script = [call("open_my_ticket", order_id=5512, category="damaged item", subject="Book arrived damaged",
                         description="The book in my order arrived damaged.")]
    await speak(session, ears, "The book arrived damaged, can you report it?")
    assert session.spoken[-1].startswith("I'll open a damaged item ticket about your order 5512: “Book arrived "
                                         "damaged”. Shall I go ahead?") and desk.store.writes == 0
    await session.on_dtmf("1")
    await settle(session)
    ticket = desk.store.tickets[1451]
    assert (ticket.customer_id, ticket.channel, ticket.created_by, ticket.priority) == (1, "phone", "phone-agent", "normal")
    assert session.spoken[-1] == "Opened ticket 1451 for Maria Gonzalez."


async def test_ticket_declined_by_voice(runtime: PhoneRuntime, desk: Desk, model: ScriptedChatModel) -> None:
    session, io, ears = await verified(runtime, desk)
    model.script = [call("open_my_ticket", order_id=5512, category="delivery", subject="Late parcel",
                         description="My parcel is late.")]
    await speak(session, ears, "My parcel is late, please open a ticket.")
    await speak(session, ears, "No, thanks.")
    assert session.spoken[-1] == "Okay, I won't open that ticket." and desk.store.writes == 0


async def test_goodbye(runtime: PhoneRuntime, desk: Desk, model: ScriptedChatModel) -> None:
    session, io, ears = await verified(runtime, desk)
    model.script = [call("end_call")]
    await speak(session, ears, "That's all, thanks. Bye!")
    await session.wait()
    assert session.spoken[-1] == "Thanks for calling. Goodbye!" and io.hung_up == 1 and session.end_reason == "goodbye"


async def test_ask_for_a_person(runtime: PhoneRuntime, desk: Desk, model: ScriptedChatModel) -> None:
    session, io, ears = await verified(runtime, desk, transfer=True)
    await speak(session, ears, "Can I talk to a real person?")  # handled without the model (the script is empty)
    await session.wait()
    assert io.transferred == 1 and session.end_reason == "transferred"


async def test_the_agent_can_transfer_too(runtime: PhoneRuntime, desk: Desk, model: ScriptedChatModel) -> None:
    session, io, ears = await verified(runtime, desk, transfer=True)
    model.script = [call("transfer_to_person", reason="the caller is upset about a refund")]
    await speak(session, ears, "This is ridiculous, I've waited three weeks for my refund.")
    await session.wait()
    assert io.transferred == 1 and session.end_reason == "transferred"


async def test_no_one_to_transfer_to(runtime: PhoneRuntime, desk: Desk, model: ScriptedChatModel) -> None:
    session, io, ears = await verified(runtime, desk)
    model.script = [call("transfer_to_person", reason="wants a person"),
                    say("No one can take your call right now. Shall I open a ticket so our team calls you back?")]
    await speak(session, ears, "Can I talk to a real person?")
    assert io.transferred == 0 and session.state == "help" and "calls you back" in session.spoken[-1]


async def test_barge_in(runtime: PhoneRuntime, desk: Desk) -> None:
    session, io, ears = await start_call(runtime, desk)
    session._playing, session._playing_until = "say-x", time.monotonic() + 30  # a long reply is playing
    rng = np.random.default_rng(1)
    voice = ulaw_encode((rng.normal(0, 6000, 8000)).astype(np.int16))  # 1 s of the caller talking over it
    for i in range(0, len(voice), FRAME_BYTES):
        await session.on_audio(voice[i:i + FRAME_BYTES])
    assert io.cleared == 1 and session._playing is None


async def test_silence(runtime: PhoneRuntime, desk: Desk) -> None:
    session, io, ears = await start_call(runtime, desk, timing=Timing(silence_nudge=0.3, silence_end=0.3))
    await session.wait()
    assert callmod.STILL_THERE in session.spoken and session.end_reason == "silence" and io.hung_up == 1


async def test_time_limit(runtime: PhoneRuntime, desk: Desk) -> None:
    session, io, ears = await start_call(runtime, desk, timing=Timing(max_seconds=0.3))
    await session.wait()
    assert session.end_reason == "time limit" and callmod.TOO_LONG in session.spoken


async def test_caller_hangs_up(runtime: PhoneRuntime, desk: Desk) -> None:
    session, io, ears = await start_call(runtime, desk)
    await session.on_stop()
    assert session.end_reason == "caller hung up" and io.hung_up == 0


async def test_keypad_zero_for_a_person(runtime: PhoneRuntime, desk: Desk) -> None:
    session, io, ears = await start_call(runtime, desk, transfer=True)
    await session.on_dtmf("0")
    await session.wait()
    assert io.transferred == 1 and session.end_reason == "transferred (keypad 0)"
