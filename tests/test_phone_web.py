"""The public demo's browser call (copilot/phone/web.py): the same CallSession as a phone
call, fed from the page's microphone; fake speech, the scripted model, the real desk."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any

import gradio as gr
import numpy as np
import pytest

from copilot.auth import TokenIssuer
from copilot.phone.agent import PhoneRuntime
from copilot.phone.call import PhoneDeps, Timing
from copilot.phone.web import PhoneDemo, WebIO, build_demo
from tests.agent_fakes import ScriptedChatModel, call, say
from tests.desk_fakes import SECRET, Desk, running_desk
from tests.test_copilot_agent import ask
from tests.test_phone_call import Ears, Mouth

pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
def desk(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Desk]:
    with running_desk(tmp_path_factory.mktemp("web-desk")) as server:
        yield server


@pytest.fixture
def model() -> ScriptedChatModel:
    return ScriptedChatModel()


@pytest.fixture
async def demo(desk: Desk, model: ScriptedChatModel) -> AsyncIterator[tuple[PhoneDemo, Ears]]:
    desk.reset()
    settings = desk.settings.model_copy(update={"phone_max_calls": 2, "phone_calls_per_number_hour": 2})
    runtime = PhoneRuntime(settings, users=desk.users, model_factory=lambda: model, talktable_tools=[ask],
                           issuer=TokenIssuer(SECRET, settings.desk_url))
    await runtime.start()
    ears = Ears()
    deps = PhoneDeps(runtime, ears, Mouth(), settings, desk.audit)
    yield PhoneDemo(deps, timing=Timing(silence_nudge=60, silence_end=60), secret="test"), ears
    await runtime.close()


def visitor(tab: str = "tab-1", ip: str = "203.0.113.7") -> Any:
    return SimpleNamespace(session_hash=tab, headers={"x-forwarded-for": f"{ip}, 10.0.0.1"}, client=None)


async def speak(demo: PhoneDemo, ears: Ears, text: str, tab: str = "tab-1") -> None:
    """The visitor says `text`: browser-rate audio (48 kHz), voice then silence, in 0.5 s chunks."""
    ears.lines.append(text)
    session = demo.calls[tab].session
    await settle(demo, tab)
    rng = np.random.default_rng(0)
    audio = np.concatenate([rng.normal(0, 6000, 24_000), np.zeros(48_000)]).astype(np.int16)
    for i in range(0, len(audio), 24_000):
        await demo.hear((48_000, audio[i:i + 24_000]), visitor(tab))
    await settle(demo, tab)
    assert session.state != "greeting"


async def settle(demo: PhoneDemo, tab: str = "tab-1") -> None:
    """Let the page 'play' everything queued, as its timer would."""
    call_ = demo.calls[tab]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        call_.io.audible_until = 0  # the page finished playing
        await demo.poll(visitor(tab))
        await asyncio.sleep(0.05)
        if call_.session.idle and not call_.io.queue:
            return
    raise AssertionError("the call did not settle")


async def test_nothing_is_public(demo: tuple[PhoneDemo, Ears]) -> None:
    info = build_demo(demo[0]).get_api_info()
    assert not info["named_endpoints"] and not info.get("unnamed_endpoints")


async def test_a_browser_call(demo: tuple[PhoneDemo, Ears], desk: Desk, model: ScriptedChatModel) -> None:
    phone, ears = demo
    status, chat, timer = await phone.start(visitor())
    assert "Connected" in status and chat[0]["content"].startswith("Thanks for calling the bookstore")
    first = phone.calls["tab-1"].io.take()
    assert first is not None and first[0] == 8000  # the greeting, as phone audio
    await speak(phone, ears, "My order number is five five one two.")
    await speak(phone, ears, "Gonzalez")
    assert phone.calls["tab-1"].session.state == "help"
    model.script = [call("my_order", order_id=5512), say("Your order 5512 is on its way.")]
    await speak(phone, ears, "Where is my order?")
    audio, chat, status, _ = await phone.poll(visitor())
    assert chat[-1] == {"role": "assistant", "content": "Your order 5512 is on its way."}
    assert any(m.get("metadata", {}).get("title", "").startswith("🔧 my_order") for m in chat)
    assert "Verified" in status


async def test_the_keypad(demo: tuple[PhoneDemo, Ears]) -> None:
    phone, ears = demo
    await phone.start(visitor())
    await settle(phone)
    for digit in "5512#":
        await phone.key(digit, visitor())
    await settle(phone)
    assert phone.calls["tab-1"].session.state == "name"


async def test_the_page_doesnt_hear_its_own_reply(demo: tuple[PhoneDemo, Ears]) -> None:
    phone, ears = demo
    await phone.start(visitor())
    io: WebIO = phone.calls["tab-1"].io
    assert io.take() is not None and io.audible_until > time.monotonic()  # the greeting is playing
    loud = (np.random.default_rng(1).normal(0, 6000, 48_000)).astype(np.int16)
    await phone.hear((48_000, loud), visitor())
    assert phone.calls["tab-1"].session.endpointer.speaking_for == 0  # ignored: it was the speaker's echo


async def test_limits(demo: tuple[PhoneDemo, Ears]) -> None:
    phone, _ = demo
    await phone.start(visitor("a", ip="198.51.100.1"))
    await phone.start(visitor("b", ip="198.51.100.2"))
    busy, _, _ = await phone.start(visitor("c", ip="198.51.100.3"))
    assert "All lines are busy" in busy  # 2 calls at once in this test
    await phone.hang_up(visitor("a"))
    await phone.hang_up(visitor("b"))
    await phone.start(visitor("d", ip="198.51.100.9"))
    await phone.hang_up(visitor("d"))
    await phone.start(visitor("e", ip="198.51.100.9"))
    await phone.hang_up(visitor("e"))
    again, _, _ = await phone.start(visitor("f", ip="198.51.100.9"))
    assert "several calls in the last hour" in again  # 2 per visitor per hour in this test


async def test_hang_up_and_goodbye(demo: tuple[PhoneDemo, Ears]) -> None:
    phone, ears = demo
    await phone.start(visitor())
    await speak(phone, ears, "5512")
    await speak(phone, ears, "Gonzalez")
    await speak(phone, ears, "That's all, thanks. Goodbye!")
    _, chat, status, timer = await phone.poll(visitor())
    assert "The call ended. Thanks for trying it!" in status and "tab-1" not in phone.calls
    status, _ = await phone.hang_up(visitor())  # hanging up an ended call is harmless
    assert "Call ended" in status


async def test_visitors_are_hashed(demo: tuple[PhoneDemo, Ears]) -> None:
    phone, _ = demo
    a, b = phone.visitor(visitor(ip="203.0.113.7")), phone.visitor(visitor(ip="203.0.113.8"))
    assert a != b and "203" not in a and len(a) == 16
