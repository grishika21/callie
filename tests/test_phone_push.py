"""Push-to-talk calls (callie/phone/push.py), as the Streamlit demo drives them: a whole
recording per turn, keypad and typed turns, limits. Fake speech, the scripted model, the
real desk. Also the Streamlit package's files (requirements, apt packages)."""

from __future__ import annotations

import io
import re
import wave
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import numpy as np
import pytest

from callie.auth import TokenIssuer
from callie.phone.agent import PhoneRuntime
from callie.phone.audio import PHONE_RATE
from callie.phone.call import GREETING, PhoneDeps
from callie.phone.demo_db import postgres_bin
from callie.phone.push import PushToTalkLine, decode_recording, push_to_talk_timing, to_phone_line, wav_bytes
from tests.agent_fakes import ScriptedChatModel, call, say
from tests.desk_fakes import SECRET, Desk, running_desk
from tests.test_callie_agent import ask
from tests.test_phone_call import Ears, Mouth

pytestmark = pytest.mark.anyio
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def desk(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Desk]:
    with running_desk(tmp_path_factory.mktemp("push-desk")) as server:
        yield server


@pytest.fixture
def model() -> ScriptedChatModel:
    return ScriptedChatModel()


@pytest.fixture
async def line(desk: Desk, model: ScriptedChatModel) -> AsyncIterator[tuple[PushToTalkLine, Ears]]:
    desk.reset()
    settings = desk.settings.model_copy(update={"phone_max_calls": 2, "phone_calls_per_number_hour": 2})
    runtime = PhoneRuntime(settings, users=desk.users, model_factory=lambda: model, talktable_tools=[ask],
                           issuer=TokenIssuer(SECRET, settings.desk_url))
    await runtime.start()
    ears = Ears()
    yield PushToTalkLine(PhoneDeps(runtime, ears, Mouth(), settings, desk.audit), secret="test"), ears
    await runtime.close()


def recording(seconds: float = 1.0, rate: int = 16_000, loud: bool = True) -> tuple[int, np.ndarray]:
    """What the browser records: voice-like noise (or silence) at its own rate."""
    n = int(seconds * rate)
    samples = np.random.default_rng(0).normal(0, 6000, n) if loud else np.zeros(n)
    return rate, samples.astype(np.int16)


async def verified(line: PushToTalkLine, ears: Ears, tab: str = "tab-1"):  # type: ignore[no-untyped-def]
    call_, _ = await line.start(tab, "203.0.113.7")
    assert call_ is not None
    ears.lines += ["My order number is five five one two.", "Gonzalez"]
    await call_.speak(recording())
    await call_.speak(recording())
    assert call_.session.state == "help"
    return call_


async def test_a_push_to_talk_call(line: tuple[PushToTalkLine, Ears], model: ScriptedChatModel) -> None:
    phone, ears = line
    call_, message = await phone.start("tab-1", "203.0.113.7")
    assert call_ is not None and message == "" and call_.session.spoken == [GREETING]
    greeting = call_.io.take()
    assert greeting is not None and greeting[0] == PHONE_RATE  # phone audio, to play at once
    ears.lines.append("My order number is five five one two.")
    reply = await call_.speak(recording(rate=48_000))
    assert reply.heard_speech and reply.audio is not None and call_.session.state == "name"
    assert call_.status == "Your last name, please"
    ears.lines.append("Gonzalez")
    await call_.speak(recording())
    assert call_.session.state == "help" and call_.status.startswith("Verified")
    model.script = [call("my_order", order_id=5512), say("Your order 5512 is on its way.")]
    ears.lines.append("Where is my order?")
    reply = await call_.speak(recording())
    turn = call_.session.turns[-1]
    assert (turn.heard, turn.reply) == ("Where is my order?", "Your order 5512 is on its way.")
    assert [s.tool for s in turn.steps] == ["my_order"] and reply.audio is not None
    assert ears.prompts[0] is not None and "order number" in ears.prompts[0]  # the step's own Whisper hint


async def test_a_pause_doesnt_cut_the_answer(line: tuple[PushToTalkLine, Ears]) -> None:
    """The whole recording is one answer, even with a pause in it (a phone call would split it)."""
    phone, ears = line
    call_, _ = await phone.start("tab-1", "203.0.113.7")
    rate, voice = recording()
    pause = np.zeros(rate, dtype=np.int16)  # 1 s, longer than the endpointer's 0.7 s
    ears.lines.append("My order number is 5512.")
    await call_.speak((rate, np.concatenate([voice, pause, voice])))
    assert len(ears.prompts) == 1 and call_.session.state == "name"


async def test_a_silent_recording(line: tuple[PushToTalkLine, Ears]) -> None:
    phone, ears = line
    call_, _ = await phone.start("tab-1", "203.0.113.7")
    call_.io.take()  # the greeting
    reply = await call_.speak(recording(loud=False))
    assert not reply.heard_speech and reply.audio is None and ears.prompts == []  # nothing went to Whisper


async def test_the_keypad_and_typing(line: tuple[PushToTalkLine, Ears], desk: Desk, model: ScriptedChatModel) -> None:
    phone, _ = line
    call_, _ = await phone.start("tab-1", "203.0.113.7")
    call_.io.take()  # the greeting
    for digit in "5512":
        assert (await call_.key(digit)).audio is None
    assert call_.session.keypad_digits == "5512"
    reply = await call_.key("#")
    assert reply.audio is not None and call_.session.state == "name"
    await call_.type("Gonzalez")
    assert call_.session.state == "help"
    model.script = [call("open_my_ticket", order_id=5512, category="damaged item", subject="Book arrived damaged",
                         description="The book in my order arrived damaged.")]
    await call_.type("The book arrived damaged, please open a ticket.")
    assert "Shall I go ahead?" in call_.session.spoken[-1] and desk.store.writes == 0
    await call_.key("1")  # yes
    ticket = desk.store.tickets[1451]
    assert (ticket.channel, ticket.created_by) == ("phone", "phone-agent")


async def test_goodbye_ends_the_call(line: tuple[PushToTalkLine, Ears]) -> None:
    phone, ears = line
    call_ = await verified(phone, ears)
    ears.lines.append("That's all, thanks. Goodbye!")
    reply = await call_.speak(recording())
    assert call_.ended and reply.audio is not None and call_.status == "The call ended. Thanks for trying it!"
    assert (await call_.speak(recording())).audio is None  # an ended call hears nothing
    again, _ = await phone.start("tab-1", "203.0.113.7")
    assert again is not None and again is not call_  # "Call again" starts a new call


async def test_limits(line: tuple[PushToTalkLine, Ears]) -> None:
    phone, _ = line
    await phone.start("a", "198.51.100.1")
    await phone.start("b", "198.51.100.2")
    busy, message = await phone.start("c", "198.51.100.3")
    assert busy is None and "All lines are busy" in message  # 2 at once in this test
    await phone.hang_up("a")
    await phone.hang_up("b")
    for tab in ("d", "e"):
        await phone.start(tab, "198.51.100.9")
        await phone.hang_up(tab)
    again, message = await phone.start("f", "198.51.100.9")
    assert again is None and "several calls in the last hour" in message  # 2 per visitor per hour here
    assert phone.visitor("198.51.100.9") != phone.visitor("198.51.100.8") and "198" not in phone.visitor("198.51.100.9")


def test_push_to_talk_timing_never_nudges() -> None:
    timing = push_to_talk_timing(360)
    assert timing.max_seconds == 360 and timing.silence_nudge > 10**6 and timing.filler_after > 10**6


def test_recordings_in_and_out() -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:  # stereo 16-bit at 48 kHz
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(np.full((4800, 2), 1000, dtype=np.int16).tobytes())
    rate, samples = decode_recording(buffer.getvalue())
    assert rate == 48_000 and samples.shape == (4800, 2)
    assert len(to_phone_line((rate, samples))) == 800  # 0.1 s of 8 kHz μ-law
    with wave.open(io.BytesIO(wav_bytes((8000, np.zeros(80, dtype=np.int16))))) as handle:
        assert (handle.getframerate(), handle.getnchannels(), handle.getnframes()) == (8000, 1, 80)


# --- the Streamlit package -----------------------------------------------------------------------


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        match = re.match(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==([^\s;#]+)", line.strip())
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def test_streamlit_requirements_match_the_project() -> None:
    demo = _pins(ROOT / "streamlit_demo" / "requirements.txt")
    project = _pins(ROOT / "requirements.txt") | _pins(ROOT / "requirements-callie.txt")
    drift = {name: (version, project.get(name)) for name, version in demo.items()
             if name not in ("streamlit", "torch") and project.get(name) != version}
    assert drift == {}
    assert "gradio" not in demo and "pytest" not in demo
    missing = set(project) - set(demo) - {"gradio", "pytest"}
    assert missing == set()


def test_streamlit_package_files() -> None:
    assert (ROOT / "packages.txt").read_text().split() == ["postgresql"]  # the demo's own database
    assert "streamlit_demo/streamlit_app.py" in (ROOT / "streamlit_demo" / "streamlit_app.py").read_text()
    assert ".streamlit/secrets.toml" in (ROOT / ".gitignore").read_text()


@pytest.mark.skipif(postgres_bin() is None, reason="PostgreSQL's programs aren't installed here")
def test_the_demo_database(tmp_path: Path) -> None:  # pragma: no cover - runs where postgres is installed
    import psycopg

    from callie.phone.demo_db import start_demo_postgres, stop_demo_postgres

    url = start_demo_postgres(tmp_path / "pg", port=55432)
    try:
        with psycopg.connect(url) as conn:  # the desk login: fresh demo data, and it can't delete
            assert conn.execute("SELECT count(*) FROM orders WHERE order_id = 5512").fetchone() == (1,)
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute("DELETE FROM support_tickets")
        start_demo_postgres(tmp_path / "pg", port=55432)  # a restart replaces the old server and data
    finally:
        stop_demo_postgres(tmp_path / "pg")


async def test_warm_up_builds_the_catalog_index(desk: Desk, model: ScriptedChatModel) -> None:
    """TalkTable loads its embedding model on first use (~35 s on 2 CPUs): the warm-up does it."""
    from langchain_core.tools import tool

    listed: list[bool] = []

    @tool
    def list_databases() -> str:
        """List the databases."""
        listed.append(True)
        return "sqlite"

    runtime = PhoneRuntime(desk.settings, users=desk.users, model_factory=lambda: model, talktable_tools=[ask],
                           issuer=TokenIssuer(SECRET, desk.settings.desk_url))
    await runtime.start()
    runtime._catalog_lister = list_databases  # what start() keeps from the real TalkTable process
    model.script = [say("OK")]
    await runtime.warm_up()
    assert listed == [True]
    await runtime.close()
