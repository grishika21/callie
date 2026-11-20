"""The Copilot web app's handlers (nothing is launched): the real service and agent with a
scripted model, the real Support Desk (HTTP + tokens, fake store), and stand-in speech."""

from __future__ import annotations

import time
import wave
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gradio as gr
import numpy as np
import pytest
from langchain_core.tools import tool

from copilot.agent import CopilotRuntime
from copilot.auth import TokenIssuer
from copilot.service import CopilotService
from copilot.speech.stt import Transcript
from copilot.speech.tts import Speech
from copilot.ui import ACKNOWLEDGE, Handlers, build_app, render_chat, render_ticket
from tests.agent_fakes import ScriptedChatModel, call, say
from tests.desk_fakes import SECRET, Desk, running_desk
from tests.test_copilot_agent import TICKET, ask, list_databases

SKIP = gr.skip()


class FakeTranscriber:
    def __init__(self) -> None:
        self.text, self.heard = "", []

    def transcribe(self, audio: tuple[int, np.ndarray], prompt: str | None = None) -> Transcript:
        self.heard.append(audio)
        return Transcript(self.text, 0.01, len(audio[1]) / audio[0])


class FakeSpeaker:
    def __init__(self) -> None:
        self.said: list[str] = []

    def speak(self, text: str) -> Speech:
        self.said.append(text)
        return Speech(16_000, np.zeros(1600, dtype=np.int16), 0.001)  # 0.1 s of audio


@tool
def slow_ask(question: str) -> str:
    """Answer a question about the bookstore's data (a slow stand-in)."""
    time.sleep(2.3)
    return "12 open tickets."


@pytest.fixture(scope="module")
def desk(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Desk]:
    with running_desk(tmp_path_factory.mktemp("ui-desk")) as server:
        yield server


@pytest.fixture
def model() -> ScriptedChatModel:
    return ScriptedChatModel()


@pytest.fixture
def service(desk: Desk, model: ScriptedChatModel, request: pytest.FixtureRequest) -> Iterator[CopilotService]:
    desk.reset()
    slow = getattr(request, "param", None) == "slow-talktable"
    talktable = [slow_ask.model_copy(update={"name": "ask"}), list_databases] if slow else [ask, list_databases]
    runtime = CopilotRuntime(desk.settings, model_factory=lambda: model, talktable_tools=talktable,
                             issuer=TokenIssuer(SECRET, desk.settings.desk_url))
    svc = CopilotService(desk.settings, runtime=runtime, transcriber=FakeTranscriber(), speaker=FakeSpeaker(),
                         users=desk.users, audit=desk.audit)
    svc.start(warm_up=False)
    yield svc
    svc.stop()


def req(user: str = "alice", tab: str = "tab-1") -> Any:
    return SimpleNamespace(username=user, session_hash=tab)


def last(outputs: Iterator[tuple[Any, ...]]) -> tuple[Any, ...]:
    *_, final = list(outputs)
    return final


# --- the app ------------------------------------------------------------------------------------


def test_nothing_is_public(service: CopilotService) -> None:
    """Unlike TalkTable's Space, the Copilot has no API: every event is private."""
    info = build_app(service).get_api_info()
    assert not info["named_endpoints"] and not info.get("unnamed_endpoints")


def test_login_is_checked_and_audited(service: CopilotService, desk: Desk) -> None:
    assert service.authenticate("alice", "correct horse battery")
    assert not service.authenticate("alice", "wrong password!") and not service.authenticate("mallory", "x")
    assert [e["outcome"] for e in desk.audit.read() if e["tool"] == "login"] == ["ok", "denied", "denied"]


def test_greeting_per_role(service: CopilotService) -> None:
    handlers = Handlers(service)
    assert "Signed in as <b>Alice</b> (agent). You can look things up and open tickets." in handlers.greet(req())[0]
    assert "opening tickets needs the agent role" in handlers.greet(req("victor", "tab-2"))[0]


def test_a_tab_belongs_to_its_account(service: CopilotService, desk: Desk) -> None:
    alice = service.tab("tab-1", "alice")
    assert service.tab("tab-1", "alice") is alice
    victor = service.tab("tab-1", "victor")  # same tab, another login: a fresh conversation
    assert victor is not alice and victor.user.username == "victor"
    with pytest.raises(PermissionError):
        service.tab("tab-1", "mallory")
    desk.users.remove("victor")
    try:
        with pytest.raises(PermissionError):
            service.tab("tab-1", "victor")  # removed accounts stop at once
    finally:
        desk.users.add("victor", "correct horse battery", "viewer")


# --- turns ----------------------------------------------------------------------------------------


def test_a_typed_question(service: CopilotService, model: ScriptedChatModel) -> None:
    model.script = [call("get_order", order_id=5512), say("Order 5512 is in transit with SwiftShip.")]
    steps = list(Handlers(service).text("Where is order 5512?", req()))
    first, final = steps[0], steps[-1]
    assert first[0][-1] == {"role": "user", "content": "Where is order 5512?"}  # shown at once
    chat, status, voice, card, _html, _prio, box, spoken = final
    assert chat[-1] == {"role": "assistant", "content": "Order 5512 is in transit with SwiftShip."}
    assert chat[1]["metadata"]["title"].startswith("🔧 get_order(order_id=5512) · ")
    assert "Ready" in status and voice[0] == 16_000 and card["visible"] is False and box == ""
    assert spoken.startswith("Order 5 5 1 2 is in transit with SwiftShip.␞")
    assert service.speaker.said == ["Order 5 5 1 2 is in transit with SwiftShip."]  # type: ignore[attr-defined]


def test_the_ticket_card(service: CopilotService, model: ScriptedChatModel, desk: Desk) -> None:
    handlers = Handlers(service)
    model.script = [call("get_order", order_id=5512), call("create_ticket", **TICKET)]
    _, _, _, card, card_html, priority, _, _ = last(handlers.text("Open a ticket for order 5512, it is damaged.", req()))
    assert card["visible"] is True and priority["value"] == "high"
    assert "Maria Gonzalez (customer 1)" in card_html and "Book arrived damaged" in card_html
    _, _, _, card, card_html, priority, _, _ = last(handlers.priority("urgent", req()))  # a draft: nothing written
    assert desk.store.writes == 0 and priority["value"] == "urgent" and "urgent-priority" in card_html
    chat, status, _, card, _, _, _, _ = last(handlers.approve(req()))
    assert card["visible"] is False and chat[-1]["content"] == "Opened ticket 1451 for Maria Gonzalez."
    assert desk.store.tickets[1451].priority == "urgent" and desk.store.tickets[1451].created_by == "alice"


def test_dont_open(service: CopilotService, model: ScriptedChatModel, desk: Desk) -> None:
    handlers = Handlers(service)
    model.script = [call("get_order", order_id=5512), call("create_ticket", **TICKET)]
    last(handlers.text("Open a ticket for order 5512, it is damaged.", req()))
    chat, _, _, card, _, _, _, _ = last(handlers.reject(req()))
    assert card["visible"] is False and chat[-1]["content"] == "Okay, I won't open that ticket."
    assert chat[-2]["metadata"]["title"].endswith("not opened (your decision)")
    assert desk.store.writes == 0
    assert list(handlers.approve(req())) == []  # nothing is waiting any more


def test_push_to_talk(service: CopilotService, model: ScriptedChatModel, tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(np.zeros((48_000, 2), dtype=np.int16).tobytes())
    service.transcriber.text = "where is order five five one two"  # type: ignore[attr-defined]
    model.script = [call("get_order", order_id=5512), say("It's in transit.")]
    outputs = list(Handlers(service).recording(str(path), req()))
    assert "Transcribing" in outputs[0][1]
    rate, audio = service.transcriber.heard[0]  # type: ignore[attr-defined]
    assert rate == 16_000 and len(audio) == 16_000  # decoded and resampled to 16 kHz mono
    assert model.seen[0][-1].content == "where is order 5512"  # the spoken id, as digits
    assert outputs[-1][0][-1]["content"] == "It's in transit." and outputs[-1][-1] is None  # recorder cleared


def test_nothing_heard(service: CopilotService, tmp_path: Path) -> None:
    path = tmp_path / "quiet.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(np.zeros(8000, dtype=np.int16).tobytes())
    outputs = list(Handlers(service).recording(str(path), req()))
    assert "I didn't hear anything" in outputs[-1][1]


def test_hands_free(service: CopilotService, model: ScriptedChatModel) -> None:
    handlers = Handlers(service)
    rng = np.random.default_rng(0)
    quiet = lambda: (48_000, (rng.normal(0, 30, 24_000)).astype(np.int16))  # noqa: E731
    loud = lambda: (48_000, (rng.normal(0, 6000, 24_000)).astype(np.int16))  # noqa: E731
    ticks = [handlers.stream(chunk, req()) for chunk in [quiet(), quiet(), loud(), loud(), quiet(), quiet(), quiet()]]
    done = [t for t in ticks if isinstance(t, str)]  # the rest are gr.skip()
    assert len(done) == 1  # one utterance, after the pause
    service.transcriber.text = "where is order 5512"  # type: ignore[attr-defined]
    model.script = [call("get_order", order_id=5512), say("It's in transit.")]
    final = last(handlers.utterance(done[0], req()))
    assert final[0][-1]["content"] == "It's in transit."
    tab = service.tab("tab-1", "alice")
    assert tab.deaf_until > time.monotonic()  # it doesn't listen while its own reply plays
    assert handlers.stream(loud(), req()) == SKIP


def test_one_turn_at_a_time(service: CopilotService) -> None:
    tab = service.tab("tab-1", "alice")
    tab.busy = True
    [(_, status, *_)] = list(Handlers(service).text("Where is order 5512?", req()))
    assert "Still working" in status


@pytest.mark.parametrize("service", ["slow-talktable"], indirect=True)
def test_slow_talktable_questions_are_acknowledged(service: CopilotService, model: ScriptedChatModel) -> None:
    model.script = [call("ask", question="How many tickets are open?"), say("12 are open.")]
    outputs = list(Handlers(service).text("How many tickets are open?", req()))
    statuses = [o[1] for o in outputs if isinstance(o[1], str)]
    assert any("Asking TalkTable" in s for s in statuses)
    assert ACKNOWLEDGE in service.speaker.said  # type: ignore[attr-defined]
    assert outputs[-1][0][-1]["content"] == "12 are open."


def test_rendering_escapes_text() -> None:
    from copilot.agent import PendingTicket, ToolStep, TurnResult
    from copilot.session import Entry

    turn = TurnResult(reply="x", steps=[ToolStep("get_ticket", {"ticket_id": 7}, True, {"description": "<b>hi</b>"}, 3.0)])
    chat = render_chat([Entry("staff", "<script>alert(1)</script>"), Entry("copilot", "a < b", turn=turn)])
    assert chat[0]["content"] == "&lt;script&gt;alert(1)&lt;/script&gt;" and chat[2]["content"] == "a &lt; b"
    assert chat[1]["content"].startswith("```json") and chat[1]["metadata"]["title"] == "🔧 get_ticket(ticket_id=7) · 3 ms"
    card = render_ticket(PendingTicket("i", {"subject": "<img src=x onerror=alert(1)>", "customer_id": 1}, "r"))
    assert "<img" not in card and "&lt;img" in card and "<table" not in card
