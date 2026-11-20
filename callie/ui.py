"""Callie's voice web app (Gradio), for logged-in support staff.

    transcript (with each tool call, collapsed)       status line ("Asking TalkTable... 12 s")
    ticket card when a ticket waits for an OK: say yes / no / "make it urgent", or click
    push-to-talk mic, or hands-free (it listens continuously), or type
    the reply, spoken (Piper) and shown in full

Only staff can log in (Gradio's login, checked against data/callie_users.yaml), and
the app is launched without an MCP server and with every event private: unlike
TalkTable's public Space, nothing here is an API.
"""

from __future__ import annotations

import html
import json
import logging
import time
from collections.abc import Callable, Coroutine, Iterator
from typing import Any

import gradio as gr
import numpy as np

from callie.agent import PendingTicket, ToolStep, TurnResult
from callie.service import CallieService, Tab
from callie.session import Entry
from callie.speech.stt import load_audio
from callie.speech.text import speakable
from callie.speech.tts import Speech

logger = logging.getLogger(__name__)

PRIORITIES = ["low", "normal", "high", "urgent"]
ACKNOWLEDGE = "Let me look that up. It can take a little while."
_TOOL_STATUS = {"get_order": "Checking the order", "find_customer": "Looking up the customer",
                "list_tickets": "Reading the tickets", "get_ticket": "Reading the ticket",
                "create_ticket": "Opening the ticket", "ask": "Asking TalkTable (this can take a while)",
                "list_databases": "Checking what TalkTable can see"}
_PRIVATE: dict[str, Any] = {"api_visibility": "private"}

CALLIE_CSS = """
.cp-user { color: #8b949e; font-size: .85rem; margin: 0 0 .4rem; }
.cp-user b { color: #e6edf3; font-weight: 600; }
.cp-status { color: #8b949e; font-size: .9rem; padding: .35rem .1rem; }
.cp-status b { color: #00D4AA; font-weight: 600; }
.cp-status.cp-busy b::before { content: "◌ "; }
.cp-status.cp-ready b::before { content: "● "; }
.cp-status.cp-error b { color: #ff7b72; }
.cp-card { border: 1px solid #00d4aa66; background: #00d4aa0d; border-radius: 10px; padding: .7rem .9rem; }
.cp-card h3 { margin: 0 0 .35rem; font-size: .95rem; color: #00D4AA; }
.cp-card p { margin: 0 0 .5rem; color: #e6edf3; }
.cp-fields { display: grid; grid-template-columns: 7.5rem 1fr; font-size: .85rem; }
.cp-fields span { padding: 3px 6px; border-top: 1px solid #30363d; color: #c9d1d9; overflow-wrap: anywhere; }
.cp-fields span:nth-child(odd) { color: #8b949e; }
.cp-hint { color: #8b949e; font-size: .8rem; margin-top: .45rem; }
#cp-chat .placeholder { max-width: 620px; margin: 0 auto; text-align: center; color: #8b949e; }
#cp-chat .placeholder h3 { color: #e6edf3; }
"""

# Speaks a reply in the browser when CALLIE_TTS=browser (Piper is off then).
BROWSER_SPEAK_JS = """(text) => {
  if (!text || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(text.split("\\u241e")[0]));
}"""


# --- rendering ------------------------------------------------------------------------------------


def _e(value: object) -> str:
    return html.escape(str(value), quote=False)


def render_status(state: str, detail: str = "") -> str:
    kind = {"ready": "cp-ready", "error": "cp-error"}.get(state, "cp-busy")
    label = {"ready": "Ready", "listening": "Listening", "transcribing": "Transcribing", "error": "Problem"}.get(
        state, state)
    return f'<div class="cp-status {kind}"><b>{_e(label)}</b> {_e(detail)}</div>'


def busy_status(running: tuple[str, float] | None, elapsed: float) -> str:
    if running is None:
        return render_status("Thinking", f"{elapsed:.0f} s")
    tool, seconds = running
    return render_status(_TOOL_STATUS.get(tool, f"Running {tool}"), f"{seconds:.0f} s")


def _step_message(step: ToolStep) -> dict[str, Any]:
    args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in step.args.items())
    declined = step.ok is False and "rejected" in str(step.result).lower()
    state = ("waiting for your OK" if step.ok is None else "" if step.ok
             else "not opened (your decision)" if declined else "failed")
    took = f" · {step.ms:.0f} ms" if step.ms is not None else ""
    body = json.dumps(step.result, ensure_ascii=False, indent=1, default=str) if step.result is not None else ""
    if len(body) > 1500:
        body = body[:1500] + "\n…"
    return {"role": "assistant", "content": f"```json\n{body}\n```" if body else "(no result yet)",
            "metadata": {"title": f"🔧 {step.tool}({args}){took} {state}".strip(), "status": "done"}}


def render_chat(transcript: list[Entry]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for entry in transcript:
        if entry.speaker == "staff":
            messages.append({"role": "user", "content": _e(entry.text)})
            continue
        for step in entry.turn.steps if entry.turn else []:
            messages.append(_step_message(step))
        messages.append({"role": "assistant", "content": _e(entry.text)})
    return messages


def render_ticket(pending: PendingTicket | None) -> str:
    if pending is None:
        return ""
    a = pending.args
    rows = [("Customer", a.get("customer_id")), ("Order", a.get("order_id") or "none"), ("Category", a.get("category")),
            ("Priority", a.get("priority", "normal")), ("Subject", a.get("subject")),
            ("Description", a.get("description"))]
    cells = "".join(f"<span>{_e(k)}</span><span>{_e(v)}</span>" for k, v in rows)
    return (f'<div class="cp-card"><h3>A ticket is waiting for your OK</h3><p>{_e(pending.readback)}</p>'
            f'<div class="cp-fields">{cells}</div><div class="cp-hint">Say “yes”, “no”, or a change such as '
            "“make it urgent”; or use the buttons.</div></div>")


def card_updates(pending: PendingTicket | None) -> tuple[Any, Any, Any]:
    """(card group, card html, priority dropdown)."""
    if pending is None:
        return gr.update(visible=False), "", gr.update(value=None)
    return gr.update(visible=True), render_ticket(pending), gr.update(value=pending.args.get("priority", "normal"))


def speech_value(speech: Speech | None) -> tuple[int, np.ndarray] | None:
    return None if speech is None else (speech.sample_rate, speech.samples)


def render_user(tab: Tab) -> str:
    user = tab.user
    rights = ("You can look things up and open tickets." if "tickets:write" in user.scopes
              else "You can look things up; opening tickets needs the agent role.")
    return (f'<p class="cp-user">Signed in as <b>{_e(user.display_name)}</b> ({_e(user.role)}). {rights} '
            '<a href="/logout">Log out</a></p>')


# --- handlers -------------------------------------------------------------------------------------

# A turn's outputs: chat, status, voice, card, card html, priority, text box, spoken text (browser TTS)
Outputs = tuple[Any, Any, Any, Any, Any, Any, Any, Any]


def _skip() -> Any:
    return gr.skip()


class Handlers:
    """The app's event handlers, bound to one service. Each turn is a generator: it shows
    the request at once, then the status while the agent works, then the reply."""

    def __init__(self, service: CallieService) -> None:
        self.service = service

    def _tab(self, request: gr.Request) -> Tab:
        return self.service.tab(request.session_hash or "", request.username)

    def greet(self, request: gr.Request) -> tuple[str, list[dict[str, Any]]]:
        tab = self._tab(request)
        return render_user(tab), render_chat(tab.session.transcript)

    def close(self, request: gr.Request) -> None:
        self.service.close_tab(request.session_hash or "")

    def _turn(self, tab: Tab, shown: str | None, work: Callable[[], Coroutine[Any, Any, TurnResult]]) -> Iterator[Outputs]:
        if tab.busy:
            yield (_skip(), render_status("Still working", "on your last request"), _skip(), _skip(), _skip(), _skip(),
                   _skip(), _skip())
            return
        tab.busy = True
        try:
            chat = render_chat(tab.session.transcript)
            if shown:
                chat += [{"role": "user", "content": _e(shown)}]
            yield chat, render_status("Thinking"), _skip(), _skip(), _skip(), _skip(), "", _skip()
            future = self.service.submit(work())
            started, acknowledged = time.monotonic(), False
            while True:
                try:
                    result = future.result(timeout=0.4)
                    break
                except TimeoutError:
                    pass
                running = tab.session.agent.running_tool(tab.session.thread_id)
                voice = _skip()
                if running and running[0] == "ask" and running[1] > 1.5 and not acknowledged:
                    acknowledged = True  # TalkTable is slow: say so, so silence doesn't feel broken
                    voice = speech_value(self.service.speaker.speak(ACKNOWLEDGE)) or _skip()
                yield (_skip(), busy_status(running, time.monotonic() - started), voice, _skip(), _skip(), _skip(),
                       _skip(), _skip())
            speech = self.service.speaker.speak(speakable(result.reply))
            tab.deaf_until = time.monotonic() + (speech.audio_seconds if speech else 0.0) + 0.6
            tab.endpointer.reset()
            yield (render_chat(tab.session.transcript), render_status("ready", f"{result.seconds:.1f} s"),
                   speech_value(speech), *card_updates(result.pending), "", f"{speakable(result.reply)}␞{time.time()}")
        except PermissionError as exc:
            yield (_skip(), render_status("error", str(exc)), _skip(), _skip(), _skip(), _skip(), _skip(), _skip())
        except Exception:
            logger.exception("turn failed")
            yield (render_chat(tab.session.transcript), render_status("error", "Something went wrong. Try again."),
                   _skip(), _skip(), _skip(), _skip(), _skip(), _skip())
        finally:
            tab.busy = False

    # Each entry point below is a generator over the same outputs.

    def text(self, message: str, request: gr.Request) -> Iterator[Outputs]:
        message = (message or "").strip()
        if not message:
            return
        tab = self._tab(request)
        yield from self._turn(tab, message, lambda: tab.session.hear(message))

    def _heard(self, tab: Tab, audio: tuple[int, np.ndarray]) -> Iterator[Outputs]:
        yield (_skip(), render_status("transcribing"), _skip(), _skip(), _skip(), _skip(), _skip(), _skip())
        heard = self.service.transcriber.transcribe(audio)
        logger.info("heard %.1f s of audio in %.1f s: %r", heard.audio_seconds, heard.seconds, heard.text)
        if not heard.text.strip():
            yield (_skip(), render_status("ready", "I didn't hear anything. Try again?"), _skip(), _skip(), _skip(),
                   _skip(), _skip(), _skip())
            return
        yield from self._turn(tab, heard.text, lambda: tab.session.hear(heard.text))

    def recording(self, path: str | None, request: gr.Request) -> Iterator[tuple[Any, ...]]:
        """Push to talk: the clip arrives when the staff member stops recording."""
        if not path:
            return
        tab = self._tab(request)
        for outputs in self._heard(tab, load_audio(path)):
            yield (*outputs, None)  # and clear the recorder for the next clip

    def stream(self, chunk: tuple[int, np.ndarray] | None, request: gr.Request) -> Any:
        """Hands-free: every half second of microphone audio. Cheap: it only finds where an
        utterance ends, then pokes `utterance` (a hidden box) to handle it."""
        if chunk is None:
            return gr.skip()
        tab = self._tab(request)
        if tab.busy or time.monotonic() < tab.deaf_until:
            tab.endpointer.reset()  # don't hear Callie's own voice, or speech during a turn
            return gr.skip()
        utterance = tab.endpointer.feed(chunk)
        if utterance is None:
            return gr.skip()
        tab.heard.append(utterance)
        return str(time.time())

    def utterance(self, _tick: str, request: gr.Request) -> Iterator[Outputs]:
        tab = self._tab(request)
        while tab.heard:
            yield from self._heard(tab, (16_000, tab.heard.pop(0)))

    def approve(self, request: gr.Request) -> Iterator[Outputs]:
        tab = self._tab(request)
        if tab.session.pending is not None:
            yield from self._turn(tab, "(clicked Open ticket)", tab.session.confirm)

    def reject(self, request: gr.Request) -> Iterator[Outputs]:
        tab = self._tab(request)
        if tab.session.pending is not None:
            yield from self._turn(tab, "(clicked Don't open)", tab.session.reject)

    def priority(self, value: str, request: gr.Request) -> Iterator[Outputs]:
        tab = self._tab(request)
        if tab.session.pending is not None and value:
            yield from self._turn(tab, f"(set priority to {value})", lambda: tab.session.change(priority=value))

    def hands_free(self, on: bool) -> tuple[Any, Any, str]:
        hint = "Listening: just speak. Pause when you're done." if on else "Press the mic, speak, then stop."
        return gr.update(visible=not on), gr.update(visible=on), render_status("ready", hint)


# --- the app ----------------------------------------------------------------------------------------


def build_app(service: CallieService) -> gr.Blocks:
    """The Blocks app. Theme, CSS and the login are passed to `launch()` (see callie_app.py)."""
    h = Handlers(service)
    with gr.Blocks(title="Callie", fill_height=True) as app:
        gr.HTML('<div class="tt-brand"><div class="tt-logo">C</div><h1>Callie</h1></div>'
                '<p class="tt-pitch">Your voice assistant for orders, customers and support tickets.</p>')
        user_line = gr.HTML()
        chat = gr.Chatbot(elem_id="cp-chat", show_label=False, height=440, latex_delimiters=[], buttons=["copy"],
                          placeholder="### Ask about an order, a customer or a ticket\n"
                                      "“Where is order 5512?” · “Open a ticket for order 5512, the book arrived "
                                      "damaged.” · “How many tickets were opened by voice today?”")
        status = gr.HTML(render_status("ready", "Press the mic, speak, then stop. Or type below."))
        with gr.Group(visible=False) as card:
            card_html = gr.HTML()
            with gr.Row():
                approve = gr.Button("Open ticket", variant="primary")
                reject = gr.Button("Don't open", variant="secondary")
                priority = gr.Dropdown(PRIORITIES, label="Priority", value=None, interactive=True, scale=1)
        with gr.Row(equal_height=True):
            mic = gr.Audio(sources=["microphone"], type="filepath", label="Push to talk", scale=3)
            live = gr.Audio(sources=["microphone"], type="numpy", streaming=True, label="Hands-free", visible=False,
                            scale=3)
            voice = gr.Audio(label="Callie", autoplay=True, interactive=False, scale=2)
        hands_free = gr.Checkbox(label="Hands-free (listens continuously; it won't listen while it speaks)")
        with gr.Row(equal_height=True):
            text = gr.Textbox(placeholder="Or type: Where is order 5512?", show_label=False, scale=8, container=False)
            send = gr.Button("Send", variant="primary", scale=1, min_width=90)
        utterance = gr.Textbox(visible=False)  # hands-free: a finished utterance is waiting
        spoken = gr.Textbox(visible=False)  # the text to speak, for CALLIE_TTS=browser

        outputs = [chat, status, voice, card, card_html, priority, text, spoken]
        turn: dict[str, Any] = {"outputs": outputs, "concurrency_limit": 8, "show_progress": "hidden", **_PRIVATE}
        app.load(h.greet, None, [user_line, chat], **_PRIVATE)
        app.unload(h.close)
        text.submit(h.text, [text], **turn)
        send.click(h.text, [text], **turn)
        mic.stop_recording(h.recording, [mic], **{**turn, "outputs": [*outputs, mic]})
        live.stream(h.stream, [live], [utterance], stream_every=0.5, time_limit=None, concurrency_limit=None,
                    show_progress="hidden", **_PRIVATE)
        utterance.change(h.utterance, [utterance], **turn)
        approve.click(h.approve, None, **turn)
        reject.click(h.reject, None, **turn)
        priority.input(h.priority, [priority], **turn)
        hands_free.change(h.hands_free, [hands_free], [mic, live, status], **_PRIVATE)
        if service.settings.callie_tts == "browser":
            spoken.change(None, [spoken], None, js=BROWSER_SPEAK_JS, **_PRIVATE)
    return app
