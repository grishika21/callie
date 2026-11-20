"""Callie on Streamlit Community Cloud: call the bookstore's AI phone assistant, push-to-talk.

Deploy at share.streamlit.io: Create app -> repository grishika21/callie, branch main, main
file path `streamlit_demo/streamlit_app.py`. Advanced settings: Python 3.11, and the secret
GROQ_API_KEY = "...". That's all: the app starts its own PostgreSQL with fresh demo data
(`postgresql` in packages.txt), the Support Desk, a catalog-only TalkTable, Groq Whisper (with
local base.en as the fallback) and Piper, then takes calls through callie/phone/push.py.

Streamlit can't stream audio both ways, so a call goes turn by turn: record, stop, hear the
reply. It is the same call as on the phone line (verification, agent, guards), only the
transport differs. The real phone line (Twilio) needs a host with its own web routes.

Locally (with PostgreSQL's programs installed, or DESK_POSTGRES_URL pointing at a desk database):
    streamlit run streamlit_demo/streamlit_app.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import subprocess
import sys
import threading
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # the code's data paths (data/...) are relative to the repository
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The demo's settings, as in space/Dockerfile (measured in docs/callie_plan.md, H2). The
# environment wins, so a host can change any of them without a code change.
DEMO_ENV = {
    "AGENT_PROVIDER": "groq",
    "AGENT_MODEL": "openai/gpt-oss-120b",
    "AGENT_REASONING_EFFORT": "low",
    "AGENT_FALLBACK_MODELS": "openai/gpt-oss-20b@low,qwen/qwen3.8-27b@none",
    "AGENT_MAX_RETRIES": "1",
    "CALLIE_TALKTABLE_LLM": "default",
    "LLM_PROVIDER": "groq",
    "LLM_MODEL": "qwen/qwen3.8-27b",
    "PHONE_STT": "groq",
    "PHONE_STT_MODEL": "base.en",
    "PHONE_MAX_CALLS": "4",
    "PHONE_MAX_MINUTES": "6",
    "PHONE_CALLS_PER_NUMBER_HOUR": "4",
    "CACHE_ENABLED": "true",
    "TOKENIZERS_PARALLELISM": "false",
}
for _key, _value in DEMO_ENV.items():
    os.environ.setdefault(_key, _value)

from callie.audit import AuditLog  # noqa: E402  (after the settings above)
from callie.config import CallieSettings, get_callie_settings  # noqa: E402
from callie.desk.launch import ensure_desk  # noqa: E402
from callie.phone.agent import PhoneRuntime  # noqa: E402
from callie.phone.call import PhoneDeps  # noqa: E402
from callie.phone.demo_db import start_demo_postgres  # noqa: E402
from callie.phone.push import PushToTalkCall, PushToTalkLine, Reply, decode_recording, wav_bytes  # noqa: E402
from callie.speech.stt import GroqTranscriber, Transcriber, WhisperTranscriber  # noqa: E402
from callie.speech.tts import VOICES_DIR, PiperSpeaker  # noqa: E402

logger = logging.getLogger("callie.streamlit")
T = TypeVar("T")

INTRO = """\
Talk to **Callie**, the bookstore's AI phone assistant. Press **Call**, then answer by voice: press the
microphone, talk, press stop, and Callie answers out loud. No microphone? Type instead, at the bottom.

**Demo customer:** order number **5512**, last name **Virk** (spell it if asked: V, I, R, K). Then try:
*"Where is my order?"* · *"The book arrived damaged, please open a ticket."* (then *yes*, or keypad **1**)
· *"Do you have any books by Eric Gross?"* · *"What's the status of order 7888?"* (someone else's) · *"Goodbye."*
"""

HOW_IT_WORKS = """\
- **The same call as the phone line.** Callie's phone line uses Twilio Media Streams. This page turns your
  recording into the same 8 kHz phone audio and hands it to the same call engine: Whisper hears it, a call
  script and an AI agent answer, and Piper speaks. Streamlit can't stream audio both ways, so here a call
  goes turn by turn.
- **Verification first, without any AI:** the order number and the last name must match the Support Desk's
  records. Three tries; it never says which part was wrong.
- **Only your own data:** after verification, every request to the Support Desk carries a token for that one
  customer, so someone else's order is "not found". Book questions go to a TalkTable that can only see the
  book catalog.
- **Tickets need your "yes"** (or keypad 1). They're written with channel `phone`, created by `phone-agent`.
- **Limits:** calls at once, calls per visitor per hour, call length. The data is fictional and resets when
  the app restarts. The AI runs on Groq's free tier, so it may be busy now and then.

Code, architecture and measurements: [github.com/grishika21/callie](https://github.com/grishika21/callie)
"""

KEYS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "0", "#"]
_MARKDOWN = re.compile(r"([\\`*_{}\[\]()#+!|<>~$])")


def plain(text: str) -> str:
    """Text shown as itself, not as Markdown (a transcript can contain anything)."""
    return _MARKDOWN.sub(r"\\\1", text)


# --- the backend: started once per app process, shared by every visitor ------------------------


@dataclass
class Backend:
    loop: asyncio.AbstractEventLoop
    line: PushToTalkLine
    settings: CallieSettings

    def run(self, coro: Coroutine[Any, Any, T], timeout: float = 180.0) -> T:
        """Run on the backend's event loop (it holds TalkTable's stdio session and the calls)."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)


def _secrets_to_environment() -> None:
    """Streamlit's root-level secrets are already environment variables on Community Cloud;
    this also covers a local .streamlit/secrets.toml."""
    try:
        values = {key: value for key, value in st.secrets.items() if isinstance(value, str | int | float)}
    except Exception:  # no secrets file: fine, the environment is used as it is
        return
    for key, value in values.items():
        os.environ.setdefault(key, str(value))


def _transcriber(settings: CallieSettings) -> Transcriber:
    local = WhisperTranscriber(settings.phone_stt_model)
    if settings.phone_stt == "groq" and settings.groq_api_key is not None:
        return GroqTranscriber(settings.groq_api_key.get_secret_value(), settings.groq_stt_model, fallback=local)
    return local


@st.cache_resource(show_spinner=False)
def backend() -> Backend:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "mcp", "faster_whisper", "sentence_transformers", "huggingface_hub",
                  "langchain_mcp_adapters"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _secrets_to_environment()
    if not os.environ.get("GROQ_API_KEY") and os.environ.get("AGENT_PROVIDER") == "groq":
        raise RuntimeError("GROQ_API_KEY isn't set. Add it under the app's Settings → Secrets.")
    if not os.environ.get("DESK_POSTGRES_URL"):
        os.environ["DESK_POSTGRES_URL"] = start_demo_postgres()
    os.environ.setdefault("CALLIE_JWT_SECRET", secrets.token_urlsafe(48))
    settings = get_callie_settings()
    voice = VOICES_DIR / f"{settings.callie_tts_voice}.onnx"
    if not voice.exists():
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "piper.download_voices", settings.callie_tts_voice,
                        "--download-dir", str(VOICES_DIR)], check=True, capture_output=True)

    ensure_desk(settings)
    transcriber, speaker = _transcriber(settings), PiperSpeaker(settings.callie_tts_voice)
    transcriber.load()  # type: ignore[attr-defined]
    speaker.load()

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, name="callie-loop", daemon=True).start()

    async def start() -> PushToTalkLine:
        runtime = PhoneRuntime(settings)
        await runtime.start()
        seconds = await runtime.warm_up()
        logger.info("agent %s/%s ready in %.1f s; speech: %s", settings.agent_provider, settings.agent_model,
                    seconds, type(transcriber).__name__)
        deps = PhoneDeps(runtime, transcriber, speaker, settings, AuditLog(settings.audit_file))
        return PushToTalkLine(deps, secret=settings.jwt_secret())

    line = asyncio.run_coroutine_threadsafe(start(), loop).result(600)
    return Backend(loop, line, settings)


# --- the page ---------------------------------------------------------------------------------------


def visitor_address() -> str:
    address = getattr(st.context, "ip_address", None)
    if not isinstance(address, str) or not address:  # None behind some proxies (and a mock in AppTest)
        forwarded = st.context.headers.get("X-Forwarded-For")
        address = forwarded.split(",")[0].strip() if isinstance(forwarded, str) else ""
    return address or st.session_state["tab"]


def remember(reply: Reply) -> None:
    state = st.session_state
    if reply.audio is not None:
        state.reply_audio = wav_bytes(reply.audio)
        state.reply_id += 1
    if not reply.heard_speech:
        state.notice = "I didn't hear anything. Press the microphone, talk, then press stop."


def show_transcript(call: PushToTalkCall) -> None:
    session = call.session
    if session.spoken:
        with st.chat_message("assistant", avatar="📞"):
            st.markdown(plain(session.spoken[0]))
    for turn in session.turns:
        with st.chat_message("user"):
            st.markdown(plain(turn.heard))
        with st.chat_message("assistant", avatar="📞"):
            for step in turn.steps:
                args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in step.args.items())
                took = f" · {step.ms:.0f} ms" if step.ms is not None else ""
                outcome = "" if step.ok else " · waiting for your yes" if step.ok is None else " · not done"
                with st.expander(f"🔧 {step.tool}({args}){took}{outcome}"):
                    st.json(step.result if isinstance(step.result, dict | list) else {"result": str(step.result)})
            st.markdown(plain(turn.reply))


def show_reply_audio() -> None:
    state = st.session_state
    if state.get("reply_audio"):
        fresh = state.reply_id != state.played_id
        st.audio(state.reply_audio, format="audio/wav", autoplay=fresh)
        state.played_id = state.reply_id


def main() -> None:
    st.set_page_config(page_title="Callie: call the bookstore", page_icon="📞", layout="centered")
    state = st.session_state
    state.setdefault("tab", secrets.token_hex(8))
    for key, value in (("mic", 0), ("reply_id", 0), ("played_id", 0), ("reply_audio", None), ("notice", "")):
        state.setdefault(key, value)

    st.title("📞 Callie")
    st.caption("An AI phone assistant for a bookstore, built on TalkTable")
    st.markdown(INTRO)
    try:
        with st.spinner("Starting Callie: the database, the Support Desk, TalkTable and the speech models. "
                        "This takes about a minute after the app wakes up…"):
            app = backend()
    except Exception as exc:  # shown to the visitor without a traceback
        logger.exception("startup failed")
        st.error(f"Callie couldn't start: {exc}")
        st.stop()
    app.line.sweep()
    call = app.line.get(state.tab)

    if call is None or call.ended:
        if call is not None:
            st.info(call.status)
        if st.button("📞 Call" if call is None else "📞 Call again", type="primary", width="stretch"):
            new, message = app.run(app.line.start(state.tab, visitor_address()))
            if new is None:
                st.warning(message)
            else:
                remember(Reply(new.io.take()))
                st.rerun()
        if call is not None:
            show_transcript(call)
            show_reply_audio()
        with st.expander("How it works"):
            st.markdown(HOW_IT_WORKS)
        return

    status = call.status
    if call.session.keypad_digits:
        status += f" · keypad: {call.session.keypad_digits} (press # when done)"
    st.success(status, icon="🟢")
    if state.notice:
        st.warning(state.notice)
        state.notice = ""

    talk, keypad = st.columns([3, 2], gap="medium")
    with talk:
        recording = st.audio_input("Your turn: press the microphone, talk, then press stop", key=f"mic-{state.mic}")
        if st.button("Hang up", icon="☎️"):
            app.run(app.line.hang_up(state.tab))
            state.reply_audio = None
            st.rerun()
    with keypad:
        st.caption("Keypad: order number then #, 1 = yes, 2 = no, 0 = a person")
        for row in range(4):
            for column, digit in zip(st.columns(3, gap="small"), KEYS[row * 3:row * 3 + 3], strict=True):
                if column.button(digit, key=f"key-{digit}", width="stretch"):
                    with st.spinner("…"):
                        remember(app.run(call.key(digit)))
                    st.rerun()
    typed = st.chat_input("No microphone? Type what you'd say")

    if recording is not None:
        with st.spinner("Callie is listening and thinking…"):
            remember(app.run(call.speak(decode_recording(recording.getvalue()))))
        state.mic += 1  # a fresh recorder for the next turn
        st.rerun()
    if typed:
        with st.spinner("Callie is thinking…"):
            remember(app.run(call.type(typed)))
        st.rerun()

    show_reply_audio()
    show_transcript(call)
    with st.expander("How it works"):
        st.markdown(HOW_IT_WORKS)


main()
