"""The public demo (a Hugging Face Docker Space): one app on one port.

    /            "Call the bookstore": the browser is the phone (callie/phone/web.py)
    /twilio/*    the Twilio gateway, when TWILIO_AUTH_TOKEN is set (real phone calls)

Inside the same container (space/start.sh starts them first): Postgres with the demo data
and the insert-only desk login; then this process starts the Support Desk on 127.0.0.1,
a catalog-only TalkTable (stdio), Whisper (Groq's, with a local fallback) and Piper. The
caller agent runs on Groq (AGENT_PROVIDER=groq), with fallback models for rate limits.

    python -m callie.phone.space          (the Space's command; locally: http://127.0.0.1:7860)
"""

from __future__ import annotations

import logging
import os
import sys

import gradio as gr
import uvicorn
from fastapi import FastAPI
from starlette.responses import PlainTextResponse

from callie.audit import AuditLog
from callie.config import CallieSettings, get_callie_settings
from callie.desk.launch import ensure_desk
from callie.phone.agent import PhoneRuntime
from callie.phone.call import PhoneDeps
from callie.phone.gateway import PhoneGateway, runtime_lifespan
from callie.phone.web import PhoneDemo, build_demo
from callie.speech.stt import GroqTranscriber, Transcriber, WhisperTranscriber
from callie.speech.tts import PiperSpeaker
from callie.ui import CALLIE_CSS
from talktable.webui import CSS, FORCE_DARK_JS, build_theme

logger = logging.getLogger("callie.space")


def space_settings() -> CallieSettings:
    settings = get_callie_settings()
    host = os.environ.get("SPACE_HOST")  # set by Hugging Face: <user>-<space>.hf.space
    if host and not settings.phone_public_url:
        settings = settings.model_copy(update={"phone_public_url": f"https://{host}"})
    return settings


def transcriber_for(settings: CallieSettings) -> Transcriber:
    local = WhisperTranscriber(settings.phone_stt_model)
    if settings.phone_stt == "groq" and settings.groq_api_key is not None:
        return GroqTranscriber(settings.groq_api_key.get_secret_value(), settings.groq_stt_model, fallback=local)
    return local


def create_app(settings: CallieSettings | None = None) -> FastAPI:
    settings = settings or space_settings()
    ensure_desk(settings)
    transcriber, speaker = transcriber_for(settings), PiperSpeaker(settings.callie_tts_voice)
    transcriber.load()  # type: ignore[attr-defined]
    speaker.load()
    runtime = PhoneRuntime(settings)
    audit = AuditLog(settings.audit_file)
    deps = PhoneDeps(runtime, transcriber, speaker, settings, audit, can_transfer=bool(settings.phone_transfer_number))
    gateway = PhoneGateway(settings, deps) if settings.twilio_auth_token is not None else None
    demo = PhoneDemo(deps, other_calls=(lambda: len(gateway.calls)) if gateway else (lambda: 0),
                     secret=settings.jwt_secret())

    async def start() -> None:
        await runtime.start()
        logger.info("agent %s/%s ready in %.1f s; speech: %s; phone line: %s", settings.agent_provider,
                    settings.agent_model, await runtime.warm_up(), type(transcriber).__name__,
                    f"on, {settings.phone_public_url}/twilio/voice" if gateway else "off (no TWILIO_AUTH_TOKEN)")

    app = FastAPI(lifespan=runtime_lifespan(start, runtime.close), docs_url=None, redoc_url=None, openapi_url=None)
    if gateway is not None:
        app.router.routes.extend(gateway.routes())  # before Gradio's catch-all mount at "/"
    app.add_route("/health", lambda request: PlainTextResponse("ok"))
    blocks = build_demo(demo).queue(default_concurrency_limit=8)
    return gr.mount_gradio_app(app, blocks, path="/", theme=build_theme(), css=CSS + CALLIE_CSS, js=FORCE_DARK_JS,
                               ssr_mode=False, mcp_server=False, footer_links=["gradio"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "mcp", "faster_whisper", "sentence_transformers", "huggingface_hub",
                  "langchain_mcp_adapters", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    uvicorn.run(create_app(), host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "7860")),
                ws_ping_interval=None)


if __name__ == "__main__":
    main()
