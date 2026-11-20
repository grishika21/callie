"""TalkTable Copilot's phone line: customers call a Twilio number and the Copilot answers.

    .venv/bin/python phone_server.py           gateway on http://127.0.0.1:8770
    scripts/phone_tunnel.sh                    the same, behind a Cloudflare quick tunnel,
                                               with the Twilio number pointed at it

Needs TWILIO_AUTH_TOKEN (every request from Twilio is signature-checked) and, for the
setup script, TWILIO_ACCOUNT_SID and TWILIO_PHONE_NUMBER. Starts the Support Desk in this
process unless one is running, a catalog-only TalkTable (stdio), whisper and Piper.
Details: docs/copilot_plan.md (Phone line).
"""

from __future__ import annotations

import logging
import sys

import uvicorn

from copilot.audit import AuditLog
from copilot.config import get_copilot_settings
from copilot.desk.launch import ensure_desk
from copilot.phone.agent import PhoneRuntime
from copilot.phone.call import PhoneDeps
from copilot.phone.gateway import PhoneGateway, runtime_lifespan
from copilot.speech.stt import WhisperTranscriber
from copilot.speech.tts import make_speaker

logger = logging.getLogger("copilot.phone")


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "mcp", "faster_whisper", "sentence_transformers", "huggingface_hub",
                  "langchain_mcp_adapters", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = get_copilot_settings()
    if settings.twilio_auth_token is None:
        raise SystemExit("Set TWILIO_AUTH_TOKEN in .env (Twilio console > Account > API keys & tokens).")
    ensure_desk(settings)

    transcriber = WhisperTranscriber(settings.phone_stt_model)
    speaker = make_speaker("piper", settings.copilot_tts_voice)
    transcriber.load()
    speaker.load()  # type: ignore[attr-defined]
    runtime = PhoneRuntime(settings)
    audit = AuditLog(settings.audit_file)
    deps = PhoneDeps(runtime, transcriber, speaker, settings, audit,
                     can_transfer=bool(settings.phone_transfer_number))
    gateway = PhoneGateway(settings, deps)

    async def start() -> None:
        await runtime.start()
        logger.info("agent model ready in %.1f s", await runtime.warm_up())
        logger.info("phone line ready on http://%s:%s (transfer: %s, public URL: %s)", settings.phone_host,
                    settings.phone_port, "yes" if settings.phone_transfer_number else "no, callback tickets",
                    settings.phone_public_url or "from the tunnel's requests")

    uvicorn.run(gateway.app(runtime_lifespan(start, runtime.close)), host=settings.phone_host,
                port=settings.phone_port, log_level="warning", ws_ping_interval=None)


if __name__ == "__main__":
    main()
