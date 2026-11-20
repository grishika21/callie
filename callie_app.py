"""Callie: the voice web app for support staff.

    .venv/bin/python callie_app.py              http://127.0.0.1:7861 (staff login)

Starts the agent (and TalkTable over stdio), loads whisper and Piper, and, unless one is
already running at DESK_URL, the Support Desk MCP server in this process. Staff
accounts: python -m callie.auth add-user <name> --role agent|viewer.

It listens on 127.0.0.1 only (CALLIE_HOST / CALLIE_PORT). The microphone works on
http://127.0.0.1 or http://localhost, which browsers treat as secure. Unlike app.py
(TalkTable's public Space), it has no MCP server and no public API.
"""

from __future__ import annotations

import argparse
import logging
import sys

from callie.config import get_callie_settings
from callie.desk.launch import desk_running, start_desk_in_process
from callie.service import CallieService
from callie.ui import CALLIE_CSS, build_app
from talktable.webui import CSS, FORCE_DARK_JS, build_theme

logger = logging.getLogger("callie.app")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-desk", action="store_true", help="don't start the Support Desk here")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "mcp", "faster_whisper", "sentence_transformers", "huggingface_hub", "langchain_mcp_adapters"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = get_callie_settings()
    users = settings.users_file
    if not users.exists():
        raise SystemExit(f"No staff accounts yet ({users}). Add one: python -m callie.auth add-user <name> --role agent")
    if not desk_running(settings.desk_url):
        if args.no_desk:
            raise SystemExit(f"The Support Desk isn't running at {settings.desk_url}: python support_desk_server.py")
        start_desk_in_process(settings)

    service = CallieService(settings)
    logger.info("starting the agent (%s), TalkTable, whisper %s and %s speech...", settings.agent_model,
                settings.callie_stt_model, settings.callie_tts)
    service.start()
    try:
        build_app(service).queue(default_concurrency_limit=8).launch(
            server_name=settings.callie_host,
            server_port=settings.callie_port,
            auth=service.authenticate,
            auth_message="Callie is for support staff. Log in with your Callie account.",
            theme=build_theme(),
            css=CSS + CALLIE_CSS,
            js=FORCE_DARK_JS,
            ssr_mode=False,
            mcp_server=False,
            footer_links=["gradio"],
        )
    finally:
        service.stop()


if __name__ == "__main__":
    main()
