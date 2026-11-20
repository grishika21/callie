"""Start the Support Desk inside another process (the staff web app, the phone line, the
Hugging Face Space) unless one is already running at DESK_URL."""

from __future__ import annotations

import logging
import threading
import time

import httpx
import uvicorn

from copilot.config import CopilotSettings
from copilot.desk.server import create_server

logger = logging.getLogger(__name__)


def desk_running(url: str) -> bool:
    try:
        return httpx.post(url, timeout=2).status_code in (401, 403)  # up, and asking for a token
    except httpx.HTTPError:
        return False


def start_desk_in_process(settings: CopilotSettings) -> None:
    app = create_server(settings).streamable_http_app()
    server = uvicorn.Server(uvicorn.Config(app, host=settings.desk_host, port=settings.desk_port, log_level="warning"))
    threading.Thread(target=server.run, name="support-desk", daemon=True).start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline:
            raise SystemExit("The Support Desk did not start")
        time.sleep(0.1)
    logger.info("Support Desk started in this process at %s", settings.desk_url)


def ensure_desk(settings: CopilotSettings) -> None:
    if not desk_running(settings.desk_url):
        start_desk_in_process(settings)
