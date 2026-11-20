"""What the web app needs, shared by every browser tab.

* The agent runtime (with its long-lived TalkTable stdio session) runs on its own event
  loop in a background thread; Gradio's handler threads hand it work and wait.
* One Copilot session per browser tab, tied to the logged-in staff member: a tab's
  session is never used by a different account.
* Speech to text and text to speech run in the handler's own thread (CPU work).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, TypeVar

from copilot.agent import CopilotRuntime
from copilot.audit import AuditLog
from copilot.auth import StaffUser, UserStore
from copilot.config import CopilotSettings
from copilot.session import CopilotSession
from copilot.speech.stt import Transcriber, make_transcriber
from copilot.speech.tts import Speaker, make_speaker
from copilot.speech.vad import Endpointer

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class Tab:
    """One browser tab's conversation, plus its hands-free microphone state."""

    user: StaffUser
    session: CopilotSession
    endpointer: Endpointer = field(default_factory=Endpointer)
    busy: bool = False
    deaf_until: float = 0.0  # don't listen while the Copilot is speaking (it would hear itself)
    heard: list[Any] = field(default_factory=list)  # finished hands-free utterances, waiting to be handled
    lock: threading.Lock = field(default_factory=threading.Lock)


class CopilotService:
    def __init__(self, settings: CopilotSettings, *, runtime: CopilotRuntime | None = None,
                 transcriber: Transcriber | None = None, speaker: Speaker | None = None,
                 users: UserStore | None = None, audit: AuditLog | None = None) -> None:
        self.settings = settings
        self.users = users or UserStore(settings.users_file)
        self.audit = audit or AuditLog(settings.audit_file)
        self.runtime = runtime or CopilotRuntime(settings)
        self.transcriber = transcriber or make_transcriber(settings.copilot_stt, settings.copilot_stt_model)
        self.speaker = speaker or make_speaker(settings.copilot_tts, settings.copilot_tts_voice)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="copilot-agent", daemon=True)
        self._tabs: dict[str, Tab] = {}
        self._lock = threading.Lock()

    # --- lifecycle ---------------------------------------------------------------------

    def start(self, *, warm_up: bool = True) -> None:
        self._thread.start()
        self.run(self.runtime.start())
        if warm_up:
            seconds = self.run(self.runtime.warm_up())
            logger.info("agent model %s ready in %.1f s", self.settings.agent_model, seconds)
            for part in (self.transcriber, self.speaker):
                if hasattr(part, "load"):
                    started = time.perf_counter()
                    part.load()
                    logger.info("%s ready in %.1f s", type(part).__name__, time.perf_counter() - started)

    def stop(self) -> None:
        try:
            self.run(self.runtime.close(), timeout=10)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def submit(self, work: Coroutine[Any, Any, T]) -> concurrent.futures.Future[T]:
        return asyncio.run_coroutine_threadsafe(work, self._loop)

    def run(self, work: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        return self.submit(work).result(timeout)

    # --- staff and tabs ----------------------------------------------------------------

    def authenticate(self, username: str, password: str) -> bool:
        """Gradio's login check."""
        ok = self.users.authenticate(username, password) is not None
        self.audit.write(user=username.strip().lower()[:32], role=None, tool="login", source="app",
                         outcome="ok" if ok else "denied")
        return ok

    def tab(self, tab_id: str, username: str | None) -> Tab:
        """This tab's conversation. The account must still exist and be enabled."""
        user = self.users.get((username or "").strip().lower())
        if user is None or user.disabled:
            raise PermissionError("This account is no longer active. Log in again.")
        with self._lock:
            tab = self._tabs.get(tab_id)
            if tab is not None and (tab.user.username, tab.user.role) == (user.username, user.role):
                return tab
        agent = self.run(self.runtime.agent_for(user))  # a new tab, or the account changed
        tab = Tab(user, CopilotSession(agent, self.audit))
        with self._lock:
            self._tabs[tab_id] = tab
        return tab

    def close_tab(self, tab_id: str) -> None:
        with self._lock:
            self._tabs.pop(tab_id, None)
