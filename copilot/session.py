"""One conversation (one browser tab, or one CLI run): who is talking, the agent thread,
a ticket waiting for a decision, the transcript and the tool trace.

`hear(text)` takes everything the staff member says, typed or transcribed:
* spoken ids are normalized first ("order five five one two" -> "order 5512");
* while a ticket waits, the answer is parsed without an LLM (copilot/speech/confirm.py):
  yes opens it; no doesn't; "make it urgent" changes the draft and reads it back again
  (nothing is written until a yes); a new request moves on, leaving the ticket unopened;
  anything unclear is asked once more, and a second unclear answer leaves it unopened.
Every decision is written to the audit log.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal

from copilot.agent import CopilotAgent, Decision, InvalidEditError, PendingTicket, TurnResult
from copilot.audit import AuditLog
from copilot.speech.confirm import parse_confirmation
from copilot.speech.text import normalize_transcript

ASK_AGAIN = "Sorry, I didn't catch that. Should I open the ticket? Say yes, no, or what to change."
GAVE_UP = "I didn't open the ticket. Ask me again when you're ready."


@dataclass
class Entry:
    speaker: Literal["staff", "copilot"]
    text: str
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    turn: TurnResult | None = None


class CopilotSession:
    def __init__(self, agent: CopilotAgent, audit: AuditLog | None = None) -> None:
        self.agent, self.audit = agent, audit
        self.thread_id = uuid.uuid4().hex
        self.transcript: list[Entry] = []
        self.pending: PendingTicket | None = None
        self.draft: dict[str, Any] = {}  # spoken changes to the pending ticket, not sent yet
        self._unclear = 0

    @property
    def user(self) -> Any:
        return self.agent.user

    async def hear(self, text: str) -> TurnResult:
        """Anything the staff member says or types (see the module docstring)."""
        text = normalize_transcript(text)
        if self.pending is None:
            return await self.say(text)
        answer = parse_confirmation(text)
        if answer.kind == "approve":
            self.transcript.append(Entry("staff", text))
            return await self.confirm()
        if answer.kind == "reject":
            self.transcript.append(Entry("staff", text))
            return await self.reject()
        if answer.kind == "edit":
            self.transcript.append(Entry("staff", text))
            return await self._change_draft(answer.changes)
        if answer.kind == "new_request":
            return await self.say(text)
        self.transcript.append(Entry("staff", text))
        self._unclear += 1
        if self._unclear >= 2:
            result = await self._decide("reject", "unclear", reason="The staff member's answer was unclear twice. "
                                                                    "Don't open it.")
            result.reply = GAVE_UP
            self.transcript[-1].text = GAVE_UP
            return result
        return self._note(TurnResult(reply=ASK_AGAIN, pending=self.pending))

    async def say(self, text: str) -> TurnResult:
        """A new request from the staff member (a pending ticket is left unopened first)."""
        if self.pending is not None:
            await self._decide("reject", "moved_on", reason="The staff member moved on without approving. Don't open it.")
        self.transcript.append(Entry("staff", text))
        return self._record(await self.agent.send(self.thread_id, text))

    async def confirm(self) -> TurnResult:
        """Open the pending ticket, with any changes said since it was proposed."""
        return await (self.edit(**self.draft) if self.draft else self.approve())

    async def change(self, **changes: Any) -> TurnResult:
        """Change the pending ticket's draft (read back; nothing is written yet)."""
        return await self._change_draft(changes)

    async def approve(self) -> TurnResult:
        return await self._decide("approve", "approved")

    async def reject(self) -> TurnResult:
        return await self._decide("reject", "rejected")

    async def edit(self, **changes: Any) -> TurnResult:
        return await self._decide(changes, "edited")

    async def _change_draft(self, changes: dict[str, Any]) -> TurnResult:
        assert self.pending is not None
        try:
            self.agent.check_edit(changes)
        except InvalidEditError as exc:
            return self._note(TurnResult(reply=f"I can't change that: {exc} Should I open it as it is?",
                                         pending=self.pending))
        self.draft |= changes
        args = {**self.pending.args, **self.draft}
        self.pending = replace(self.pending, args=args, readback=await self.agent.readback(args))
        self._unclear = 0
        return self._note(TurnResult(reply=f"{self.pending.readback} Shall I open it?", pending=self.pending))

    async def _decide(self, decision: Decision, label: str, *, reason: str | None = None) -> TurnResult:
        pending = self.pending
        if pending is None:
            raise RuntimeError("no ticket is waiting for a decision")
        result = await self.agent.decide(self.thread_id, decision, reason=reason)
        if self.audit is not None:
            args = {**pending.args, **decision} if isinstance(decision, dict) else pending.args
            self.audit.write(user=self.user.username, role=self.user.role, tool=self.agent.write_tool, source="agent",
                             args=args, decision=label, outcome="error" if result.error else "ok",
                             ticket_id=result.ticket_id, latency_ms=result.seconds * 1000)
        return self._record(result)

    def _note(self, result: TurnResult) -> TurnResult:
        """A reply the session gives itself (no agent turn)."""
        self.transcript.append(Entry("copilot", result.reply, turn=result))
        return result

    def _record(self, result: TurnResult) -> TurnResult:
        if result.pending is None or self.pending is None or result.pending.interrupt_id != self.pending.interrupt_id:
            self.draft, self._unclear = {}, 0
        self.pending = result.pending
        self.transcript.append(Entry("copilot", result.reply, turn=result))
        if self.audit is not None:  # TalkTable calls (the desk audits its own)
            for step in result.steps:
                if step.tool in ("ask", "list_databases"):
                    self.audit.write(user=self.user.username, role=self.user.role, tool=step.tool, source="agent",
                                     args=step.args, outcome="ok" if step.ok else "error", latency_ms=step.ms)
        return result
