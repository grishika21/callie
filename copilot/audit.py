"""Append-only JSONL audit log (data/audit/copilot.jsonl, git-ignored).

One line per tool call or human decision: when, who, which tool, the arguments (emails
masked, long text shortened), what happened, how long it took and the ticket it touched.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
_MAX_TEXT = 200


def mask_email(text: str) -> str:
    """maria.gonzalez@example.com -> m***@example.com"""
    return EMAIL_RE.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", text)


def mask_value(value: Any) -> Any:
    if isinstance(value, str):
        masked = mask_email(value)
        return masked if len(masked) <= _MAX_TEXT else masked[:_MAX_TEXT] + "…"
    if isinstance(value, dict):
        return {str(k): mask_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [mask_value(v) for v in value]
    return value


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, *, user: str | None, role: str | None, tool: str, outcome: str,
              args: dict[str, Any] | None = None, latency_ms: float | None = None,
              ticket_id: int | None = None, decision: str | None = None, detail: str | None = None,
              source: str = "desk", context: dict[str, Any] | None = None) -> None:
        event = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "source": source,
                 "user": user, "role": role, "context": context or None, "tool": tool, "args": mask_value(args or {}),
                 "decision": decision, "outcome": outcome, "detail": mask_value(detail),
                 "latency_ms": None if latency_ms is None else round(latency_ms, 1), "ticket_id": ticket_id}
        line = json.dumps({k: v for k, v in event.items() if v is not None}, ensure_ascii=False, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
