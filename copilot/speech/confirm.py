"""What the staff member's answer to "Shall I open it?" means. Deterministic: no LLM.

    "yes" / "go ahead" / "sounds good"        -> approve
    "no" / "cancel" / "never mind"            -> reject
    "make it urgent" / "change it to refund"  -> edit (the ticket is read back again first)
    "actually, where is order 7888?"          -> new request (the ticket is not opened)
    "hmm" / "what?"                           -> unclear: ask once more
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

Kind = Literal["approve", "reject", "edit", "new_request", "unclear"]

_YES = re.compile(r"^(?:yes|yeah|yep|yup|yea|sure|correct|right|ok(?:ay)?|alright|all right|confirm(?:ed)?|approve[d]?|"
                  r"go ahead|do it|please do|sounds good|that's (?:right|correct|good|fine)|looks good|perfect|"
                  r"open it|send it|submit it|absolutely|of course)\b")
_NO = re.compile(r"^(?:no|nope|nah|cancel|don't|do not|stop|never ?mind|scratch that|forget it|wrong|not now|"
                 r"reject|leave it)\b")
_PRIORITY = {"low": "low", "normal": "normal", "medium": "normal", "regular": "normal", "high": "high",
             "urgent": "urgent", "critical": "urgent", "top": "urgent", "highest": "urgent"}
_CATEGORY = {"delivery": "delivery", "shipping": "delivery", "payment": "payment", "billing": "payment",
             "damaged": "damaged item", "damaged item": "damaged item", "damage": "damaged item",
             "refund": "refund", "account": "account", "login": "account", "other": "other"}
_PRIORITY_EDIT = re.compile(r"\b(?:make it|mark it(?: as)?|set it to|change it to|set (?:the )?priority to|"
                            r"change (?:the )?priority to|priority (?:to|should be|is)|it's|it is|should be)\s+"
                            r"(?:a\s+)?(low|normal|medium|regular|high|urgent|critical|top|highest)\b"
                            r"|\b(low|normal|medium|high|urgent|critical|top|highest)[ -]priority\b")
_CATEGORY_EDIT = re.compile(r"\b(?:category (?:to|should be|is)|change it to|make it|it's|it is|should be|"
                            r"set it to|change (?:the )?category to)\s+(?:a\s+|an\s+)?"
                            r"(delivery|shipping|payment|billing|damaged item|damaged|damage|refund|account|login|"
                            r"other)\b(?:\s+(?:ticket|issue|problem|one))?")
_TEXT_EDIT = re.compile(r"\b(?:change|set|make) the (subject|description) to\s+(.+)$")
_ORDER_EDIT = re.compile(r"\b(?:it's|it is|change it to|make it|the order is|should be|for)\s+order\s+(\d{2,7})\b")
_REQUEST = re.compile(r"\?\s*$|^(?:actually|instead|also|and|what|where|when|who|how|which|why|is|are|does|do|can|"
                      r"could|find|look|show|check|tell|open|create|list|get)\b")


@dataclass(frozen=True)
class Confirmation:
    kind: Kind
    changes: dict[str, Any] = field(default_factory=dict)


def _words(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w\s'?-]", " ", text)
    return " ".join(text.split())


def parse_confirmation(text: str) -> Confirmation:
    said = _words(text)
    if not said:
        return Confirmation("unclear")
    changes: dict[str, Any] = {}
    if match := _PRIORITY_EDIT.search(said):
        changes["priority"] = _PRIORITY[match.group(1) or match.group(2)]
    if match := _CATEGORY_EDIT.search(said):
        changes["category"] = _CATEGORY[match.group(1)]
    if match := _ORDER_EDIT.search(said):
        changes["order_id"] = int(match.group(1))
    if match := _TEXT_EDIT.search(text.strip().lower()):
        start = text.lower().index(match.group(2))
        changes[match.group(1)] = text.strip()[start:].strip(" .")
    if changes and not _NO.match(said):
        return Confirmation("edit", changes)
    rest = len(said.split())
    if _NO.match(said):
        tail = _NO.sub("", said, count=1).strip(" ,")
        # "no, where is order 7888?" moves on; "no thanks" is just no
        return Confirmation("new_request" if len(tail.split()) > 3 and _REQUEST.search(tail) else "reject")
    if _YES.match(said):
        tail = _YES.sub("", said, count=1).strip(" ,")
        if len(tail.split()) <= 3 and "?" not in tail:  # "yes please", "yes, do it", "sure thing"
            return Confirmation("approve")
        return Confirmation("unclear")  # "yes, and where is order 7888?": settle the ticket first
    if rest >= 4 and _REQUEST.search(said):
        return Confirmation("new_request")
    return Confirmation("unclear")
