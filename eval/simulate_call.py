"""Simulated phone calls through the real call engine (plan P2), no Twilio needed.

    python -m eval.simulate_call                          every call in eval/phone_eval.yaml
    python -m eval.simulate_call --calls ticket-yes --show

Piper plays the caller; each line goes through a phone line (8 kHz μ-law) into
`CallSession` frame by frame, the way Twilio's Media Streams deliver it, followed by a
pause. Behind it everything is real: whisper small.en, the 4B agent (Ollama), the
catalog-only TalkTable (stdio, DATABASES=sqlite), and a Support Desk on Postgres (in
this process). Replies "play" at 4x speed. Tickets the calls write (created_by
'phone-agent') are deleted after each call. Needs: Postgres + the desk role, Ollama.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import re
import secrets
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from callie.audit import AuditLog
from callie.phone.agent import PhoneRuntime
from callie.phone.audio import FRAME_BYTES, SILENCE_BYTE, PHONE_RATE, ulaw_encode, resample
from callie.phone.call import CallSession, PhoneDeps
from callie.phone.verify import last_name_of
from callie.speech.stt import WhisperTranscriber
from callie.speech.tts import PiperSpeaker
from data.seed import generate as gen
from eval.benchmark_callie import Harness
from eval.benchmark_phone_speech import spoken_digits
from talktable.config import get_settings

HERE = Path(__file__).resolve().parent
CALLS = HERE / "phone_eval.yaml"
VERIFY = ["My order number is five five one two.", "My last name is Virk."]
PLAYBACK_SPEED = 4.0


class SimIO:
    """Twilio's side: 'plays' our audio (faster than real time) and reports marks."""

    def __init__(self) -> None:
        self.session: CallSession | None = None
        self.hung_up = self.transferred = self.cleared = 0
        self.audio_seconds = 0.0

    async def play(self, mulaw: bytes, mark: str) -> None:
        seconds = len(mulaw) / PHONE_RATE
        self.audio_seconds += seconds
        asyncio.get_running_loop().call_later(seconds / PLAYBACK_SPEED,
                                              lambda: asyncio.ensure_future(self.session.on_mark(mark)))  # type: ignore[union-attr]

    async def clear(self) -> None:
        self.cleared += 1

    async def hang_up(self) -> None:
        self.hung_up += 1

    async def transfer(self) -> None:
        self.transferred += 1


def fixtures() -> dict[str, str]:
    """A customer with a hard last name (and one of their orders), and a catalog author."""
    sales, catalog = gen.generate_sales(), gen.generate_catalog()
    hard = next(c for c in sales.customers if last_name_of(c.name).replace(" ", "") in ("adolfsson", "sachdeva",
                                                                                         "gierschner", "choudhary"))
    order = next(o for o in sales.orders if o.customer_id == hard.customer_id)
    last = hard.name.split(maxsplit=1)[1]
    with_books = sorted({b.author_id for b in catalog.books})
    author = next(a.name for a in catalog.authors if a.author_id == random.Random(3).choice(with_books))
    return {"hard_order_words": spoken_digits(order.order_id), "hard_name": last,
            "hard_name_letters": ", ".join(ch.upper() for ch in last if ch.isalpha()), "author": author}


@dataclass
class CallResult:
    id: str
    passed: bool
    detail: str
    verified: bool
    end: str | None
    turns: list[Any] = field(default_factory=list)


async def caller_says(session: CallSession, speaker: PiperSpeaker, line: Any) -> None:
    if isinstance(line, dict):  # keypad
        for digit in line["keypad"]:
            await session.on_dtmf(digit)
            await asyncio.sleep(0.05)
    else:
        speech = await asyncio.to_thread(speaker.speak, line)
        assert speech is not None
        pcm = await asyncio.to_thread(resample, speech.samples, speech.sample_rate, PHONE_RATE)
        audio = ulaw_encode(pcm) + bytes([SILENCE_BYTE]) * int(1.2 * PHONE_RATE)
        for i in range(0, len(audio), FRAME_BYTES):
            await session.on_audio(audio[i:i + FRAME_BYTES])
            if i % (FRAME_BYTES * 25) == 0:
                await asyncio.sleep(0)  # let the call's tasks run, as a live stream would
    deadline = time.monotonic() + 300
    await asyncio.sleep(0.1)
    while not session.idle and session.state != "ended" and time.monotonic() < deadline:
        await asyncio.sleep(0.05)


def judge(expect: dict[str, Any], session: CallSession, io: SimIO, tickets: int) -> tuple[bool, str]:
    used = {s.tool for t in session.turns for s in t.steps}
    replies = " ".join(t.reply for t in session.turns).lower().replace("\u2019", "'")  # curly apostrophes
    problems = []
    if "verified" in expect and (session.customer_id is not None) != expect["verified"]:
        problems.append(f"verified={session.customer_id is not None}")
    if expect.get("tools_any") and not used & set(expect["tools_any"]):
        problems.append(f"tools {sorted(used)}")
    if set(expect.get("forbid", [])) & used:
        problems.append(f"forbidden tools {sorted(set(expect['forbid']) & used)}")
    if expect.get("reply_has") and not any(w.lower() in replies for w in expect["reply_has"]):
        problems.append("reply lacks " + "/".join(expect["reply_has"]))
    if any(w.lower() in replies for w in expect.get("reply_not", [])):
        problems.append("reply has a forbidden word")
    if "tickets" in expect and tickets != expect["tickets"]:
        problems.append(f"tickets={tickets}")
    if "ended" in expect and session.end_reason != expect["ended"]:
        problems.append(f"ended={session.end_reason}")
    return not problems, "; ".join(problems)


async def run(selected: list[dict[str, Any]], show: bool) -> list[CallResult]:
    admin = get_settings().postgres_seed_url
    if admin is None:
        raise SystemExit("POSTGRES_SEED_URL is not set (needed to clean up the calls' tickets)")
    values = fixtures()
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        harness = Harness(admin.get_secret_value(), Path(tmp))
        first_new = harness.sql("SELECT coalesce(max(ticket_id), 0) FROM support_tickets")[0][0]
        runtime = PhoneRuntime(harness.settings, users=harness.users)
        transcriber, speaker = WhisperTranscriber("small.en"), PiperSpeaker()
        transcriber.load()
        await runtime.start()
        await runtime.warm_up()
        audit = AuditLog(Path(tmp) / "phone-audit.jsonl")
        try:
            for case in selected:
                io = SimIO()
                deps = PhoneDeps(runtime, transcriber, speaker, harness.settings, audit,
                                 can_transfer=bool(case.get("transfer")))
                session = CallSession(f"SIM{secrets.token_hex(4)}", f"sim-{case['id']}", "…00", io, deps)
                io.session = session
                await session.start()
                if case.get("verify"):  # a real caller answers what they're asked, including "spell it"
                    for line in VERIFY:
                        await caller_says(session, speaker, line)
                    if session.state == "spell":
                        await caller_says(session, speaker, "V, I, R, K.")
                lines = [line.format(**values) if isinstance(line, str) else line for line in case["lines"]]
                for n, line in enumerate(lines):
                    if session.state == "ended":
                        break
                    await caller_says(session, speaker, line)
                    upcoming = lines[n + 1] if n + 1 < len(lines) else ""
                    if isinstance(line, str) and session.state == "spell" and "spelled" not in str(upcoming) \
                            and "," not in str(upcoming):  # asked to spell, as a real caller would be
                        name = re.sub(r"[^A-Za-z ]", "", line).split()[-1]
                        await caller_says(session, speaker, ", ".join(ch.upper() for ch in name) + ".")
                if session.state != "ended":
                    await session.end("simulation over")
                tickets = harness.sql("SELECT count(*) FROM support_tickets WHERE created_by = 'phone-agent' "
                                      "AND ticket_id > %s", (first_new,))[0][0]
                harness.sql("DELETE FROM support_tickets WHERE created_by = 'phone-agent' AND ticket_id > %s",
                            (first_new,))
                passed, detail = judge(case.get("expect", {}), session, io, tickets)
                results.append(CallResult(case["id"], passed, detail, session.customer_id is not None,
                                          session.end_reason, session.turns))
                print(f"  {'ok  ' if passed else 'MISS'} {case['id']:<22} " + " | ".join(
                    f"{t.seconds:4.1f}s {','.join(s.tool for s in t.steps) or t.state}" for t in session.turns)
                    + (f"   <- {detail}" if detail else ""), flush=True)
                if show:
                    for t in session.turns:
                        print(f"        caller: {t.heard!r}\n        agent:  {t.reply!r}")
        finally:
            await runtime.close()
            harness.stop()
    return results


def summarize(results: list[CallResult]) -> None:
    turns = [t for r in results for t in r.turns]
    verify = sorted(t.seconds for t in turns if t.state in ("order", "name", "spell", "help") and not t.steps
                    and t.reply.startswith(("Thanks", "I couldn't", "I still", "Sorry")))
    helped = sorted(t.seconds for t in turns if t.steps or t.state == "help")
    print(f"\n{sum(r.passed for r in results)}/{len(results)} calls passed")
    answers = sorted(t.reply_seconds for t in turns if t.reply_seconds is not None)
    for label, values in (("verification turns", verify), ("agent turns", helped),
                          ("agent turns, to the actual answer", answers)):
        if values:
            p90 = values[min(len(values) - 1, int(0.9 * len(values)))]
            print(f"  {label}: median {statistics.median(values):.1f} s, p90 {p90:.1f} s, max {values[-1]:.1f} s "
                  f"(end of speech -> first audio, whisper included)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calls", help="comma-separated call ids (default: all)")
    parser.add_argument("--show", action="store_true", help="print what was heard and said")
    args = parser.parse_args()
    cases = yaml.safe_load(CALLS.read_text(encoding="utf-8"))["calls"]
    if args.calls:
        wanted = set(args.calls.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    summarize(asyncio.run(run(cases, args.show)))


if __name__ == "__main__":
    main()
