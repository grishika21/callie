"""Talk to Callie in a terminal (text in, text out; voice comes in V4).

    python -m callie.cli --user alice                     interactive
    python -m callie.cli --user alice --say "Where is order 5512?" --say "yes"
    python -m callie.cli --user alice --wav question.wav --wav yes.wav --speak-dir replies/

Needs the Support Desk running (python support_desk_server.py) and, for TalkTable
questions, a working TalkTable setup (it starts mcp_server.py itself). When a ticket is
proposed, answer as you would by voice: "yes", "no", "make it urgent", "it's a refund".
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import logging
import sys
import wave
from pathlib import Path

import numpy as np

from callie.agent import CallieRuntime, TurnResult
from callie.audit import AuditLog
from callie.auth import UserStore
from callie.config import get_callie_settings
from callie.session import CallieSession
from callie.speech.stt import make_transcriber
from callie.speech.text import speakable
from callie.speech.tts import make_speaker


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise SystemExit(f"{path}: only 16-bit PCM WAV files are supported")
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
        channels = handle.getnchannels()
        return handle.getframerate(), samples.reshape(-1, channels) if channels > 1 else samples


def write_wav(path: Path, rate: int, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.astype(np.int16).tobytes())

def show(result: TurnResult, *, trace: bool) -> None:
    if trace:
        for step in result.steps:
            state = "waiting" if step.ok is None else "ok" if step.ok else "error"
            took = f" {step.ms:.0f} ms" if step.ms is not None else ""
            print(f"    · {step.tool}({json.dumps(step.args, ensure_ascii=False)}) -> {state}{took}")
        print(f"    · {result.model_calls} model call(s), {result.seconds:.1f} s")
    print(f"callie> {result.reply}")


async def run(args: argparse.Namespace) -> None:
    settings = get_callie_settings()
    users = UserStore(settings.users_file)
    password = sys.stdin.readline().rstrip("\n") if args.password_stdin else getpass.getpass(f"{args.user}'s password: ")
    user = users.authenticate(args.user, password)
    if user is None:
        raise SystemExit("Wrong username or password, or the account is disabled.")
    async with CallieRuntime(settings) as runtime:
        agent = await runtime.agent_for(user)
        print(f"(loading {settings.agent_model}...)", file=sys.stderr)
        print(f"(model ready in {await runtime.warm_up():.1f} s)", file=sys.stderr)
        session = CallieSession(agent, AuditLog(settings.audit_file))
        print(f"Hello {user.display_name} ({user.role}). Tools: {', '.join(agent.tool_names)}. Ctrl-D to quit.")
        speaker = make_speaker(settings.callie_tts, settings.callie_tts_voice) if args.speak_dir else None
        if args.speak_dir:
            args.speak_dir.mkdir(parents=True, exist_ok=True)
        if args.wav:
            transcriber = make_transcriber(settings.callie_stt, settings.callie_stt_model)
            lines = []
            for path in args.wav:
                heard = transcriber.transcribe(read_wav(path))
                print(f"    · heard {path.name} in {heard.seconds:.1f} s: {heard.text!r}", file=sys.stderr)
                lines.append(heard.text)
            source = iter(lines)
        else:
            source = iter(args.say) if args.say else None
        turn = 0
        while True:
            if source is not None:
                line = next(source, None)
                if line is None:
                    break
                print(f"you> {line}")
            else:
                try:
                    line = input("you> ")
                except EOFError:
                    break
            if not line.strip():
                continue
            result = await session.hear(line)
            show(result, trace=not args.quiet)
            if speaker is not None and (speech := speaker.speak(speakable(result.reply))) is not None:
                turn += 1
                out = args.speak_dir / f"reply-{turn:02d}.wav"
                write_wav(out, speech.sample_rate, speech.samples)
                print(f"    · spoke {speech.audio_seconds:.1f} s of audio ({speech.seconds:.2f} s) -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password-stdin", action="store_true")
    parser.add_argument("--say", action="append", help="a message to send (repeatable); no prompt")
    parser.add_argument("--wav", action="append", type=Path, help="a 16-bit WAV to transcribe and send (repeatable)")
    parser.add_argument("--speak-dir", type=Path, help="write each spoken reply as a WAV here")
    parser.add_argument("--quiet", action="store_true", help="hide the tool trace")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING, stream=sys.stderr)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
