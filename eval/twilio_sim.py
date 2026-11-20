"""Play Twilio against a running phone line, over its public URL (plan P3).

    python -m eval.twilio_sim --url https://<name>.trycloudflare.com --token <TWILIO_AUTH_TOKEN of the server>

Does what Twilio does on an incoming call: a signed POST to /twilio/voice, then the
signed Media Streams WebSocket from the TwiML, with the caller's voice (Piper, 8 kHz
μ-law, 20 ms frames, in real time); it "plays" the replies (sending marks back when they
would have finished), transcribes them so you can read what a caller heard, then a
signed POST to /twilio/after. It also checks that unsigned requests are refused (403).
No Twilio account needed: the server just has to use the same auth token.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import secrets
import time
import xml.etree.ElementTree as ET

import httpx
import numpy as np
import websockets
from twilio.request_validator import RequestValidator

from copilot.phone.audio import FRAME_BYTES, PHONE_RATE, SILENCE_BYTE, from_phone, resample, ulaw_encode
from copilot.speech.stt import WhisperTranscriber
from copilot.speech.tts import PiperSpeaker

LINES = ["My order number is five five one two.", "My last name is Virk.", "Where is my order?",
         "That's all, thank you. Goodbye!"]


class Caller:
    def __init__(self, ws: websockets.ClientConnection, stream_sid: str) -> None:
        self.ws, self.stream_sid = ws, stream_sid
        self.audio = bytearray()  # the reply being received
        self.replies: list[tuple[float, bytes]] = []  # (seconds after the caller stopped talking, audio)
        self.spoke_at = time.monotonic()
        self.closed = asyncio.Event()

    async def listen(self) -> None:
        first = 0.0
        try:
            async for raw in self.ws:
                message = json.loads(raw)
                if message["event"] == "media":
                    if not self.audio:
                        first = time.monotonic() - self.spoke_at
                    self.audio += base64.b64decode(message["media"]["payload"])
                elif message["event"] == "mark":
                    clip, self.audio = bytes(self.audio), bytearray()
                    self.replies.append((first, clip))
                    await asyncio.sleep(len(clip) / PHONE_RATE)  # "play" it, then report the mark
                    await self.ws.send(json.dumps({"event": "mark", "streamSid": self.stream_sid,
                                                   "mark": {"name": message["mark"]["name"]}}))
                elif message["event"] == "clear":
                    self.audio = bytearray()
        except websockets.ConnectionClosed:
            pass
        self.closed.set()

    async def _send(self, audio: bytes) -> None:
        for i in range(0, len(audio), FRAME_BYTES):
            payload = base64.b64encode(audio[i:i + FRAME_BYTES]).decode()
            await self.ws.send(json.dumps({"event": "media", "streamSid": self.stream_sid,
                                           "media": {"track": "inbound", "payload": payload}}))
            await asyncio.sleep(0.02)  # real time, like a phone

    async def say(self, pcm: np.ndarray) -> None:
        await self._send(ulaw_encode(pcm))
        self.spoke_at = time.monotonic()  # the caller stopped talking
        await self._send(bytes([SILENCE_BYTE]) * int(1.2 * PHONE_RATE))

    async def quiet(self, timeout: float = 60) -> None:
        """Wait until the gateway has answered and gone quiet."""
        deadline, seen = time.monotonic() + timeout, len(self.replies)
        while time.monotonic() < deadline and not self.closed.is_set():
            await asyncio.sleep(0.3)
            if len(self.replies) > seen and not self.audio:
                await asyncio.sleep(1.5)
                if not self.audio:
                    return


async def run(url: str, token: str, account: str) -> None:
    validator, url = RequestValidator(token), url.rstrip("/")
    call = {"CallSid": "CA" + secrets.token_hex(16), "AccountSid": account, "From": "+15550100042",
            "To": "+15550109999", "CallStatus": "ringing", "Direction": "inbound"}
    async with httpx.AsyncClient(timeout=20) as http:
        unsigned = await http.post(f"{url}/twilio/voice", data=call)
        print(f"unsigned POST /twilio/voice -> {unsigned.status_code}")
        response = await http.post(f"{url}/twilio/voice", data=call,
                                   headers={"X-Twilio-Signature": validator.compute_signature(f"{url}/twilio/voice", call)})
        print(f"signed POST /twilio/voice -> {response.status_code}\n  {response.text}")
        root = ET.fromstring(response.text)
        stream = root.find("./Connect/Stream")
        assert stream is not None, "no <Stream> in the TwiML"
        params = {p.get("name"): p.get("value") for p in stream.findall("Parameter")}
        after = root.find("./Redirect").text  # type: ignore[union-attr]
        try:
            async with websockets.connect(stream.get("url")) as ws:
                await ws.recv()
            print("unsigned media stream -> accepted (WRONG)")
        except websockets.InvalidStatus as exc:
            print(f"unsigned media stream -> refused, HTTP {exc.response.status_code}")

        speaker, ears = PiperSpeaker(), WhisperTranscriber("small.en")
        ears.load()
        sid = "MZ" + secrets.token_hex(16)
        headers = {"X-Twilio-Signature": validator.compute_signature(stream.get("url"), {})}
        async with websockets.connect(stream.get("url"), additional_headers=headers, max_size=None) as ws:
            caller = Caller(ws, sid)
            listener = asyncio.create_task(caller.listen())
            await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
            await ws.send(json.dumps({"event": "start", "sequenceNumber": "1", "streamSid": sid, "start": {
                "streamSid": sid, "accountSid": account, "callSid": call["CallSid"], "tracks": ["inbound"],
                "customParameters": params, "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000,
                                                            "channels": 1}}}))
            await caller.quiet()
            for line in LINES:
                if caller.closed.is_set():
                    break
                heard_before = len(caller.replies)
                speech = speaker.speak(line)
                await caller.say(resample(speech.samples, speech.sample_rate, PHONE_RATE))
                await caller.quiet()
                spoken = " ".join(ears.transcribe(from_phone(clip)).text for _, clip in caller.replies[heard_before:])
                wait = caller.replies[heard_before][0] if len(caller.replies) > heard_before else float("nan")
                print(f"\ncaller: {line}\nagent ({wait:.1f} s after the caller stopped): {spoken}")
                if re.search(r"\bspell\b", spoken, re.IGNORECASE) and "last name" in line.lower():
                    letters = ", ".join(ch.upper() for ch in line.split()[-1] if ch.isalpha())
                    speech = speaker.speak(letters + ".")
                    heard_before = len(caller.replies)
                    await caller.say(resample(speech.samples, speech.sample_rate, PHONE_RATE))
                    await caller.quiet()
                    spoken = " ".join(ears.transcribe(from_phone(c)).text for _, c in caller.replies[heard_before:])
                    print(f"caller: {letters}\nagent: {spoken}")
            if not caller.closed.is_set():
                await ws.send(json.dumps({"event": "stop", "streamSid": sid, "stop": {"callSid": call["CallSid"]}}))
            await asyncio.wait_for(caller.closed.wait(), 10)
            listener.cancel()
        print("\nmedia stream closed by the gateway" if caller.closed.is_set() else "")
        form = {**call, "CallStatus": "in-progress"}
        final = await http.post(after, data=form, headers={"X-Twilio-Signature": validator.compute_signature(after, form)})
        print(f"signed POST /twilio/after -> {final.status_code}: {final.text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True, help="the auth token the phone server checks signatures with")
    parser.add_argument("--account", default="ACsimulated", help="AccountSid to send (match TWILIO_ACCOUNT_SID)")
    args = parser.parse_args()
    asyncio.run(run(args.url, args.token, args.account))


if __name__ == "__main__":
    main()
