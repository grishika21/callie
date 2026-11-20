"""The Twilio side of the phone line: three webhooks and a Media Streams WebSocket.

    POST /twilio/voice    a call comes in: check limits, answer with TwiML that streams it here
    WS   /twilio/stream   the call's audio both ways (μ-law 8 kHz), driving a CallSession
    POST /twilio/after    the stream ended: <Dial> the transfer number, or <Hangup/>
    POST /twilio/status   Twilio's call status callback (cleanup and audit)

Every request must carry a valid X-Twilio-Signature (HMAC with the account's auth token),
the WebSocket handshake included; anything else gets 403. Transfers and hang-ups need no
Twilio REST credentials: when the stream closes, Twilio asks /twilio/after what to do.
Caller numbers are never stored: only an HMAC of the number and its last two digits.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import logging
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect
from twilio.request_validator import RequestValidator
from twilio.twiml.voice_response import Connect, VoiceResponse

from copilot.audit import AuditLog
from copilot.config import CopilotSettings
from copilot.phone.agent import GATEWAY_ACCOUNT
from copilot.phone.audio import frames
from copilot.phone.call import CallSession, PhoneDeps, Timing

logger = logging.getLogger(__name__)

BUSY = "Sorry, all our lines are busy right now. Please call again in a few minutes."
TOO_MANY = "Sorry, we can't take more calls from this number right now. Please try again later."


def caller_id(secret: str, number: str) -> tuple[str, str]:
    """(an HMAC of the caller's number, '…' + its last two digits): what the logs keep."""
    digest = hmac.new(secret.encode(), number.encode(), hashlib.sha256).hexdigest()[:16]
    digits = "".join(ch for ch in number if ch.isdigit())
    return digest, f"…{digits[-2:]}" if digits else "…"


class TwilioIO:
    """CallIO over a Media Streams WebSocket."""

    def __init__(self, websocket: WebSocket, stream_sid: str, gateway: PhoneGateway, call_sid: str) -> None:
        self.websocket, self.stream_sid, self.gateway, self.call_sid = websocket, stream_sid, gateway, call_sid

    async def _send(self, message: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):  # the caller may have hung up
            await self.websocket.send_text(json.dumps({**message, "streamSid": self.stream_sid}))

    async def play(self, mulaw: bytes, mark: str) -> None:
        for payload in frames(mulaw):
            await self._send({"event": "media", "media": {"payload": payload}})
        await self._send({"event": "mark", "mark": {"name": mark}})

    async def clear(self) -> None:
        await self._send({"event": "clear"})

    async def hang_up(self) -> None:
        self.gateway.outcomes[self.call_sid] = "hangup"
        with contextlib.suppress(Exception):
            await self.websocket.close()  # Twilio then asks /twilio/after, which hangs up

    async def transfer(self) -> None:
        self.gateway.outcomes[self.call_sid] = "transfer"
        with contextlib.suppress(Exception):
            await self.websocket.close()


class PhoneGateway:
    def __init__(self, settings: CopilotSettings, deps: PhoneDeps, *, timing: Timing | None = None,
                 audit: AuditLog | None = None) -> None:
        if settings.twilio_auth_token is None:
            raise SystemExit("TWILIO_AUTH_TOKEN is not set: the gateway can't check Twilio's signatures")
        self.settings, self.deps = settings, deps
        self.timing = timing or Timing(max_seconds=settings.phone_max_minutes * 60)
        self.audit = audit or deps.audit
        self.validator = RequestValidator(settings.twilio_auth_token.get_secret_value())
        self._secret = settings.jwt_secret()
        self.calls: dict[str, CallSession] = {}
        self.outcomes: dict[str, str] = {}  # call sid -> "transfer" / "hangup", for /twilio/after
        self.callers: dict[str, tuple[str, str]] = {}  # call sid -> (number hash, label)
        self._recent: dict[str, deque[float]] = defaultdict(deque)  # calls per number hash
        self.finished: list[CallSession] = []

    # --- signatures --------------------------------------------------------------------------

    def public_url(self, scope_path: str, headers: Any, *, websocket: bool = False) -> str:
        """The URL Twilio signed: PHONE_PUBLIC_URL, or the tunnel's host from the request."""
        if self.settings.phone_public_url:
            base = self.settings.phone_public_url.rstrip("/")
        else:
            proto = headers.get("x-forwarded-proto", "https")
            base = f"{proto}://{headers.get('host', 'localhost')}"
        if websocket:
            base = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        return base + scope_path

    def _signed(self, url: str, params: dict[str, Any], signature: str | None) -> bool:
        return bool(signature) and self.validator.validate(url, params, signature)

    async def _form(self, request: Request) -> dict[str, str] | None:
        form = {k: str(v) for k, v in (await request.form()).items()}
        url = self.public_url(request.url.path + (f"?{request.url.query}" if request.url.query else ""),
                              request.headers)
        if not self._signed(url, form, request.headers.get("x-twilio-signature")):
            logger.warning("rejected an unsigned or wrongly signed request to %s", request.url.path)
            return None
        if self.settings.twilio_account_sid and form.get("AccountSid") != self.settings.twilio_account_sid:
            logger.warning("rejected a request for another Twilio account")
            return None
        return form

    # --- webhooks ------------------------------------------------------------------------------

    async def voice(self, request: Request) -> Response:
        form = await self._form(request)
        if form is None:
            return PlainTextResponse("Forbidden", status_code=403)
        call_sid, number = form.get("CallSid", ""), form.get("From", "")
        digest, label = caller_id(self._secret, number)
        twiml = VoiceResponse()
        recent = self._recent[digest]
        while recent and recent[0] < time.monotonic() - 3600:
            recent.popleft()
        if len(self.calls) >= self.settings.phone_max_calls:
            twiml.say(BUSY)
            twiml.hangup()
            self._log(call_sid, label, "rejected", "busy")
        elif len(recent) >= self.settings.phone_calls_per_number_hour:
            twiml.say(TOO_MANY)
            twiml.hangup()
            self._log(call_sid, label, "rejected", "too many calls from this number")
        else:
            recent.append(time.monotonic())
            self.callers[call_sid] = (digest, label)
            connect = Connect()
            stream = connect.stream(url=self.public_url("/twilio/stream", request.headers, websocket=True))
            stream.parameter(name="caller", value=digest)
            stream.parameter(name="label", value=label)
            twiml.append(connect)
            twiml.redirect(self.public_url("/twilio/after", request.headers), method="POST")
        return Response(str(twiml), media_type="text/xml")

    async def after(self, request: Request) -> Response:
        form = await self._form(request)
        if form is None:
            return PlainTextResponse("Forbidden", status_code=403)
        twiml = VoiceResponse()
        outcome = self.outcomes.pop(form.get("CallSid", ""), "hangup")
        if outcome == "transfer" and self.settings.phone_transfer_number:
            twiml.dial(self.settings.phone_transfer_number)
        else:
            twiml.hangup()
        return Response(str(twiml), media_type="text/xml")

    async def status(self, request: Request) -> Response:
        form = await self._form(request)
        if form is None:
            return PlainTextResponse("Forbidden", status_code=403)
        call_sid = form.get("CallSid", "")
        if form.get("CallStatus") in ("completed", "busy", "failed", "no-answer", "canceled"):
            self.callers.pop(call_sid, None)
            self.outcomes.pop(call_sid, None)
        return PlainTextResponse("ok")

    # --- the media stream -------------------------------------------------------------------------

    async def stream(self, websocket: WebSocket) -> None:
        url = self.public_url("/twilio/stream", websocket.headers, websocket=True)
        if not self._signed(url, {}, websocket.headers.get("x-twilio-signature")):
            logger.warning("rejected an unsigned media stream")
            await websocket.close(code=1008)  # before accept(): the handshake fails with 403
            return
        await websocket.accept()
        session: CallSession | None = None
        try:
            while True:
                message = json.loads(await websocket.receive_text())
                event = message.get("event")
                if event == "start":
                    start = message["start"]
                    call_sid, params = start.get("callSid", ""), start.get("customParameters") or {}
                    io = TwilioIO(websocket, message.get("streamSid") or start.get("streamSid", ""), self, call_sid)
                    digest, label = self.callers.get(call_sid, (params.get("caller", "?"), params.get("label", "…")))
                    session = CallSession(call_sid, digest, label, io, self.deps, self.timing)
                    self.calls[call_sid] = session
                    await session.start()
                elif session is None:
                    continue
                elif event == "media" and message["media"].get("track", "inbound") == "inbound":
                    await session.on_audio(base64.b64decode(message["media"]["payload"]))
                elif event == "dtmf":
                    await session.on_dtmf(str(message["dtmf"].get("digit", "")))
                elif event == "mark":
                    await session.on_mark(message["mark"].get("name", ""))
                elif event == "stop":
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if session is not None:
                await session.on_stop()
                self.calls.pop(session.call_id, None)
                self.finished.append(session)
                self.finished[:] = self.finished[-50:]

    def _log(self, call_sid: str, label: str, outcome: str, detail: str) -> None:
        if self.audit is not None:
            self.audit.write(user=GATEWAY_ACCOUNT, role="phone", tool="call", source="phone", outcome=outcome,
                             detail=detail, context={"call": call_sid, "caller": label})

    def routes(self) -> list[Route | WebSocketRoute]:
        return [
            Route("/twilio/voice", self.voice, methods=["POST"]),
            Route("/twilio/after", self.after, methods=["POST"]),
            Route("/twilio/status", self.status, methods=["POST"]),
            WebSocketRoute("/twilio/stream", self.stream),
        ]

    def app(self, lifespan: Callable[[Starlette], Any] | None = None) -> Starlette:
        return Starlette(routes=[*self.routes(), Route("/health", lambda request: PlainTextResponse("ok"))],
                         lifespan=lifespan)


def runtime_lifespan(start: Callable[[], Awaitable[None]], stop: Callable[[], Awaitable[None]]) -> Any:
    """Start the phone runtime (and its TalkTable stdio session) on the server's own loop."""

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        await start()
        try:
            yield
        finally:
            await stop()

    return lifespan
