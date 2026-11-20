"""The Twilio gateway (copilot/phone/gateway.py): signed webhooks, TwiML, limits, and a call
over the Media Streams WebSocket, with fake speech, the scripted model and the real desk."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from pydantic import SecretStr
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from twilio.request_validator import RequestValidator

from copilot.auth import TokenIssuer
from copilot.phone.agent import PhoneRuntime
from copilot.phone.audio import FRAME_BYTES, SILENCE_BYTE, ulaw_encode
from copilot.phone.call import PhoneDeps, Timing
from copilot.phone.gateway import PhoneGateway, caller_id, runtime_lifespan
from tests.agent_fakes import ScriptedChatModel
from tests.desk_fakes import SECRET, Desk, running_desk
from tests.test_copilot_agent import ask
from tests.test_phone_call import Ears, Mouth

TOKEN, ACCOUNT, HOST = "test-auth-token", "AC0123", "phone.example"
BASE = f"https://{HOST}"


@pytest.fixture(scope="module")
def desk(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Desk]:
    with running_desk(tmp_path_factory.mktemp("gw-desk")) as server:
        yield server


def make_client(desk: Desk, *, max_calls: int = 2, transfer: str | None = None) -> tuple[TestClient, PhoneGateway, Ears]:
    desk.reset()
    settings = desk.settings.model_copy(update={
        "twilio_auth_token": SecretStr(TOKEN), "twilio_account_sid": ACCOUNT, "phone_max_calls": max_calls,
        "phone_transfer_number": transfer, "phone_calls_per_number_hour": 3})
    model, ears = ScriptedChatModel(), Ears()
    runtime = PhoneRuntime(settings, users=desk.users, model_factory=lambda: model, talktable_tools=[ask],
                           issuer=TokenIssuer(SECRET, settings.desk_url))
    deps = PhoneDeps(runtime, ears, Mouth(), settings, desk.audit, can_transfer=bool(transfer))
    gateway = PhoneGateway(settings, deps, timing=Timing(silence_nudge=60, silence_end=60))
    app = gateway.app(runtime_lifespan(runtime.start, runtime.close))
    return TestClient(app, base_url=BASE), gateway, ears


def sign(url: str, params: dict[str, str] | None = None) -> str:
    return RequestValidator(TOKEN).compute_signature(url, params or {})


def post(client: TestClient, path: str, params: dict[str, str], *, signature: str | None = "auto") -> Any:
    headers = {}
    if signature == "auto":
        headers["X-Twilio-Signature"] = sign(BASE + path, params)
    elif signature is not None:
        headers["X-Twilio-Signature"] = signature
    return client.post(path, data=params, headers=headers)


CALL = {"CallSid": "CA1", "AccountSid": ACCOUNT, "From": "+15551230042", "To": "+15559870000"}


# --- webhooks ---------------------------------------------------------------------------------


def test_incoming_call_gets_a_media_stream(desk: Desk) -> None:
    client, gateway, _ = make_client(desk)
    with client:
        response = post(client, "/twilio/voice", CALL)
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/xml")
    xml = response.text
    assert f'<Stream url="wss://{HOST}/twilio/stream">' in xml
    assert '<Parameter name="label" value="…42"' in xml and "+15551230042" not in xml  # the number isn't passed on
    assert f'<Redirect method="POST">{BASE}/twilio/after</Redirect>' in xml


@pytest.mark.parametrize("signature", [None, "bad-signature", sign(BASE + "/twilio/voice", {**CALL, "From": "+1999"})])
def test_unsigned_or_tampered_webhooks_are_refused(desk: Desk, signature: str | None) -> None:
    client, _, _ = make_client(desk)
    with client:
        assert post(client, "/twilio/voice", CALL, signature=signature).status_code == 403
        assert post(client, "/twilio/after", CALL, signature=signature).status_code == 403
        assert post(client, "/twilio/status", CALL, signature=signature).status_code == 403


def test_another_twilio_account_is_refused(desk: Desk) -> None:
    client, _, _ = make_client(desk)
    with client:
        assert post(client, "/twilio/voice", {**CALL, "AccountSid": "AC-other"}).status_code == 403


def test_calls_per_number_and_busy_lines(desk: Desk) -> None:
    client, gateway, _ = make_client(desk, max_calls=1)
    with client:
        for n in range(3):  # 3 calls/hour per number in this test
            assert "<Stream" in post(client, "/twilio/voice", {**CALL, "CallSid": f"CA{n}"}).text
        assert "can't take more calls from this number" in post(client, "/twilio/voice", {**CALL, "CallSid": "CA9"}).text
        gateway.calls["CA-live"] = object()  # type: ignore[assignment]  # one call in progress = the limit
        busy = post(client, "/twilio/voice", {**CALL, "CallSid": "CA10", "From": "+15550000001"}).text
        assert "all our lines are busy" in busy and "<Hangup" in busy


def test_after_the_stream(desk: Desk) -> None:
    client, gateway, _ = make_client(desk, transfer="+15550001111")
    with client:
        gateway.outcomes["CA1"] = "transfer"
        assert "<Dial>+15550001111</Dial>" in post(client, "/twilio/after", CALL).text
        assert "<Hangup" in post(client, "/twilio/after", CALL).text  # the outcome was used up


def test_caller_numbers_are_hashed() -> None:
    digest, label = caller_id("secret", "+15551230042")
    assert label == "…42" and len(digest) == 16 and "555" not in digest
    assert caller_id("secret", "+15551230042")[0] == digest != caller_id("other", "+15551230042")[0]


# --- the media stream -----------------------------------------------------------------------------


def test_unsigned_media_stream_is_refused(desk: Desk) -> None:
    client, gateway, _ = make_client(desk)
    with client, pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/twilio/stream") as ws:
            ws.receive_text()
    assert not gateway.calls


def until_mark(ws: Any) -> tuple[int, str]:
    """Read the gateway's messages until a mark: (audio frames received, mark name)."""
    count = 0
    while True:
        message = json.loads(ws.receive_text())
        assert message["streamSid"] == "MZ1"
        if message["event"] == "media":
            assert len(base64.b64decode(message["media"]["payload"])) == FRAME_BYTES
            count += 1
        elif message["event"] == "mark":
            return count, message["mark"]["name"]


def say_into(ws: Any) -> None:
    voice = ulaw_encode((np.random.default_rng(0).normal(0, 6000, 4800)).astype(np.int16))
    audio = voice + bytes([SILENCE_BYTE]) * 8000
    for i in range(0, len(audio), FRAME_BYTES):
        payload = base64.b64encode(audio[i:i + FRAME_BYTES]).decode()
        ws.send_text(json.dumps({"event": "media", "streamSid": "MZ1", "media": {"track": "inbound", "payload": payload}}))


def test_a_call_over_the_media_stream(desk: Desk) -> None:
    client, gateway, ears = make_client(desk)
    with client:
        post(client, "/twilio/voice", CALL)
        # Starlette's test client sends Host: testserver for WebSockets; behind the tunnel it is the tunnel's host
        with client.websocket_connect("/twilio/stream", headers={"X-Twilio-Signature": sign("wss://testserver/twilio/stream")}) as ws:
            ws.send_text(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
            ws.send_text(json.dumps({"event": "start", "streamSid": "MZ1", "start": {
                "streamSid": "MZ1", "callSid": "CA1", "accountSid": ACCOUNT, "tracks": ["inbound"],
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
                "customParameters": {"caller": "spoofed", "label": "spoofed"}}}))
            frames, mark = until_mark(ws)  # the greeting
            assert frames > 0 and mark == "say-1"
            ws.send_text(json.dumps({"event": "mark", "streamSid": "MZ1", "mark": {"name": mark}}))
            for digit in "5512#":  # the order number on the keypad
                ws.send_text(json.dumps({"event": "dtmf", "streamSid": "MZ1", "dtmf": {"track": "inbound_track", "digit": digit}}))
            _, mark = until_mark(ws)  # "And the last name?"
            ws.send_text(json.dumps({"event": "mark", "streamSid": "MZ1", "mark": {"name": mark}}))
            ears.lines.append("Gonzalez")
            say_into(ws)
            _, mark = until_mark(ws)  # "Thanks, Maria..."
            session = gateway.calls["CA1"]
            assert session.state == "help" and session.customer_id == 1
            assert session.caller == caller_id(gateway._secret, CALL["From"])[0]  # from the webhook, not the stream
            ws.send_text(json.dumps({"event": "stop", "streamSid": "MZ1", "stop": {"callSid": "CA1"}}))
        assert not gateway.calls and gateway.finished[-1].end_reason == "caller hung up"
    phone = [e for e in desk.audit.read() if e["source"] == "phone"]
    assert [(e["tool"], e["outcome"]) for e in phone] == [("call", "started"), ("verify", "ok"), ("call", "ended")]
    assert all(e["context"]["caller"] == "…42" for e in phone) and "+1555" not in desk.audit.path.read_text()
    assert re.search(r'"customer_id": 1', desk.audit.path.read_text())
