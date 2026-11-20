"""The Support Desk MCP server over real HTTP, with real tokens and the real MCP client.

The database is the in-memory FakeDeskStore (tests/desk_fakes.py), so these run
anywhere; the Postgres store is tested live in test_desk_store.py.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult

from copilot.config import CopilotSettings
from copilot.desk.server import clean_text, create_server
from copilot.desk.store import Category, Priority
from data.seed.extended import TICKET_CATEGORIES, TICKET_PRIORITIES
from tests.desk_fakes import PASSWORD, SECRET, Desk, running_desk

pytestmark = pytest.mark.anyio

TICKET = {"customer_id": 1, "order_id": 5512, "category": "damaged item", "priority": "high",
          "subject": "Book arrived damaged", "description": "The cover of the book in order 5512 is torn."}


@pytest.fixture(scope="module")
def desk(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Desk]:
    with running_desk(tmp_path_factory.mktemp("desk"), desk_tickets_per_hour=3) as server:
        yield server


@pytest.fixture(autouse=True)
def fresh_state(request: pytest.FixtureRequest) -> None:
    if "desk" in request.fixturenames:
        request.getfixturevalue("desk").reset()


@asynccontextmanager
async def session(desk: Desk, token: str) -> AsyncIterator[ClientSession]:
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=10) as http:
        async with streamable_http_client(desk.url, http_client=http) as (read, write, _):
            async with ClientSession(read, write) as client:
                await client.initialize()
                yield client


async def call(desk: Desk, user: str, tool: str, **args: Any) -> CallToolResult:
    async with session(desk, desk.token(user)) as client:
        return await client.call_tool(tool, args)


def error_text(result: CallToolResult) -> str:
    assert result.isError, result.structuredContent
    return result.content[0].text  # type: ignore[union-attr]


# --- authentication ---------------------------------------------------------------------

def _post(desk: Desk, token: str | None) -> httpx.Response:
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    return httpx.post(desk.url, json=body, headers=headers, timeout=5)


def _forge(desk: Desk, **overrides: Any) -> str:
    now = int(time.time())
    claims = {"iss": "talktable-copilot", "aud": desk.settings.desk_url, "sub": "alice", "role": "agent",
              "scope": "data:read desk:read tickets:write", "iat": now, "exp": now + 600, **overrides}
    return jwt.encode(claims, SECRET, algorithm="HS256")


def test_no_token_is_401(desk: Desk) -> None:
    response = _post(desk, None)
    assert response.status_code == 401
    assert 'error="invalid_token"' in response.headers["www-authenticate"]
    assert "oauth-protected-resource" in response.headers["www-authenticate"]


@pytest.mark.parametrize("make_token", [
    lambda d: "garbage",
    lambda d: _forge(d, exp=int(time.time()) - 60),                  # expired
    lambda d: _forge(d, aud="http://127.0.0.1:1/mcp"),               # another audience
    lambda d: jwt.encode(jwt.decode(_forge(d), options={"verify_signature": False}),
                         "wrong-key-" + "k" * 40, algorithm="HS256"),  # wrong key
    lambda d: _forge(d)[:-4] + "AAAA",                               # tampered signature
    lambda d: _forge(d, sub="mallory"),                              # no such account
], ids=["garbage", "expired", "audience", "key", "tampered", "unknown-user"])
def test_bad_tokens_are_401(desk: Desk, make_token: Any) -> None:
    assert _post(desk, make_token(desk)).status_code == 401


async def test_a_token_without_desk_scopes_can_use_no_tool(desk: Desk) -> None:
    """Any valid token reaches the endpoint (the phone gateway has no desk:read), but every
    tool checks its own scope."""
    async with session(desk, _forge(desk, scope="data:read")) as client:
        for tool, args in [("find_customer", {"query": "1"}), ("get_order", {"order_id": 5512}),
                           ("create_ticket", TICKET), ("verify_caller", {"order_id": 5512, "last_name": "Gonzalez"}),
                           ("my_orders", {}), ("open_my_ticket", {k: v for k, v in TICKET.items()
                                                                  if k not in ("customer_id", "priority")})]:
            assert "scope" in error_text(await client.call_tool(tool, args)), tool
    assert desk.store.writes == 0


async def test_agent_lists_and_calls_tools(desk: Desk) -> None:
    async with session(desk, desk.token("alice")) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools) == {"find_customer", "get_order", "list_tickets", "get_ticket", "create_ticket",
                              "verify_caller", "my_orders", "my_order", "my_tickets", "my_ticket", "open_my_ticket"}
        assert tools["get_order"].annotations.readOnlyHint is True  # type: ignore[union-attr]
        assert tools["create_ticket"].annotations.readOnlyHint is False  # type: ignore[union-attr]
        schema = tools["create_ticket"].inputSchema["properties"]
        assert set(schema["category"]["enum"]) == set(TICKET_CATEGORIES)
        assert schema["order_id"]["type"] == "integer"  # a plain int: 0 = no order (V0 finding)
        assert "created_by" not in schema and "channel" not in schema  # set by the server only
        found = await client.call_tool("find_customer", {"query": "maria"})
        assert [m["customer_id"] for m in found.structuredContent["matches"]] == [1, 3]  # type: ignore[index]
        order = await client.call_tool("get_order", {"order_id": 5512})
        assert order.structuredContent["customer_name"] == "Maria Gonzalez"  # type: ignore[index]


def test_literals_match_the_database_constraints() -> None:
    assert set(Category.__args__) == set(TICKET_CATEGORIES)  # type: ignore[attr-defined]
    assert set(Priority.__args__) == set(TICKET_PRIORITIES)  # type: ignore[attr-defined]


# --- authorization ------------------------------------------------------------------------


async def test_viewer_reads_but_cannot_open_tickets(desk: Desk) -> None:
    assert not (await call(desk, "victor", "get_order", order_id=5512)).isError
    result = await call(desk, "victor", "create_ticket", **TICKET)
    assert "needs the tickets:write scope (your role: viewer)" in error_text(result)
    assert desk.store.writes == 0
    assert desk.audit.read()[-1] | {"ts": 0, "latency_ms": 0} == {
        "ts": 0, "latency_ms": 0, "source": "desk", "user": "victor", "role": "viewer", "tool": "create_ticket",
        "outcome": "denied", "args": TICKET, "detail": "This needs the tickets:write scope (your role: viewer)."}


async def test_demoted_user_loses_write_access_immediately(desk: Desk) -> None:
    token = desk.token("alice")
    desk.users.add("alice", PASSWORD, "viewer")
    try:
        async with session(desk, token) as client:
            result = await client.call_tool("create_ticket", TICKET)
        assert "tickets:write" in error_text(result) and desk.store.writes == 0
    finally:
        desk.users.add("alice", PASSWORD, "agent")


async def test_server_decides_who_when_and_how(desk: Desk) -> None:
    """The model can't choose created_by, channel or opened_at: extra arguments are ignored."""
    result = await call(desk, "alice", "create_ticket", **TICKET, created_by="mallory", channel="email",
                        opened_at="2020-01-01T00:00:00")
    ticket = result.structuredContent["ticket"]  # type: ignore[index]
    assert not result.isError and result.structuredContent["duplicate"] is False  # type: ignore[index]
    assert (ticket["created_by"], ticket["channel"], ticket["opened_at"]) == ("alice", "voice", "2026-11-19T10:00:00")
    assert result.structuredContent["message"] == "Opened ticket 1451 for Maria Gonzalez."  # type: ignore[index]
    assert desk.audit.read()[-1]["ticket_id"] == 1451


# --- validation ------------------------------------------------------------------------------


@pytest.mark.parametrize(("changes", "message"), [
    ({"category": "spam"}, "Invalid arguments for create_ticket: category: Input should be"),
    ({"priority": "critical"}, "priority: Input should be"),
    ({"subject": "x" * 121}, "subject: String should have at most 120 characters"),
    ({"description": "short"}, "description: String should have at least 10 characters"),
    ({"customer_id": "{\"$eq\": 1}"}, "customer_id: Input should be a valid integer"),  # a V0 4B mistake
    ({"order_id": -1}, "order_id: Input should be greater than or equal to 0"),
    ({"subject": "​​\x07\x07"}, "The subject is empty"),
    ({"customer_id": 99}, "There is no customer 99"),
    ({"order_id": 4242}, "There is no order 4242"),
    ({"order_id": 6020}, "Order 6020 does not belong to customer 1"),
])
async def test_bad_tickets_are_rejected(desk: Desk, changes: dict[str, Any], message: str) -> None:
    result = await call(desk, "alice", "create_ticket", **{**TICKET, **changes})
    assert message in error_text(result)
    assert desk.store.writes == 0
    assert desk.audit.read()[-1]["outcome"] in {"invalid", "rejected"}


async def test_text_is_cleaned(desk: Desk) -> None:
    result = await call(desk, "alice", "create_ticket", **{**TICKET, "subject": "  Book‮ arrived\x00\x07  damaged ",
                                                           "description": "Line one\x1b[31m\n\n\n\nline two​"})
    ticket = result.structuredContent["ticket"]  # type: ignore[index]
    assert ticket["subject"] == "Book arrived damaged"
    assert ticket["description"] == "Line one[31m\n\nline two"


def test_clean_text() -> None:
    assert clean_text("a\tb\r\nc", multiline=False) == "a b c"
    assert clean_text("é", multiline=False) == "é"  # NFC


async def test_ticket_without_an_order(desk: Desk) -> None:
    result = await call(desk, "alice", "create_ticket", **{**TICKET, "order_id": 0, "category": "account",
                                                           "subject": "Cannot log in",
                                                           "description": "The password reset email never arrives."})
    assert result.structuredContent["ticket"]["order_id"] is None  # type: ignore[index]


# --- abuse limits -----------------------------------------------------------------------------


async def test_duplicates_return_the_open_ticket(desk: Desk) -> None:
    first = await call(desk, "alice", "create_ticket", **TICKET)
    again = await call(desk, "alice", "create_ticket", **{**TICKET, "subject": "Torn cover", "priority": "urgent"})
    assert again.structuredContent["duplicate"] is True  # type: ignore[index]
    assert again.structuredContent["ticket"]["ticket_id"] == first.structuredContent["ticket"]["ticket_id"]  # type: ignore[index]
    assert "already open" in again.structuredContent["message"]  # type: ignore[index]
    assert desk.store.writes == 1
    assert [e["outcome"] for e in desk.audit.read()] == ["ok", "duplicate"]
    other = await call(desk, "alice", "create_ticket", **{**TICKET, "category": "delivery"})  # not a duplicate
    assert other.structuredContent["duplicate"] is False and desk.store.writes == 2  # type: ignore[index]


async def test_hourly_rate_limit(desk: Desk) -> None:
    for category in ("delivery", "payment", "refund"):  # the test limit is 3 per hour
        assert not (await call(desk, "alice", "create_ticket", **{**TICKET, "category": category})).isError
    result = await call(desk, "alice", "create_ticket", **{**TICKET, "category": "other"})
    assert "You have opened 3 tickets in the last hour" in error_text(result)
    assert desk.store.writes == 3 and desk.audit.read()[-1]["outcome"] == "rate_limited"


# --- reads, errors, audit -----------------------------------------------------------------------


async def test_lookups(desk: Desk) -> None:
    await call(desk, "alice", "create_ticket", **TICKET)
    async with session(desk, desk.token("victor")) as client:
        listed = (await client.call_tool("list_tickets", {"customer_id": 1, "open_only": True})).structuredContent
        assert [t["ticket_id"] for t in listed["tickets"]] == [1451]  # type: ignore[index]
        full = (await client.call_tool("get_ticket", {"ticket_id": 1451})).structuredContent
        assert full["description"] == TICKET["description"]  # type: ignore[index]
        assert "Give a customer_id or an order_id" in error_text(await client.call_tool("list_tickets", {}))
        assert "There is no ticket 9" in error_text(await client.call_tool("get_ticket", {"ticket_id": 9}))
        assert "There is no order 1" in error_text(await client.call_tool("get_order", {"order_id": 1}))
        assert "Unknown tool" in error_text(await client.call_tool("delete_ticket", {"ticket_id": 1}))


async def test_internal_errors_stay_internal(desk: Desk, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(order_id: int) -> None:
        raise RuntimeError("connection to postgresql://talktable_desk:hunter2@db failed")

    monkeypatch.setattr(desk.store, "get_order", broken)
    text = error_text(await call(desk, "alice", "get_order", order_id=5512))
    assert text == "Something went wrong inside the Support Desk. The server log has the details."
    entry = desk.audit.read()[-1]
    assert entry["outcome"] == "error" and "hunter2" not in str(entry)


async def test_audit_masks_emails(desk: Desk) -> None:
    await call(desk, "victor", "find_customer", query="maria.gonzalez@example.com")
    entry = desk.audit.read()[-1]
    assert entry["args"] == {"query": "m***@example.com"} and entry["outcome"] == "ok"
    assert entry["user"] == "victor" and entry["latency_ms"] >= 0
    assert "maria.gonzalez" not in desk.audit.path.read_text()


def test_requires_a_database_url(tmp_path: Path) -> None:
    settings = CopilotSettings(_env_file=None, copilot_jwt_secret=SECRET, desk_postgres_url=None,
                               copilot_users_path=tmp_path / "u.yaml")
    with pytest.raises(SystemExit, match="DESK_POSTGRES_URL"):
        create_server(settings)
