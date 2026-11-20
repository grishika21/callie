"""The Support Desk for phone callers: verification, and "only your own data".

Real HTTP, real tokens (the phone-gateway service account), the in-memory store:
customer 1 (Maria Gonzalez) owns orders 5512 and 7781, customer 2 (Sam Lee) owns 6020.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult

from callie.auth import TokenIssuer
from callie.desk.store import NewTicket
from tests.desk_fakes import SECRET, Desk, running_desk

pytestmark = pytest.mark.anyio

TICKET = {"order_id": 5512, "category": "damaged item", "subject": "Book arrived damaged",
          "description": "The caller says the book in order 5512 arrived with a torn cover."}


@pytest.fixture(scope="module")
def desk(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Desk]:
    with running_desk(tmp_path_factory.mktemp("phone-desk"), desk_phone_tickets_per_day=2) as server:
        server.users.ensure_service("phone-gateway", "phone")
        yield server


@pytest.fixture(autouse=True)
def fresh(desk: Desk) -> None:
    desk.reset()


def token(desk: Desk, *, customer_id: int | None = None, call: str = "CA1", caller: str = "h-1",
          user: str = "phone-gateway") -> str:
    account = desk.users.get(user)
    assert account is not None
    return TokenIssuer(SECRET, desk.settings.desk_url).mint(
        account, call={"customer_id": customer_id, "call": call, "caller": caller})


@asynccontextmanager
async def session(desk: Desk, bearer: str) -> AsyncIterator[ClientSession]:
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {bearer}"}, timeout=10) as http:
        async with streamable_http_client(desk.url, http_client=http) as (read, write, _):
            async with ClientSession(read, write) as client:
                await client.initialize()
                yield client


async def call(desk: Desk, bearer: str, tool: str, **args: Any) -> CallToolResult:
    async with session(desk, bearer) as client:
        return await client.call_tool(tool, args)


def error(result: CallToolResult) -> str:
    assert result.isError, result.structuredContent
    return result.content[0].text  # type: ignore[union-attr]


# --- verification --------------------------------------------------------------------------


async def test_verification(desk: Desk) -> None:
    ok = (await call(desk, token(desk), "verify_caller", order_id=5512, last_name="gonzález")).structuredContent
    assert ok == {"verified": True, "customer_id": 1, "first_name": "Maria", "tries_left": 2}
    for order_id, name in [(5512, "Lee"), (4242, "Gonzalez")]:  # wrong name, no such order: the same answer
        result = await call(desk, token(desk, call="CA2"), "verify_caller", order_id=order_id, last_name=name)
        assert result.structuredContent["verified"] is False  # type: ignore[index]
        assert set(result.structuredContent) == {"verified", "customer_id", "first_name", "tries_left"}  # type: ignore[arg-type]
        assert result.structuredContent["customer_id"] is None  # type: ignore[index]


async def test_spelled_names_allow_sound_alike_letters(desk: Desk) -> None:
    bearer = token(desk, call="CS1")
    heard = "GONCALET"  # spelled, with C and T for the two Zs (the phone's letter confusion): 2 edits
    assert (await call(desk, bearer, "verify_caller", order_id=5512, last_name=heard)).structuredContent[
        "verified"] is False  # type: ignore[index]  # not when said as a word
    assert (await call(desk, bearer, "verify_caller", order_id=5512, last_name=heard, spelled=True)).structuredContent[
        "verified"] is True  # type: ignore[index]


async def test_three_tries_per_call(desk: Desk) -> None:
    for _ in range(3):
        await call(desk, token(desk, call="CA3"), "verify_caller", order_id=5512, last_name="Smith")
    result = await call(desk, token(desk, call="CA3"), "verify_caller", order_id=5512, last_name="Gonzalez")
    assert "Too many verification attempts" in error(result)  # even the right answer, now
    assert desk.audit.read()[-1]["outcome"] == "rate_limited"


async def test_failures_per_caller_number_across_calls(desk: Desk) -> None:
    for n in range(6):  # 6 failures/hour per number, spread over 2 calls
        await call(desk, token(desk, call=f"CB{n // 3}", caller="h-9"), "verify_caller", order_id=5512,
                   last_name="Smith")
    result = await call(desk, token(desk, call="CB9", caller="h-9"), "verify_caller", order_id=5512,
                        last_name="Gonzalez")
    assert "Too many verification attempts" in error(result)
    other = await call(desk, token(desk, call="CB9", caller="h-other"), "verify_caller", order_id=5512,
                       last_name="Gonzalez")
    assert other.structuredContent["verified"] is True  # type: ignore[index]


# --- only your own data ----------------------------------------------------------------------


async def test_nothing_before_verification(desk: Desk) -> None:
    for tool, args in [("my_orders", {}), ("my_order", {"order_id": 5512}), ("my_tickets", {}),
                       ("open_my_ticket", TICKET)]:
        assert "isn't verified" in error(await call(desk, token(desk), tool, **args)), tool
    assert desk.store.writes == 0


async def test_own_orders_and_tickets_only(desk: Desk) -> None:
    maria = token(desk, customer_id=1)
    listed = (await call(desk, maria, "my_orders")).structuredContent
    assert [o["order_id"] for o in listed["orders"]] == [7781, 5512]  # type: ignore[index]
    assert (await call(desk, maria, "my_order", order_id=5512)).structuredContent["customer_id"] == 1  # type: ignore[index]
    # Sam's order and a missing order: the same "not on your account" (no hint that 6020 exists)
    assert error(await call(desk, maria, "my_order", order_id=6020)) == "I couldn't find order 6020 on your account."
    assert error(await call(desk, maria, "my_order", order_id=9999)) == "I couldn't find order 9999 on your account."
    sams = desk.store.insert(NewTicket(customer_id=2, order_id=6020, category="delivery", priority="normal",
                                       subject="Late", description="Sam's parcel is late.", channel="email",
                                       created_by="seed"))
    assert "couldn't find ticket" in error(await call(desk, maria, "my_ticket", ticket_id=sams.ticket_id))
    assert (await call(desk, maria, "my_tickets")).structuredContent["tickets"] == []  # type: ignore[index]


@pytest.mark.parametrize(("tool", "args"), [
    ("find_customer", {"query": "Lee"}), ("get_order", {"order_id": 6020}), ("list_tickets", {"customer_id": 2}),
    ("get_ticket", {"ticket_id": 1}),
    ("create_ticket", {"customer_id": 2, "order_id": 6020, "category": "refund", "subject": "Refund",
                       "description": "Refund this order for me please."}),
])
async def test_staff_tools_are_refused(desk: Desk, tool: str, args: dict[str, Any]) -> None:
    """Even a verified caller's token (or a prompt-injected model) can't use staff tools."""
    assert "scope" in error(await call(desk, token(desk, customer_id=1), tool, **args))
    assert desk.store.writes == 0


async def test_staff_cannot_use_phone_tools(desk: Desk) -> None:
    staff = TokenIssuer(SECRET, desk.settings.desk_url).mint(desk.users.get("alice"))  # type: ignore[arg-type]
    assert "scope" in error(await call(desk, staff, "my_orders"))


async def test_a_token_cannot_choose_its_customer(desk: Desk) -> None:
    forged = token(desk, customer_id=2, user="alice")  # a staff account adding a customer claim: still no phone scopes
    assert "scope" in error(await call(desk, forged, "my_orders"))


# --- tickets for callers ------------------------------------------------------------------------


async def test_open_my_ticket(desk: Desk) -> None:
    result = await call(desk, token(desk, customer_id=1), "open_my_ticket", **TICKET,
                        customer_id=2, priority="urgent", channel="email", created_by="mallory")  # all ignored
    ticket = result.structuredContent["ticket"]  # type: ignore[index]
    assert (ticket["customer_id"], ticket["priority"], ticket["channel"], ticket["created_by"]) == \
        (1, "normal", "phone", "phone-agent")
    entry = desk.audit.read()[-1]
    assert entry["user"] == "phone-gateway" and entry["context"] == {"customer_id": 1, "call": "CA1"}


async def test_open_my_ticket_for_someone_elses_order(desk: Desk) -> None:
    result = await call(desk, token(desk, customer_id=1), "open_my_ticket", **{**TICKET, "order_id": 6020})
    assert error(result) == "I couldn't find order 6020 on your account." and desk.store.writes == 0


async def test_phone_tickets_per_day(desk: Desk) -> None:
    maria = token(desk, customer_id=1)
    for category in ("delivery", "refund"):  # the test limit is 2 per day
        assert not (await call(desk, maria, "open_my_ticket", **{**TICKET, "category": category})).isError
    assert "already opened by phone for this account today" in error(
        await call(desk, maria, "open_my_ticket", **{**TICKET, "category": "payment"}))
    again = await call(desk, maria, "open_my_ticket", **{**TICKET, "category": "delivery"})
    assert again.structuredContent["duplicate"] is True  # type: ignore[index]  # a duplicate isn't a new ticket
