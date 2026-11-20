"""An in-memory Support Desk store with the same interface as PostgresDeskStore, so the
server's rules (auth, validation, duplicates, rate limit) are tested without a database.
The Postgres store itself is tested live in test_desk_store.py."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import uvicorn

from copilot.audit import AuditLog
from copilot.auth import TokenIssuer, UserStore
from copilot.config import CopilotSettings
from copilot.desk.server import create_server

from copilot.desk.store import (
    CustomerMatches,
    CustomerSummary,
    NewTicket,
    OrderList,
    OrderStatus,
    OrderSummary,
    Ticket,
    TicketList,
    TicketSummary,
)

CUSTOMERS = {
    1: ("Maria Gonzalez", "maria.gonzalez@example.com"),
    2: ("Sam Lee", "sam.lee@example.org"),
    3: ("Maria Smith", "maria.smith@example.net"),
}
ORDERS = {5512: 1, 6020: 2, 7781: 1}  # order_id -> customer_id


class FakeDeskStore:
    def __init__(self) -> None:
        self.tickets: dict[int, Ticket] = {}
        self.clock = datetime(2026, 11, 19, 10, 0, 0)
        self.writes = 0

    def _customer(self, customer_id: int) -> CustomerSummary:
        name, email = CUSTOMERS[customer_id]
        return CustomerSummary(customer_id=customer_id, name=name, email=email, city="Berlin",
                               signup_date=date(2024, 1, 1), loyalty_tier="gold",
                               orders=sum(c == customer_id for c in ORDERS.values()), open_tickets=0)

    def find_customers(self, query: str, limit: int) -> CustomerMatches:
        words = query.lower().split()
        found = [self._customer(i) for i, (name, email) in CUSTOMERS.items()
                 if query == str(i) or query.lower() == email or all(w in name.lower() for w in words)]
        return CustomerMatches(matches=found[:limit], more=len(found) > limit)

    def get_customer(self, customer_id: int) -> CustomerSummary | None:
        return self._customer(customer_id) if customer_id in CUSTOMERS else None

    def get_order(self, order_id: int) -> OrderStatus | None:
        if order_id not in ORDERS:
            return None
        customer_id = ORDERS[order_id]
        return OrderStatus(order_id=order_id, customer_id=customer_id, customer_name=CUSTOMERS[customer_id][0],
                           order_date=date(2026, 9, 1), status="shipped", total_amount=42.5, items=2, copies=3,
                           payment_method="card", payment_status="paid", carrier="SwiftShip",
                           shipped_date=date(2026, 9, 2), delivered_date=None, shipment_status="in_transit",
                           returned_items=0, refunded=0.0, return_reasons=[], open_tickets=0)

    def orders_of(self, customer_id: int, limit: int) -> OrderList:
        mine = sorted((o for o, c in ORDERS.items() if c == customer_id), reverse=True)
        found = [OrderSummary(order_id=o, order_date=date(2026, 9, 1), status="shipped", total_amount=42.5, items=2,
                              carrier="SwiftShip", shipment_status="in_transit") for o in mine]
        return OrderList(orders=found[:limit], more=len(found) > limit)

    def list_tickets(self, *, customer_id: int | None, order_id: int | None, open_only: bool,
                     limit: int) -> TicketList:
        found = [t for t in sorted(self.tickets.values(), key=lambda t: t.ticket_id, reverse=True)
                 if (customer_id is None or t.customer_id == customer_id)
                 and (order_id is None or t.order_id == order_id) and (not open_only or t.status == "open")]
        return TicketList(tickets=[TicketSummary(**t.model_dump()) for t in found[:limit]], more=len(found) > limit)

    def get_ticket(self, ticket_id: int) -> Ticket | None:
        return self.tickets.get(ticket_id)

    @contextmanager
    def ticket_writer(self, customer_id: int) -> Iterator[FakeDeskStore]:
        yield self

    # TicketWriter
    def tickets_by_user_since(self, username: str, minutes: int) -> int:
        since = self.clock - timedelta(minutes=minutes)
        return sum(t.created_by == username and t.opened_at >= since for t in self.tickets.values())

    def tickets_for_customer_by_since(self, customer_id: int, created_by: str, minutes: int) -> int:
        since = self.clock - timedelta(minutes=minutes)
        return sum(t.customer_id == customer_id and t.created_by == created_by and t.opened_at >= since
                   for t in self.tickets.values())

    def open_duplicate(self, customer_id: int, order_id: int | None, category: str, minutes: int) -> Ticket | None:
        since = self.clock - timedelta(minutes=minutes)
        for t in self.tickets.values():
            if (t.customer_id, t.order_id, t.category) == (customer_id, order_id, category) \
                    and t.status == "open" and t.opened_at >= since:
                return t
        return None

    def insert(self, new: NewTicket) -> Ticket:
        self.writes += 1
        ticket_id = 1451 + len(self.tickets)
        ticket = Ticket(**new.model_dump(), ticket_id=ticket_id, opened_at=self.clock, resolved_at=None,
                        satisfaction=None, status="open")
        self.tickets[ticket_id] = ticket
        return ticket


# --- a real desk server (HTTP, auth) over the fake store ------------------------------------

SECRET = "desk-test-secret-" + "z" * 40
PASSWORD = "correct horse battery"


@dataclass
class Desk:
    url: str
    settings: CopilotSettings
    store: FakeDeskStore
    users: UserStore
    audit: AuditLog

    def token(self, username: str) -> str:
        user = self.users.get(username)
        assert user is not None
        return TokenIssuer(SECRET, self.settings.desk_url, 900).mint(user)

    def reset(self) -> None:
        self.store.tickets.clear()
        self.store.writes = 0
        self.audit.path.unlink(missing_ok=True)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def running_desk(tmp: Path, **overrides: Any) -> Iterator[Desk]:
    """The real Support Desk server on a free port, with accounts alice (agent) and victor (viewer)."""
    port = _free_port()
    settings = CopilotSettings(_env_file=None, copilot_jwt_secret=SECRET, desk_port=port,
                               copilot_users_path=tmp / "users.yaml", copilot_audit_path=tmp / "audit.jsonl",
                               **overrides)
    users = UserStore(settings.users_file)
    users.add("alice", PASSWORD, "agent")
    users.add("victor", PASSWORD, "viewer")
    store, audit = FakeDeskStore(), AuditLog(settings.audit_file)
    app = create_server(settings, store=store, users=users, audit=audit).streamable_http_app()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "the desk server did not start"
        time.sleep(0.05)
    try:
        yield Desk(settings.desk_url, settings, store, users, audit)
    finally:
        server.should_exit = True
        thread.join(5)
