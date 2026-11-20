"""The Support Desk's database access: fixed, parameterized queries only.

No SQL here is written by a model. Reads run in READ ONLY transactions with a statement
timeout (as in talktable/adapters/postgres_adapter.py). Ticket creation runs in one
transaction that first takes an advisory lock on the customer, so two quick requests
can't both pass the duplicate check. The database login (`talktable_desk`) can't do
more than this anyway: see data/seed/desk_role.py.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import date, datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel
from sqlalchemy import Connection, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

Category = Literal["delivery", "payment", "damaged item", "refund", "account", "other"]
Priority = Literal["low", "normal", "high", "urgent"]
Channel = Literal["email", "chat", "phone", "voice"]

_LOCK_NAMESPACE = 8765  # pg_advisory_xact_lock(namespace, customer_id)
_MAX_NAME_WORDS = 4


class DeskUnavailableError(Exception):
    """The support database can't be reached right now (safe to show)."""


# --- results -----------------------------------------------------------------------


class CustomerSummary(BaseModel):
    customer_id: int
    name: str
    email: str
    city: str
    signup_date: date
    loyalty_tier: str | None
    orders: int
    open_tickets: int


class CustomerMatches(BaseModel):
    matches: list[CustomerSummary]
    more: bool  # True when more customers matched than are listed


class OrderStatus(BaseModel):
    order_id: int
    customer_id: int
    customer_name: str
    order_date: date
    status: str
    total_amount: float
    items: int  # order lines
    copies: int
    payment_method: str | None
    payment_status: str | None
    carrier: str | None
    shipped_date: date | None
    delivered_date: date | None
    shipment_status: str | None
    returned_items: int
    refunded: float
    return_reasons: list[str]
    open_tickets: int


class OrderSummary(BaseModel):
    order_id: int
    order_date: date
    status: str
    total_amount: float
    items: int
    carrier: str | None
    shipment_status: str | None


class OrderList(BaseModel):
    orders: list[OrderSummary]
    more: bool


class TicketSummary(BaseModel):
    ticket_id: int
    customer_id: int
    order_id: int | None
    category: str
    priority: str
    channel: str
    subject: str
    opened_at: datetime
    resolved_at: datetime | None
    status: Literal["open", "resolved"]


class Ticket(TicketSummary):
    description: str
    satisfaction: int | None
    created_by: str | None


class TicketList(BaseModel):
    tickets: list[TicketSummary]
    more: bool


class NewTicket(BaseModel):
    customer_id: int
    order_id: int | None
    category: Category
    priority: Priority
    subject: str
    description: str
    channel: Channel
    created_by: str


# --- the interface the server uses (tests use an in-memory one) -----------------------


class TicketWriter(Protocol):
    def tickets_by_user_since(self, username: str, minutes: int) -> int: ...
    def tickets_for_customer_by_since(self, customer_id: int, created_by: str, minutes: int) -> int: ...
    def open_duplicate(self, customer_id: int, order_id: int | None, category: str, minutes: int) -> Ticket | None: ...
    def insert(self, new: NewTicket) -> Ticket: ...


class DeskStore(Protocol):
    def find_customers(self, query: str, limit: int) -> CustomerMatches: ...
    def get_customer(self, customer_id: int) -> CustomerSummary | None: ...
    def get_order(self, order_id: int) -> OrderStatus | None: ...
    def orders_of(self, customer_id: int, limit: int) -> OrderList: ...
    def list_tickets(self, *, customer_id: int | None, order_id: int | None, open_only: bool,
                     limit: int) -> TicketList: ...
    def get_ticket(self, ticket_id: int) -> Ticket | None: ...
    def ticket_writer(self, customer_id: int) -> AbstractContextManager[TicketWriter]: ...


# --- Postgres ------------------------------------------------------------------------

_CUSTOMER_SQL = """
SELECT c.customer_id, c.name, c.email, c.city, c.signup_date, la.tier AS loyalty_tier,
       (SELECT count(*) FROM orders o WHERE o.customer_id = c.customer_id) AS orders,
       (SELECT count(*) FROM support_tickets t
         WHERE t.customer_id = c.customer_id AND t.resolved_at IS NULL) AS open_tickets
FROM customers c
LEFT JOIN loyalty_accounts la ON la.customer_id = c.customer_id
WHERE {where}
ORDER BY c.name, c.customer_id
LIMIT :limit
"""

_ORDER_SQL = """
SELECT o.order_id, o.customer_id, c.name AS customer_name, o.order_date, o.status, o.total_amount,
       li.items, li.copies, p.method AS payment_method, p.status AS payment_status,
       s.carrier, s.shipped_date, s.delivered_date, s.status AS shipment_status,
       r.returned_items, r.refunded, r.return_reasons,
       (SELECT count(*) FROM support_tickets t
         WHERE t.order_id = o.order_id AND t.resolved_at IS NULL) AS open_tickets
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
CROSS JOIN LATERAL (SELECT count(*) AS items, coalesce(sum(i.quantity), 0) AS copies
                    FROM order_items i WHERE i.order_id = o.order_id) li
LEFT JOIN LATERAL (SELECT method, status FROM payments
                   WHERE order_id = o.order_id ORDER BY paid_at DESC LIMIT 1) p ON true
LEFT JOIN LATERAL (SELECT ca.name AS carrier, sh.shipped_date, sh.delivered_date, sh.status
                   FROM shipments sh JOIN carriers ca ON ca.carrier_id = sh.carrier_id
                   WHERE sh.order_id = o.order_id ORDER BY sh.shipped_date DESC LIMIT 1) s ON true
CROSS JOIN LATERAL (SELECT count(*) AS returned_items, coalesce(sum(rt.refund_amount), 0) AS refunded,
                           coalesce(array_agg(DISTINCT rt.reason) FILTER (WHERE rt.reason IS NOT NULL),
                                    '{}') AS return_reasons
                    FROM returns rt JOIN order_items i ON i.order_item_id = rt.order_item_id
                    WHERE i.order_id = o.order_id) r
WHERE o.order_id = :order_id
"""

_TICKET_COLUMNS = ("ticket_id, customer_id, order_id, category, priority, channel, subject, opened_at, "
                   "resolved_at, description, satisfaction, created_by")


def _like_escape(word: str) -> str:
    return word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _ticket(row: Any) -> Ticket:
    data = dict(row._mapping)
    return Ticket(**data, status="open" if data["resolved_at"] is None else "resolved")


class _PgTicketWriter:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def tickets_by_user_since(self, username: str, minutes: int) -> int:
        return self._conn.execute(text(
            "SELECT count(*) FROM support_tickets "
            "WHERE created_by = :user AND opened_at >= LOCALTIMESTAMP - make_interval(mins => :minutes)"),
            {"user": username, "minutes": minutes}).scalar_one()

    def tickets_for_customer_by_since(self, customer_id: int, created_by: str, minutes: int) -> int:
        return self._conn.execute(text(
            "SELECT count(*) FROM support_tickets WHERE customer_id = :customer_id AND created_by = :created_by "
            "AND opened_at >= LOCALTIMESTAMP - make_interval(mins => :minutes)"),
            {"customer_id": customer_id, "created_by": created_by, "minutes": minutes}).scalar_one()

    def open_duplicate(self, customer_id: int, order_id: int | None, category: str, minutes: int) -> Ticket | None:
        row = self._conn.execute(text(
            f"SELECT {_TICKET_COLUMNS} FROM support_tickets "
            "WHERE customer_id = :customer_id AND order_id IS NOT DISTINCT FROM CAST(:order_id AS integer) "
            "AND category = :category AND resolved_at IS NULL "
            "AND opened_at >= LOCALTIMESTAMP - make_interval(mins => :minutes) "
            "ORDER BY opened_at DESC, ticket_id DESC LIMIT 1"),
            {"customer_id": customer_id, "order_id": order_id, "category": category, "minutes": minutes}).first()
        return _ticket(row) if row else None

    def insert(self, new: NewTicket) -> Ticket:
        values = new.model_dump()
        columns = ", ".join(values)
        params = ", ".join(f":{name}" for name in values)
        row = self._conn.execute(text(
            f"INSERT INTO support_tickets ({columns}) VALUES ({params}) RETURNING {_TICKET_COLUMNS}"), values).one()
        return _ticket(row)


class PostgresDeskStore:
    def __init__(self, url: str, *, timeout_s: float = 5.0, connect_timeout_s: int = 5) -> None:
        self._timeout_ms = max(1, round(timeout_s * 1000))
        self._engine: Engine = create_engine(
            url, pool_size=3, max_overflow=2, pool_pre_ping=True, pool_recycle=300,
            connect_args={"connect_timeout": connect_timeout_s, "application_name": "talktable-desk"})

    @contextmanager
    def _read(self) -> Iterator[Connection]:
        try:
            with self._engine.connect() as conn:
                conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                conn.exec_driver_sql(f"SET LOCAL statement_timeout = {self._timeout_ms}")
                try:
                    yield conn
                finally:
                    conn.rollback()
        except SQLAlchemyError as exc:
            raise self._unavailable(exc) from None

    def _unavailable(self, exc: SQLAlchemyError) -> Exception:
        logger.warning("support desk query failed: %s", str(exc).splitlines()[0][:300])
        return DeskUnavailableError("The support database didn't answer. Try again in a moment.")

    def find_customers(self, query: str, limit: int = 5) -> CustomerMatches:
        query = query.strip()
        params: dict[str, Any] = {"limit": limit + 1}
        if query.isdigit():
            where, params["customer_id"] = "c.customer_id = :customer_id", int(query)
        elif "@" in query:
            where, params["email"] = "lower(c.email) = lower(:email)", query
        else:
            words = query.split()[:_MAX_NAME_WORDS] or [""]
            where = " AND ".join(f"c.name ILIKE :w{i} ESCAPE '\\'" for i in range(len(words)))
            params |= {f"w{i}": f"%{_like_escape(w)}%" for i, w in enumerate(words)}
        with self._read() as conn:
            rows = conn.execute(text(_CUSTOMER_SQL.format(where=where)), params).all()
        matches = [CustomerSummary(**row._mapping) for row in rows]
        return CustomerMatches(matches=matches[:limit], more=len(matches) > limit)

    def get_customer(self, customer_id: int) -> CustomerSummary | None:
        with self._read() as conn:
            row = conn.execute(text(_CUSTOMER_SQL.format(where="c.customer_id = :customer_id")),
                               {"customer_id": customer_id, "limit": 1}).first()
        return CustomerSummary(**row._mapping) if row else None

    def get_order(self, order_id: int) -> OrderStatus | None:
        with self._read() as conn:
            row = conn.execute(text(_ORDER_SQL), {"order_id": order_id}).first()
        return OrderStatus(**row._mapping) if row else None

    def orders_of(self, customer_id: int, limit: int = 5) -> OrderList:
        """A customer's newest orders, for a caller who doesn't know their order number."""
        with self._read() as conn:
            rows = conn.execute(text(
                "SELECT o.order_id, o.order_date, o.status, o.total_amount, "
                "(SELECT count(*) FROM order_items i WHERE i.order_id = o.order_id) AS items, "
                "s.carrier, s.status AS shipment_status FROM orders o "
                "LEFT JOIN LATERAL (SELECT ca.name AS carrier, sh.status FROM shipments sh "
                "  JOIN carriers ca ON ca.carrier_id = sh.carrier_id WHERE sh.order_id = o.order_id "
                "  ORDER BY sh.shipped_date DESC LIMIT 1) s ON true "
                "WHERE o.customer_id = :customer_id ORDER BY o.order_date DESC, o.order_id DESC LIMIT :limit"),
                {"customer_id": customer_id, "limit": limit + 1}).all()
        orders = [OrderSummary(**row._mapping) for row in rows]
        return OrderList(orders=orders[:limit], more=len(orders) > limit)

    def list_tickets(self, *, customer_id: int | None = None, order_id: int | None = None, open_only: bool = False,
                     limit: int = 10) -> TicketList:
        conditions, params = ["true"], {"limit": limit + 1}
        if customer_id is not None:
            conditions.append("customer_id = :customer_id")
            params["customer_id"] = customer_id
        if order_id is not None:
            conditions.append("order_id = :order_id")
            params["order_id"] = order_id
        if open_only:
            conditions.append("resolved_at IS NULL")
        with self._read() as conn:
            rows = conn.execute(text(f"SELECT {_TICKET_COLUMNS} FROM support_tickets WHERE {' AND '.join(conditions)} "
                                     "ORDER BY opened_at DESC, ticket_id DESC LIMIT :limit"), params).all()
        tickets = [TicketSummary(**_ticket(row).model_dump()) for row in rows]
        return TicketList(tickets=tickets[:limit], more=len(tickets) > limit)

    def get_ticket(self, ticket_id: int) -> Ticket | None:
        with self._read() as conn:
            row = conn.execute(text(f"SELECT {_TICKET_COLUMNS} FROM support_tickets WHERE ticket_id = :ticket_id"),
                               {"ticket_id": ticket_id}).first()
        return _ticket(row) if row else None

    @contextmanager
    def ticket_writer(self, customer_id: int) -> Iterator[TicketWriter]:
        """One write transaction, serialized per customer. Commits if the block succeeds."""
        try:
            with self._engine.begin() as conn:
                conn.exec_driver_sql(f"SET LOCAL statement_timeout = {self._timeout_ms}")
                conn.execute(text("SELECT pg_advisory_xact_lock(:namespace, :customer_id)"),
                             {"namespace": _LOCK_NAMESPACE, "customer_id": customer_id})
                yield _PgTicketWriter(conn)
        except SQLAlchemyError as exc:
            raise self._unavailable(exc) from None

    def ping(self) -> bool:
        try:
            with self._read() as conn:
                conn.exec_driver_sql("SELECT 1").scalar_one()
            return True
        except DeskUnavailableError:
            return False

    def close(self) -> None:
        self._engine.dispose()
