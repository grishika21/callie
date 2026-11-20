"""The Support Desk's Postgres role (`talktable_desk`), tried for real.

It may read 9 sales tables and INSERT new tickets into named support_tickets columns;
every other write must be refused by Postgres itself. `talktable_ro` must still be unable
to write. Each statement runs in a transaction that is rolled back, so the data is
unchanged afterwards. Skipped when DESK_POSTGRES_URL is unset or the role is missing.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any

import psycopg
import pytest
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from talktable.config import PROJECT_ROOT, Settings
from data.seed.desk_role import DESK_INSERT_COLUMNS, DESK_READ_TABLES
from data.seed.seed_postgres import metadata as postgres_metadata

SEEDED_TICKETS = 1450
NEW_TICKET = {"customer_id": 1, "order_id": None, "category": "delivery", "subject": "Probe",
              "description": "Inserted by tests/test_desk_role.py and rolled back.", "priority": "normal",
              "channel": "voice", "created_by": "pytest"}


class _DeskUrl(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_ignore_empty=True, extra="ignore")
    desk_postgres_url: SecretStr | None = None


def _connect(url: str) -> psycopg.Connection:
    try:
        return psycopg.connect(url, connect_timeout=3)
    except psycopg.OperationalError as exc:
        pytest.skip(f"cannot log in as the desk role ({type(exc).__name__}): "
                    "start Postgres (docker/dev-db.sh up) and run python -m data.seed.desk_role")


@pytest.fixture(scope="module")
def desk() -> Iterator[psycopg.Connection]:
    url = _DeskUrl().desk_postgres_url
    if url is None:
        pytest.skip("DESK_POSTGRES_URL is not set")
    conn = _connect(url.get_secret_value())
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def readonly(settings: Settings) -> Iterator[psycopg.Connection]:
    if settings.postgres_url is None:
        pytest.skip("POSTGRES_URL is not set")
    conn = _connect(settings.postgres_url.get_secret_value())
    yield conn
    conn.close()


def attempt(conn: psycopg.Connection, query: str, params: Any = None) -> list[tuple[Any, ...]]:
    """Run one statement in its own transaction and always roll it back."""
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ WRITE")
            cur.execute(query, params)
            return cur.fetchall() if cur.description else []
    finally:
        conn.rollback()


def insert_sql(row: dict[str, Any], returning: str = "ticket_id") -> str:
    columns = ", ".join(row)
    values = ", ".join(f"%({c})s" for c in row)
    return f"INSERT INTO support_tickets ({columns}) VALUES ({values}) RETURNING {returning}"


# --- what the desk role may do --------------------------------------------------


@pytest.mark.parametrize("table", DESK_READ_TABLES)
def test_desk_reads_its_tables(desk: psycopg.Connection, table: str) -> None:
    assert attempt(desk, f"SELECT count(*) FROM {table}")[0][0] > 0


def test_desk_inserts_a_ticket_with_database_defaults(desk: psycopg.Connection) -> None:
    rows = attempt(desk, insert_sql(NEW_TICKET, "ticket_id, opened_at, resolved_at, satisfaction, LOCALTIMESTAMP"),
                   NEW_TICKET)
    ticket_id, opened_at, resolved_at, satisfaction, now = rows[0]
    assert ticket_id > SEEDED_TICKETS  # the identity continues after the seeded numbers
    assert isinstance(opened_at, datetime) and abs(now - opened_at) < timedelta(seconds=2)
    assert opened_at.microsecond == 0  # seconds, like the seeded tickets
    assert (resolved_at, satisfaction) == (None, None)  # new tickets are open and unrated


def test_desk_insert_grant_covers_exactly_the_allowed_columns(desk: psycopg.Connection) -> None:
    rows = attempt(desk, "SELECT column_name FROM information_schema.column_privileges "
                         "WHERE table_name = 'support_tickets' AND privilege_type = 'INSERT' "
                         "AND grantee = current_user")
    assert {r[0] for r in rows} == set(DESK_INSERT_COLUMNS)


@pytest.mark.parametrize(("column", "value"), [("ticket_id", 999_999), ("opened_at", datetime(2020, 1, 1)),
                                                ("resolved_at", datetime(2026, 1, 1)), ("satisfaction", 5)])
def test_desk_cannot_set_protected_columns(desk: psycopg.Connection, column: str, value: Any) -> None:
    row = {**NEW_TICKET, column: value}
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        attempt(desk, insert_sql(row), row)


@pytest.mark.parametrize(("column", "value", "error"), [
    ("priority", "critical", psycopg.errors.CheckViolation),
    ("channel", "fax", psycopg.errors.CheckViolation),
    ("category", "spam", psycopg.errors.CheckViolation),
    ("customer_id", 999_999, psycopg.errors.ForeignKeyViolation),
    ("subject", "x" * 121, psycopg.errors.StringDataRightTruncation),
])
def test_database_rejects_bad_tickets(desk: psycopg.Connection, column: str, value: Any, error: type) -> None:
    row = {**NEW_TICKET, column: value}
    with pytest.raises(error):
        attempt(desk, insert_sql(row), row)


# --- what it may not do ---------------------------------------------------------


@pytest.mark.parametrize("sql", [
    "UPDATE support_tickets SET priority = 'low' WHERE ticket_id = 1",
    "UPDATE support_tickets SET resolved_at = now() WHERE ticket_id = 1",
    "DELETE FROM support_tickets WHERE ticket_id = 1",
    "TRUNCATE support_tickets",
    "ALTER TABLE support_tickets ADD COLUMN probe int",
    "DROP TABLE support_tickets",
])
def test_desk_cannot_change_or_remove_tickets(desk: psycopg.Connection, sql: str) -> None:
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        attempt(desk, sql)


@pytest.mark.parametrize("sql", [
    "INSERT INTO customers (customer_id, name, email, city, signup_date) "
    "VALUES (99999, 'x', 'x@example.com', 'x', '2025-01-01')",
    "UPDATE orders SET status = 'cancelled' WHERE order_id = 1",
    "DELETE FROM payments WHERE payment_id = 1",
    "INSERT INTO returns (return_id, order_item_id, reason, refund_amount, return_date) "
    "VALUES (99999, 1, 'damaged', 1, '2025-01-01')",
    "TRUNCATE orders",
    "CREATE TABLE desk_probe (x int)",
    "SELECT setval(pg_get_serial_sequence('support_tickets', 'ticket_id'), 1)",
    "SELECT nextval(pg_get_serial_sequence('support_tickets', 'ticket_id'))",
])
def test_desk_cannot_write_anything_else(desk: psycopg.Connection, sql: str) -> None:
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        attempt(desk, sql)


def test_desk_cannot_read_other_tables(desk: psycopg.Connection) -> None:
    others = sorted(set(postgres_metadata.tables) - set(DESK_READ_TABLES))
    assert others == ["coupons", "inventory", "order_coupons", "warehouses"]
    for table in others:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            attempt(desk, f"SELECT 1 FROM {table} LIMIT 1")


def test_desk_sessions_are_time_limited(desk: psycopg.Connection) -> None:
    assert attempt(desk, "SHOW statement_timeout") == [("5s",)]
    assert attempt(desk, "SHOW idle_in_transaction_session_timeout") == [("10s",)]
    with pytest.raises(psycopg.errors.QueryCanceled):
        attempt(desk, "SET LOCAL statement_timeout = '200ms'; SELECT pg_sleep(1)")


# --- TalkTable's role is unchanged ----------------------------------------------


def test_readonly_role_still_cannot_write_tickets(readonly: psycopg.Connection) -> None:
    for sql in (insert_sql(NEW_TICKET), "UPDATE support_tickets SET priority = 'low' WHERE ticket_id = 1",
                "DELETE FROM support_tickets WHERE ticket_id = 1"):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            attempt(readonly, sql, NEW_TICKET if sql.startswith("INSERT") else None)


def test_readonly_role_reads_the_new_columns(readonly: psycopg.Connection) -> None:
    rows = attempt(readonly, "SELECT count(*), count(DISTINCT channel) FROM support_tickets "
                             "WHERE subject <> '' AND priority IS NOT NULL AND created_by IS NULL")
    assert rows == [(SEEDED_TICKETS, 3)]  # the seeded tickets; Callie's tickets have created_by set
