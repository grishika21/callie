"""The Postgres Support Desk store, live, logged in as the desk role (talktable_desk).

Tickets created here have created_by starting with "pytest" and are deleted afterwards
with the admin login (the desk role itself can't delete). Skipped when Postgres, the
desk role or the admin URL is unavailable.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

import psycopg
import pytest

from callie.config import CallieSettings
from callie.desk.server import Caller, CreatedTicket, DeskService
from callie.desk.store import DeskUnavailableError, NewTicket, PostgresDeskStore
from talktable.config import Settings

AGENT = Caller("pytest-desk", "agent", frozenset({"desk:read", "tickets:write"}))


@pytest.fixture(scope="module")
def admin_url(settings: Settings) -> str:
    if settings.postgres_seed_url is None:
        pytest.skip("POSTGRES_SEED_URL is not set (needed to clean up test tickets)")
    return settings.postgres_seed_url.get_secret_value()


@pytest.fixture(scope="module")
def store(admin_url: str) -> Iterator[PostgresDeskStore]:
    url = CallieSettings().desk_postgres_url
    if url is None:
        pytest.skip("DESK_POSTGRES_URL is not set")
    desk = PostgresDeskStore(url.get_secret_value(), connect_timeout_s=3)
    if not desk.ping():
        pytest.skip("cannot log in as talktable_desk: docker/dev-db.sh up, then python -m data.seed.desk_role")
    yield desk
    desk.close()


@pytest.fixture(autouse=True)
def cleanup(request: pytest.FixtureRequest) -> Iterator[None]:
    yield
    if "store" in request.fixturenames:
        with psycopg.connect(request.getfixturevalue("admin_url"), connect_timeout=3) as conn:
            conn.execute("DELETE FROM support_tickets WHERE created_by LIKE 'pytest%'")


def test_find_customers(store: PostgresDeskStore) -> None:
    one = store.find_customers("1", 5).matches
    assert [c.customer_id for c in one] == [1]
    by_email = store.find_customers(one[0].email.upper(), 5).matches
    assert [c.customer_id for c in by_email] == [1]
    first, last = one[0].name.split()[0], one[0].name.split()[-1]
    by_name = store.find_customers(f"{last.lower()} {first}", 5).matches  # words in any order, any case
    assert 1 in [c.customer_id for c in by_name]
    many = store.find_customers("a", 5)
    assert len(many.matches) == 5 and many.more
    assert store.find_customers("%", 5).matches == [] and store.find_customers("_", 5).matches == []  # escaped
    assert store.find_customers("nobody-at-all@example.com", 5).matches == []


def test_get_order_with_shipment_and_returns(store: PostgresDeskStore) -> None:
    with store._read() as conn:  # an order that was shipped and partly returned
        order_id = conn.exec_driver_sql(
            "SELECT i.order_id FROM returns r JOIN order_items i ON i.order_item_id = r.order_item_id "
            "ORDER BY i.order_id LIMIT 1").scalar_one()
    order = store.get_order(order_id)
    assert order is not None and order.status == "returned"
    assert order.carrier and order.shipped_date and order.payment_status == "refunded"
    assert order.returned_items >= 1 and order.refunded > 0 and order.return_reasons
    assert order.items >= 1 and order.copies >= order.items
    assert store.get_order(999_999) is None


def test_tickets(store: PostgresDeskStore) -> None:
    ticket = store.get_ticket(1)
    assert ticket is not None and ticket.subject and ticket.description and ticket.created_by is None
    listed = store.list_tickets(customer_id=ticket.customer_id, order_id=None, open_only=False, limit=10)
    assert 1 in [t.ticket_id for t in listed.tickets]
    open_only = store.list_tickets(customer_id=None, order_id=None, open_only=True, limit=10)
    assert open_only.tickets and all(t.status == "open" for t in open_only.tickets)
    assert store.get_ticket(999_999) is None


def test_reads_are_read_only(store: PostgresDeskStore) -> None:
    with pytest.raises(DeskUnavailableError), store._read() as conn:
        conn.exec_driver_sql("INSERT INTO support_tickets (customer_id, category, subject, description, priority, "
                             "channel) VALUES (1, 'other', 'x', 'read-only check', 'low', 'voice')")


def test_writer_inserts_and_finds_duplicates(store: PostgresDeskStore) -> None:
    new = NewTicket(customer_id=1, order_id=None, category="account", priority="normal", subject="Probe",
                    description="Created by tests/test_desk_store.py.", channel="voice", created_by="pytest-desk")
    with store.ticket_writer(1) as writer:
        ticket = writer.insert(new)
    assert ticket.ticket_id > 1450 and ticket.status == "open" and ticket.created_by == "pytest-desk"
    assert ticket.opened_at.microsecond == 0
    with store.ticket_writer(1) as writer:
        assert writer.open_duplicate(1, None, "account", 30).ticket_id == ticket.ticket_id  # type: ignore[union-attr]
        assert writer.open_duplicate(1, None, "refund", 30) is None
        assert writer.open_duplicate(2, None, "account", 30) is None
        assert writer.tickets_by_user_since("pytest-desk", 60) == 1
        assert writer.tickets_by_user_since("someone-else", 60) == 0


def test_simultaneous_requests_create_one_ticket(store: PostgresDeskStore) -> None:
    """Two 'yes' clicks at once: the advisory lock lets only one of them write."""
    service = DeskService(store, CallieSettings(_env_file=None))
    order = store.get_order(1)
    assert order is not None
    start, results = threading.Barrier(4), []

    def create() -> None:
        start.wait()
        results.append(service.create_ticket(AGENT, order.customer_id, order.order_id, "delivery", "high",
                                             "Where is my order?", "The customer asks where order 1 is."))

    threads = [threading.Thread(target=create) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert len(results) == 4 and all(isinstance(r, CreatedTicket) for r in results)
    assert sorted(r.duplicate for r in results) == [False, True, True, True]
    assert len({r.ticket.ticket_id for r in results}) == 1


def test_phone_queries(store: PostgresDeskStore) -> None:
    order = store.get_order(5512)
    assert order is not None
    mine = store.orders_of(order.customer_id, 10)
    assert 5512 in [o.order_id for o in mine.orders]
    assert [o.order_date for o in mine.orders] == sorted((o.order_date for o in mine.orders), reverse=True)
    new = NewTicket(customer_id=order.customer_id, order_id=5512, category="delivery", priority="normal",
                    subject="Probe", description="Created by tests/test_desk_store.py.", channel="phone",
                    created_by="pytest-phone")
    with store.ticket_writer(order.customer_id) as writer:
        writer.insert(new)
    with store.ticket_writer(order.customer_id) as writer:
        assert writer.tickets_for_customer_by_since(order.customer_id, "pytest-phone", 1440) == 1
        assert writer.tickets_for_customer_by_since(order.customer_id + 1, "pytest-phone", 1440) == 0
