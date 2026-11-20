"""The Support Desk's own Postgres role, `talktable_desk`.

It may read the 9 sales tables the desk tools look at, and INSERT new support tickets,
naming only the columns below. It has no UPDATE, DELETE or TRUNCATE anywhere, can't set
a ticket's number, time, resolution or rating, and can't touch the sequence directly.
TalkTable's own role (`talktable_ro`) stays read-only and is not changed here.

Usage (from the project root), for an existing container or Neon:
    python -m data.seed.desk_role            create or update the role, then apply its grants
    python -m data.seed.desk_role --matrix   print what talktable_ro and talktable_desk may do

The password comes from TALKTABLE_DESK_PASSWORD (environment or .env). The connection URL
defaults to POSTGRES_SEED_URL: it must be the tables' owner and allowed to create roles.
A new Docker container gets the role from docker/postgres-init.sh; a re-seed drops the
tables (and their grants), so seed_postgres.py calls `apply_desk_grants` again.
"""

from __future__ import annotations

import argparse

from psycopg import sql
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Connection, create_engine, text

from talktable.config import PROJECT_ROOT, get_settings, mask_url
from data.seed.common import resolve_url

DESK_ROLE = "talktable_desk"
READ_ONLY_ROLE = "talktable_ro"
DESK_READ_TABLES = ("customers", "orders", "order_items", "payments", "shipments", "carriers", "returns",
                    "loyalty_accounts", "support_tickets")
# ticket_id, opened_at, resolved_at and satisfaction are left out on purpose: the database
# numbers and timestamps new tickets, and only people close or rate them.
DESK_INSERT_COLUMNS = ("customer_id", "order_id", "category", "subject", "description", "priority",
                       "channel", "created_by")
_ROLE_SETTINGS = {"statement_timeout": "5s", "idle_in_transaction_session_timeout": "10s"}
_PRIVILEGES = ("SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE")


class DeskRoleSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8",
                                      env_ignore_empty=True, extra="ignore")

    talktable_desk_password: SecretStr | None = None


def role_exists(conn: Connection, role: str = DESK_ROLE) -> bool:
    return conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :role"), {"role": role}).first() is not None


def apply_desk_grants(conn: Connection) -> None:
    """Reset the desk role's table grants to exactly the allowed set. Safe to repeat."""
    conn.execute(text(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {DESK_ROLE}"))
    conn.execute(text(f"REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM {DESK_ROLE}"))
    conn.execute(text(f"GRANT SELECT ON {', '.join(DESK_READ_TABLES)} TO {DESK_ROLE}"))
    conn.execute(text(f"GRANT INSERT ({', '.join(DESK_INSERT_COLUMNS)}) ON support_tickets TO {DESK_ROLE}"))


def setup_role(conn: Connection, password: str) -> list[str]:
    """Create (or update) the role, its database settings and grants. Returns what it did."""
    raw = conn.connection.dbapi_connection  # psycopg, to quote the password safely
    secret = sql.Literal(password).as_string(raw)
    verb = "ALTER" if role_exists(conn) else "CREATE"
    run = conn.execution_options(no_parameters=True).exec_driver_sql  # a '%' in the password is literal
    run(f"{verb} ROLE {DESK_ROLE} WITH LOGIN PASSWORD {secret}")
    done = [f"{verb.lower()}d role {DESK_ROLE}"]
    database = sql.Identifier(conn.execute(text("SELECT current_database()")).scalar_one()).as_string(raw)
    run(f"GRANT CONNECT ON DATABASE {database} TO {DESK_ROLE}")
    run(f"GRANT USAGE ON SCHEMA public TO {DESK_ROLE}")
    for name, value in _ROLE_SETTINGS.items():
        run(f"ALTER ROLE {DESK_ROLE} SET {name} = '{value}'")
    done.append("set " + ", ".join(f"{k}={v}" for k, v in _ROLE_SETTINGS.items()))
    present = set(conn.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = 'support_tickets'")).scalars())
    if not present:
        done.append("no tables yet: the seed script applies the grants")
    elif not present >= set(DESK_INSERT_COLUMNS):
        done.append("support_tickets has the old columns: re-seed it (python -m data.seed.seed_postgres), "
                    "which applies the grants")
    else:
        apply_desk_grants(conn)
        done.append(f"granted SELECT on {len(DESK_READ_TABLES)} tables, "
                    f"INSERT on {len(DESK_INSERT_COLUMNS)} columns of support_tickets")
    return done


def privilege_matrix(conn: Connection) -> str:
    """What each app role may do, per table, read from the system catalog.

    Letters: S select, I insert, U update, D delete, T truncate; lowercase = only on some
    columns; '-' = nothing.
    """
    roles = [r for r in (READ_ONLY_ROLE, DESK_ROLE) if role_exists(conn, r)]
    tables = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
                               "ORDER BY 1")).scalars().all()
    lines = [f"{'table':<18}" + "".join(f"{r:<16}" for r in roles)]
    for table in tables:
        cells = []
        for role in roles:
            letters = ""
            for privilege in _PRIVILEGES:
                args = {"role": role, "table": f"public.{table}", "privilege": privilege}
                if conn.execute(text("SELECT has_table_privilege(:role, :table, :privilege)"), args).scalar_one():
                    letters += privilege[0]
                elif privilege in ("SELECT", "INSERT", "UPDATE") and conn.execute(
                        text("SELECT has_any_column_privilege(:role, :table, :privilege)"), args).scalar_one():
                    letters += privilege[0].lower()
            cells.append(letters or "-")
        lines.append(f"{table:<18}" + "".join(f"{c:<16}" for c in cells))
    if DESK_ROLE in roles:
        columns = conn.execute(text(
            "SELECT column_name, has_column_privilege(:role, 'public.support_tickets', column_name, 'INSERT') "
            "FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'support_tickets' "
            "ORDER BY ordinal_position"), {"role": DESK_ROLE}).all()
        lines.append("")
        lines.append(f"{DESK_ROLE} may INSERT into support_tickets: " + ", ".join(c for c, ok in columns if ok))
        lines.append("  but not: " + ", ".join(c for c, ok in columns if not ok))
    for role, config in conn.execute(text("SELECT rolname, rolconfig FROM pg_roles WHERE rolname = ANY(:roles) "
                                          "ORDER BY rolname"), {"roles": roles}):
        lines.append(f"{role} settings: {', '.join(config or []) or 'none'}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="admin Postgres URL (default: POSTGRES_SEED_URL)")
    parser.add_argument("--matrix", action="store_true", help="only print the privilege matrix")
    args = parser.parse_args()

    settings = get_settings()
    url = resolve_url(args.url, settings.postgres_seed_url, None, "postgres")
    engine = create_engine(url)
    try:
        if not args.matrix:
            password = DeskRoleSettings().talktable_desk_password
            if password is None:
                raise SystemExit("[desk] set TALKTABLE_DESK_PASSWORD in .env or the environment first")
            print(f"[desk] setting up {DESK_ROLE} on {mask_url(url)}")
            with engine.begin() as conn:
                for line in setup_role(conn, password.get_secret_value()):
                    print(f"  {line}")
        with engine.connect() as conn:
            print(privilege_matrix(conn))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
