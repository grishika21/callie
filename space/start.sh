#!/usr/bin/env bash
# The Space's start: its own Postgres with fresh demo data (every restart resets the demo),
# the insert-only desk login, then the app. The app never gets the Postgres admin login.
set -euo pipefail
cd /app
PG_BIN="$(ls -d /usr/lib/postgresql/*/bin | head -1)"
export PGDATA=/tmp/pgdata
secret() { python -c 'import secrets; print(secrets.token_urlsafe(24))'; }
ADMIN_PASSWORD="$(secret)"
DESK_PASSWORD="$(secret)"

rm -rf "$PGDATA"
printf '%s\n' "$ADMIN_PASSWORD" > /tmp/pg-admin-password
"$PG_BIN/initdb" -D "$PGDATA" -U talktable --pwfile=/tmp/pg-admin-password --auth=scram-sha-256 >/dev/null
rm -f /tmp/pg-admin-password
"$PG_BIN/pg_ctl" -D "$PGDATA" -l /tmp/postgres.log -w start \
  -o "-c listen_addresses=127.0.0.1 -c port=5432 -c unix_socket_directories=/tmp"
PGPASSWORD="$ADMIN_PASSWORD" "$PG_BIN/psql" -h 127.0.0.1 -U talktable -d postgres -qc "CREATE DATABASE bookstore"

export POSTGRES_SEED_URL="postgresql://talktable:${ADMIN_PASSWORD}@127.0.0.1:5432/bookstore"
python -m data.seed.seed_postgres
TALKTABLE_DESK_PASSWORD="$DESK_PASSWORD" python -m data.seed.desk_role >/dev/null
unset POSTGRES_SEED_URL ADMIN_PASSWORD

export DESK_POSTGRES_URL="postgresql://talktable_desk:${DESK_PASSWORD}@127.0.0.1:5432/bookstore"
export COPILOT_JWT_SECRET="${COPILOT_JWT_SECRET:-$(secret)$(secret)}"
exec python -m copilot.phone.space
