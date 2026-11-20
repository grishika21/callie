#!/bin/sh
# Runs once, when the Postgres container is first created.
# Creates the read-only login `talktable_ro` that the TalkTable app connects with, and,
# when TALKTABLE_DESK_PASSWORD is set, the Copilot Support Desk's login `talktable_desk`.
set -eu

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE ROLE talktable_ro LOGIN PASSWORD '${TALKTABLE_RO_PASSWORD}';

    GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO talktable_ro;
    GRANT USAGE ON SCHEMA public TO talktable_ro;
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO talktable_ro;

    -- Tables the seed script creates later (owned by $POSTGRES_USER) are readable too.
    ALTER DEFAULT PRIVILEGES FOR ROLE ${POSTGRES_USER} IN SCHEMA public
        GRANT SELECT ON TABLES TO talktable_ro;

    -- Defence in depth: every session for this role is read-only and time-limited.
    ALTER ROLE talktable_ro SET default_transaction_read_only = on;
    ALTER ROLE talktable_ro SET statement_timeout = '5s';
EOSQL

# The Support Desk role gets no table rights here: the tables don't exist yet. The seed
# script grants SELECT on 9 tables and INSERT on named support_tickets columns after it
# creates them (data/seed/desk_role.py). No UPDATE, DELETE or TRUNCATE, ever.
if [ -n "${TALKTABLE_DESK_PASSWORD:-}" ]; then
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
         -v desk_password="$TALKTABLE_DESK_PASSWORD" <<-'EOSQL'
        CREATE ROLE talktable_desk LOGIN PASSWORD :'desk_password';
        SELECT format('GRANT CONNECT ON DATABASE %I TO talktable_desk', current_database()) \gexec
        GRANT USAGE ON SCHEMA public TO talktable_desk;
        ALTER ROLE talktable_desk SET statement_timeout = '5s';
        ALTER ROLE talktable_desk SET idle_in_transaction_session_timeout = '10s';
EOSQL
fi
