"""TalkTable Support Desk: Callie's second MCP server (lookups + opening tickets).

    python support_desk_server.py            serves http://127.0.0.1:8765/mcp

Every request needs a bearer token from Callie (or, for testing,
`python -m callie.auth token <user>`). It listens on 127.0.0.1 only by default
(DESK_HOST / DESK_PORT). TalkTable's own MCP server (mcp_server.py) stays read-only;
this server is the only place a ticket can be written, through its own insert-only
Postgres login. Details: docs/callie_plan.md.
"""

from __future__ import annotations

import argparse
import logging

from callie.config import get_callie_settings
from callie.desk.server import create_server
from talktable.config import mask_url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", help="interface to listen on (default: DESK_HOST, 127.0.0.1)")
    parser.add_argument("--port", type=int, help="port (default: DESK_PORT, 8765)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_callie_settings()
    overrides = {k: v for k, v in {"desk_host": args.host, "desk_port": args.port}.items() if v is not None}
    if overrides:
        settings = settings.model_copy(update=overrides)
    server = create_server(settings)
    db = settings.desk_postgres_url.get_secret_value() if settings.desk_postgres_url else ""
    logging.getLogger("support_desk").info("Support Desk on %s (database %s, users %s)", settings.desk_url,
                                           mask_url(db), settings.users_file)
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
