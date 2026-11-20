"""A private PostgreSQL for a hosted demo (what space/start.sh does, in Python).

Hosts like Streamlit Community Cloud have no database, but they can install the `postgresql`
package (packages.txt). This starts a server on 127.0.0.1 with a fresh copy of the demo data
on every start, creates the insert-only desk login, and returns only that login's URL: the
admin password lives in this function and in the seed scripts' environment, nowhere else.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

from talktable.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

ADMIN = "talktable"  # the seed scripts' admin login (as in docker/dev-db.sh and space/start.sh)
DATA_DIR = Path("/tmp/callie-pgdata")


def postgres_bin() -> Path | None:
    """Debian's PostgreSQL programs live in /usr/lib/postgresql/<version>/bin."""
    versions = sorted(Path("/usr/lib/postgresql").glob("*/bin"), key=lambda p: int(p.parent.name)
                      if p.parent.name.isdigit() else 0)
    if versions:
        return versions[-1]
    initdb = shutil.which("initdb")
    return Path(initdb).parent if initdb else None


def _run(args: list[str], **kwargs: object) -> None:
    done = subprocess.run(args, capture_output=True, text=True, **kwargs)  # type: ignore[call-overload]
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip().splitlines()[-3:]
        raise RuntimeError(f"{Path(args[0]).name} {args[1] if len(args) > 1 else ''} failed: {' / '.join(detail)}")


def stop_demo_postgres(data_dir: Path = DATA_DIR) -> None:
    bin_dir = postgres_bin()
    if bin_dir is not None and (data_dir / "postmaster.pid").exists():
        subprocess.run([str(bin_dir / "pg_ctl"), "-D", str(data_dir), "-m", "fast", "stop"], capture_output=True)


def start_demo_postgres(data_dir: Path = DATA_DIR, port: int = 5432) -> str:
    """Start (or restart) the demo's Postgres, seed it, create the desk login; returns DESK_POSTGRES_URL."""
    bin_dir = postgres_bin()
    if bin_dir is None:
        raise RuntimeError("PostgreSQL isn't installed (on Streamlit Community Cloud: list `postgresql` "
                           "in packages.txt)")
    stop_demo_postgres(data_dir)  # left over from an earlier run of the app
    shutil.rmtree(data_dir, ignore_errors=True)
    admin_password, desk_password = secrets.token_urlsafe(24), secrets.token_urlsafe(24)

    password_file = data_dir.with_name(data_dir.name + ".pw")
    password_file.unlink(missing_ok=True)
    with os.fdopen(os.open(password_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
        handle.write(admin_password + "\n")  # private from the start
    try:
        _run([str(bin_dir / "initdb"), "-D", str(data_dir), "-U", ADMIN, f"--pwfile={password_file}",
              "--auth=scram-sha-256"])
    finally:
        password_file.unlink(missing_ok=True)
    options = (f"-c listen_addresses=127.0.0.1 -c port={port} -c unix_socket_directories={data_dir.parent} "
               "-c shared_buffers=16MB -c max_connections=40")
    _run([str(bin_dir / "pg_ctl"), "-D", str(data_dir), "-l", str(data_dir.parent / "callie-postgres.log"),
          "-w", "start", "-o", options])

    import psycopg

    admin = f"postgresql://{ADMIN}:{admin_password}@127.0.0.1:{port}"
    with psycopg.connect(f"{admin}/postgres", autocommit=True) as conn:
        conn.execute("CREATE DATABASE bookstore")
    env = os.environ | {"POSTGRES_SEED_URL": f"{admin}/bookstore", "TALKTABLE_DESK_PASSWORD": desk_password}
    _run([sys.executable, "-m", "data.seed.seed_postgres"], cwd=PROJECT_ROOT, env=env)
    _run([sys.executable, "-m", "data.seed.desk_role"], cwd=PROJECT_ROOT, env=env)
    logger.info("demo Postgres ready on 127.0.0.1:%d (fresh data, insert-only desk login)", port)
    return f"postgresql://talktable_desk:{desk_password}@127.0.0.1:{port}/bookstore"
