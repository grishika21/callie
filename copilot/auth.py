"""Staff accounts, roles and the short-lived tokens the Copilot sends to the Support Desk.

* Accounts live in a small YAML file (data/copilot_users.yaml, git-ignored, mode 0600).
  Passwords are hashed with scrypt from the standard library; nothing else is stored.
* A role maps to scopes. The desk checks the scope of every tool call itself, so a
  confused or prompt-injected model still can't do more than the logged-in person.
* Tokens are HS256 JWTs with a 15-minute lifetime, issued for one audience (the desk's
  URL). The verifier also re-reads the users file, so removing or disabling an account,
  or lowering its role, takes effect on the next request rather than when tokens expire.

Command line (from the project root):
    python -m copilot.auth add-user alice --role agent     (asks for the password)
    python -m copilot.auth list-users
    python -m copilot.auth remove-user alice
    python -m copilot.auth token alice                     (a token for MCP Inspector)
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import os
import re
import secrets
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jwt
import yaml
from mcp.server.auth.provider import AccessToken

from copilot.config import TOKEN_ISSUER, CopilotSettings, get_copilot_settings

Role = Literal["viewer", "agent", "phone"]
ROLES: dict[str, frozenset[str]] = {
    "viewer": frozenset({"data:read", "desk:read"}),
    "agent": frozenset({"data:read", "desk:read", "tickets:write"}),
    # The phone gateway (a service account, never a person): it may verify callers, then
    # act for one verified customer at a time. No desk:read, no tickets:write, no data:read.
    "phone": frozenset({"desk:verify", "desk:self", "tickets:self"}),
}
# Claims a token may carry besides the standard ones. Only the phone gateway sets them.
CALL_CLAIMS = ("customer_id", "call", "caller")
CLIENT_ID = TOKEN_ISSUER
USERNAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{1,31}$")
MIN_PASSWORD_CHARS = 10

# scrypt cost: 2**14 * 8 * 128 bytes = 16 MiB of memory per hash, ~50 ms.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P, _SCRYPT_LEN = 2**14, 8, 1, 32
_LEEWAY_SECONDS = 5


# --- passwords ---------------------------------------------------------------


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_LEN)
    b64 = base64.b64encode
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${b64(salt).decode()}${b64(digest).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest, validate=True)
        actual = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt, validate=True),
                                n=int(n), r=int(r), p=int(p), dklen=len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


# Checked against when the username is unknown, so a wrong name takes as long as a wrong password.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


# --- accounts ------------------------------------------------------------------


@dataclass(frozen=True)
class StaffUser:
    username: str
    role: Role
    display_name: str
    disabled: bool = False

    @property
    def scopes(self) -> frozenset[str]:
        return ROLES[self.role]


class UserStore:
    """The users file, re-read whenever it changes on disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._stamp: tuple[int, int] | None = None
        self._records: dict[str, dict[str, Any]] = {}

    def _load(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            try:
                stat = self.path.stat()
            except FileNotFoundError:
                self._stamp, self._records = None, {}
                return self._records
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp != self._stamp:
                raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
                self._records = {str(k): dict(v) for k, v in (raw.get("users") or {}).items()}
                self._stamp = stamp
            return self._records

    def get(self, username: str) -> StaffUser | None:
        record = self._load().get(username)
        if record is None or record.get("role") not in ROLES:
            return None
        return StaffUser(username=username, role=record["role"],
                         display_name=str(record.get("display_name") or username),
                         disabled=bool(record.get("disabled", False)))

    def users(self) -> list[StaffUser]:
        return [u for name in sorted(self._load()) if (u := self.get(name)) is not None]

    def authenticate(self, username: str, password: str) -> StaffUser | None:
        """The user, if the password is right and the account is enabled."""
        record = self._load().get(username.strip().lower())
        stored = str(record.get("password", "")) if record else _DUMMY_HASH
        ok = verify_password(password, stored)
        user = self.get(username.strip().lower()) if record else None
        return user if ok and user is not None and not user.disabled else None

    def add(self, username: str, password: str, role: Role, display_name: str | None = None) -> StaffUser:
        if not USERNAME_RE.match(username):
            raise ValueError("username: 2-32 characters, lowercase letters, digits, '.', '_' or '-', "
                             "starting with a letter")
        if role not in ROLES:
            raise ValueError(f"role must be one of: {', '.join(ROLES)}")
        if len(password) < MIN_PASSWORD_CHARS:
            raise ValueError(f"password must be at least {MIN_PASSWORD_CHARS} characters")
        records = dict(self._load())
        records[username] = {"role": role, "display_name": display_name or username.capitalize(),
                             "password": hash_password(password), "disabled": False}
        self._save(records)
        user = self.get(username)
        assert user is not None
        return user

    def ensure_service(self, username: str, role: Role) -> StaffUser:
        """A service account (e.g. the phone gateway): it never logs in, so its password is
        random and never shown. Tokens for it are minted with the JWT secret."""
        user = self.get(username)
        if user is not None and user.role == role:
            return user
        return self.add(username, secrets.token_urlsafe(32), role, username.replace("-", " ").title())

    def remove(self, username: str) -> bool:
        records = dict(self._load())
        if records.pop(username, None) is None:
            return False
        self._save(records)
        return True

    def _save(self, records: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        text = ("# TalkTable Copilot staff accounts. Managed by: python -m copilot.auth\n"
                + yaml.safe_dump({"users": records}, sort_keys=True))
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, self.path)
        os.chmod(self.path, 0o600)


# --- tokens --------------------------------------------------------------------


class TokenIssuer:
    """Mints the short-lived token sent with every request to the Support Desk."""

    def __init__(self, secret: str, audience: str, ttl_seconds: int = 900) -> None:
        self._secret, self.audience, self.ttl_seconds = secret, audience, ttl_seconds

    def mint(self, user: StaffUser, *, now: float | None = None, call: dict[str, Any] | None = None) -> str:
        """`call` (phone gateway only): customer_id (once verified), call (id), caller (number hash)."""
        issued = int(now if now is not None else time.time())
        claims: dict[str, Any] = {"iss": TOKEN_ISSUER, "aud": self.audience, "sub": user.username, "role": user.role,
                                  "scope": " ".join(sorted(user.scopes)), "iat": issued, "nbf": issued,
                                  "exp": issued + self.ttl_seconds, "jti": uuid.uuid4().hex}
        claims |= {k: v for k, v in (call or {}).items() if k in CALL_CLAIMS and v is not None}
        return jwt.encode(claims, self._secret, algorithm="HS256")


class JWTVerifier:
    """`mcp.server.auth.provider.TokenVerifier` for the desk: signature, expiry, issuer and
    audience, then the account must still exist and be enabled. The scopes granted are
    the ones in the token that the account's *current* role still allows."""

    def __init__(self, secret: str, audience: str, users: UserStore | None) -> None:
        self._secret, self.audience, self.users = secret, audience, users

    def verify(self, token: str) -> AccessToken | None:
        try:
            claims = jwt.decode(token, self._secret, algorithms=["HS256"], audience=self.audience,
                                issuer=TOKEN_ISSUER, leeway=_LEEWAY_SECONDS,
                                options={"require": ["exp", "iat", "iss", "aud", "sub"]})
        except jwt.InvalidTokenError:
            return None
        username, scopes = str(claims["sub"]), set(str(claims.get("scope", "")).split())
        if self.users is not None:
            user = self.users.get(username)
            if user is None or user.disabled:
                return None
            scopes &= user.scopes
        extra = {k: claims[k] for k in CALL_CLAIMS if k in claims}
        if "customer_id" in extra and (not isinstance(extra["customer_id"], int) or isinstance(extra["customer_id"], bool)):
            return None
        return AccessToken(token=token, client_id=CLIENT_ID, scopes=sorted(scopes), expires_at=int(claims["exp"]),
                           resource=self.audience, subject=username,
                           claims={"iss": claims["iss"], "role": claims.get("role"), "jti": claims.get("jti"), **extra})

    async def verify_token(self, token: str) -> AccessToken | None:
        return self.verify(token)


def issuer_for(settings: CopilotSettings) -> TokenIssuer:
    return TokenIssuer(settings.jwt_secret(), settings.desk_url, settings.copilot_token_ttl_seconds)


def verifier_for(settings: CopilotSettings, users: UserStore | None = None) -> JWTVerifier:
    return JWTVerifier(settings.jwt_secret(), settings.desk_url, users or UserStore(settings.users_file))


# --- command line --------------------------------------------------------------


def _read_password(args: argparse.Namespace, confirm: bool) -> str:
    if args.password_stdin:
        return sys.stdin.readline().rstrip("\n")
    password = getpass.getpass("Password: ")
    if confirm and getpass.getpass("Again: ") != password:
        raise SystemExit("The passwords don't match.")
    return password


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add-user", help="add a staff account, or reset its password and role")
    add.add_argument("username")
    add.add_argument("--role", choices=sorted(ROLES), default="agent")
    add.add_argument("--display-name")
    add.add_argument("--password-stdin", action="store_true", help="read the password from stdin")
    sub.add_parser("list-users")
    remove = sub.add_parser("remove-user")
    remove.add_argument("username")
    token = sub.add_parser("token", help="print a desk token (for MCP Inspector / curl tests)")
    token.add_argument("username")
    token.add_argument("--password-stdin", action="store_true")
    args = parser.parse_args(argv)

    settings = get_copilot_settings()
    store = UserStore(settings.users_file)
    if args.command == "add-user":
        try:
            user = store.add(args.username, _read_password(args, confirm=True), args.role, args.display_name)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        print(f"saved {user.username} ({user.role}: {', '.join(sorted(user.scopes))}) in {store.path}")
    elif args.command == "list-users":
        for user in store.users():
            state = " (disabled)" if user.disabled else ""
            print(f"{user.username:<20} {user.role:<8} {user.display_name}{state}")
    elif args.command == "remove-user":
        print("removed" if store.remove(args.username) else f"no user {args.username!r}")
    elif args.command == "token":
        user = store.authenticate(args.username, _read_password(args, confirm=False))
        if user is None:
            raise SystemExit("Wrong username or password, or the account is disabled.")
        issuer = issuer_for(settings)
        print(issuer.mint(user))
        print(f"(valid {issuer.ttl_seconds // 60} min, audience {issuer.audience})", file=sys.stderr)


if __name__ == "__main__":
    main()
