"""Staff accounts and desk tokens (callie/auth.py). No server or database needed."""

from __future__ import annotations

import base64
import json
import stat
import time
from pathlib import Path

import jwt
import pytest

from callie import auth
from callie.auth import JWTVerifier, TokenIssuer, UserStore, hash_password, verify_password
from callie.config import CallieSettings

SECRET = "test-secret-" + "x" * 40
AUDIENCE = "http://127.0.0.1:8765/mcp"
PASSWORD = "correct horse battery"


@pytest.fixture
def users(tmp_path: Path) -> UserStore:
    store = UserStore(tmp_path / "users.yaml")
    store.add("alice", PASSWORD, "agent", "Alice")
    store.add("victor", PASSWORD, "viewer")
    return store


def test_password_hashing() -> None:
    stored = hash_password(PASSWORD)
    assert stored.startswith("scrypt$16384$8$1$") and PASSWORD not in stored
    assert verify_password(PASSWORD, stored)
    assert not verify_password("wrong password!", stored)
    assert hash_password(PASSWORD) != stored  # salted
    for broken in ("", "plain", "md5$1$2$3$4$5", "scrypt$16384$8$1$!!$!!"):
        assert not verify_password(PASSWORD, broken)


def test_users_file(users: UserStore) -> None:
    assert stat.S_IMODE(users.path.stat().st_mode) == 0o600
    text = users.path.read_text()
    assert PASSWORD not in text and "scrypt$" in text
    alice = users.get("alice")
    assert alice is not None and alice.role == "agent" and alice.display_name == "Alice"
    assert alice.scopes == {"data:read", "desk:read", "tickets:write"}
    assert users.get("victor").scopes == {"data:read", "desk:read"}  # type: ignore[union-attr]
    assert [u.username for u in users.users()] == ["alice", "victor"]


def test_authenticate(users: UserStore) -> None:
    assert users.authenticate("alice", PASSWORD).username == "alice"  # type: ignore[union-attr]
    assert users.authenticate(" Alice ", PASSWORD) is not None
    assert users.authenticate("alice", "wrong password!") is None
    assert users.authenticate("nobody", PASSWORD) is None
    assert users.remove("victor") and users.authenticate("victor", PASSWORD) is None


def test_disabled_account_cannot_log_in(users: UserStore) -> None:
    users.path.write_text(users.path.read_text().replace("disabled: false", "disabled: true", 1))
    assert users.authenticate("alice", PASSWORD) is None


@pytest.mark.parametrize(("username", "password", "role"), [
    ("Alice", PASSWORD, "agent"), ("a", PASSWORD, "agent"), ("bob smith", PASSWORD, "agent"),
    ("bob", "short", "agent"), ("bob", PASSWORD, "admin"),
])
def test_add_user_validation(users: UserStore, username: str, password: str, role: str) -> None:
    with pytest.raises(ValueError):
        users.add(username, password, role)  # type: ignore[arg-type]


# --- tokens -------------------------------------------------------------------------


def _verifier(users: UserStore | None) -> JWTVerifier:
    return JWTVerifier(SECRET, AUDIENCE, users)


def _claims(token: str) -> dict:
    return jwt.decode(token, options={"verify_signature": False})


def test_token_round_trip(users: UserStore) -> None:
    token = TokenIssuer(SECRET, AUDIENCE, 900).mint(users.get("alice"))  # type: ignore[arg-type]
    claims = _claims(token)
    assert claims["sub"] == "alice" and claims["aud"] == AUDIENCE and claims["iss"] == "callie"
    assert claims["exp"] - claims["iat"] == 900 and len(claims["jti"]) == 32
    access = _verifier(users).verify(token)
    assert access is not None
    assert access.subject == "alice" and access.resource == AUDIENCE
    assert access.scopes == ["data:read", "desk:read", "tickets:write"]


def _forge(**overrides: object) -> str:
    now = int(time.time())
    claims = {"iss": "callie", "aud": AUDIENCE, "sub": "alice", "role": "agent",
              "scope": "data:read desk:read tickets:write", "iat": now, "exp": now + 600}
    claims.update(overrides)
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, SECRET, algorithm="HS256")


@pytest.mark.parametrize("token", [
    "", "not-a-token", "a.b.c",
    _forge(exp=int(time.time()) - 60),                      # expired
    _forge(aud="http://127.0.0.1:9999/mcp"),                 # issued for another server
    _forge(iss="someone-else"),                              # another issuer
    _forge(exp=None),                                        # no expiry
    _forge(sub=None),                                        # nobody
    jwt.encode({"sub": "alice", "aud": AUDIENCE, "iss": "callie", "exp": int(time.time()) + 60,
                "iat": int(time.time())}, "another-secret-" + "y" * 40, algorithm="HS256"),  # wrong key
])
def test_bad_tokens_are_refused(users: UserStore, token: str) -> None:
    assert _verifier(users).verify(token) is None


def test_tampered_and_unsigned_tokens_are_refused(users: UserStore) -> None:
    header, payload, signature = _forge(scope="desk:read").split(".")
    claims = json.loads(base64.urlsafe_b64decode(payload + "=="))
    claims["scope"] = "data:read desk:read tickets:write"
    raised = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    assert _verifier(users).verify(f"{header}.{raised}.{signature}") is None
    none_header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').rstrip(b"=").decode()
    assert _verifier(users).verify(f"{none_header}.{payload}.") is None


def test_account_changes_apply_to_live_tokens(users: UserStore) -> None:
    token = _forge()
    assert _verifier(users).verify(token) is not None
    users.add("alice", PASSWORD, "viewer")  # demoted: the old token loses tickets:write at once
    assert _verifier(users).verify(token).scopes == ["data:read", "desk:read"]  # type: ignore[union-attr]
    users.remove("alice")  # removed: the old token stops working at once
    assert _verifier(users).verify(token) is None


def test_scopes_never_exceed_the_role(users: UserStore) -> None:
    token = _forge(sub="victor", role="agent", scope="data:read desk:read tickets:write admin")
    assert _verifier(users).verify(token).scopes == ["data:read", "desk:read"]  # type: ignore[union-attr]


def test_short_jwt_secret_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 32 bytes"):
        CallieSettings(_env_file=None, callie_jwt_secret="too-short")


def test_cli_add_user_and_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                capsys: pytest.CaptureFixture[str]) -> None:
    settings = CallieSettings(_env_file=None, callie_jwt_secret=SECRET, callie_users_path=tmp_path / "u.yaml")
    monkeypatch.setattr(auth, "get_callie_settings", lambda: settings)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(PASSWORD + "\n"))
    auth.main(["add-user", "carol", "--role", "viewer", "--password-stdin"])
    assert "saved carol (viewer: data:read, desk:read)" in capsys.readouterr().out
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(PASSWORD + "\n"))
    auth.main(["token", "carol", "--password-stdin"])
    token = capsys.readouterr().out.strip()
    assert JWTVerifier(SECRET, settings.desk_url, UserStore(settings.users_file)).verify(token) is not None
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("wrong password!\n"))
    with pytest.raises(SystemExit, match="Wrong username or password"):
        auth.main(["token", "carol", "--password-stdin"])
