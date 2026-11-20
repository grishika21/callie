"""Callie settings, from environment variables and the same `.env` as TalkTable.

Secrets are `SecretStr` (as in talktable/config.py), so they never show up in logs or
reprs. Call `.get_secret_value()` only where the value is handed to a driver or library.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from talktable.config import PROJECT_ROOT

TOKEN_ISSUER = "callie"
MIN_SECRET_BYTES = 32
AgentProvider = Literal["ollama", "openai", "groq"]


class CallieSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # --- Support Desk MCP server -----------------------------------------------
    # The desk's own Postgres login (talktable_desk): reads 9 tables, inserts tickets.
    desk_postgres_url: SecretStr | None = None
    desk_host: str = "127.0.0.1"
    desk_port: int = Field(default=8765, ge=1, le=65535)
    desk_query_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    desk_tickets_per_hour: int = Field(default=20, ge=1)  # per staff member
    desk_duplicate_minutes: int = Field(default=30, ge=0)  # same open ticket within this window = duplicate
    # Phone callers (the customer line): verification tries and ticket limits.
    desk_verify_tries_per_call: int = Field(default=3, ge=1)
    desk_verify_failures_per_caller_hour: int = Field(default=6, ge=1)  # per caller number, across calls
    desk_phone_tickets_per_day: int = Field(default=3, ge=1)  # per customer

    # --- Staff login and tokens -------------------------------------------------
    # Signs the short-lived tokens the Callie app sends to the desk (HS256). Both
    # processes read it from here. Generate one with:
    #   python -c "import secrets; print(secrets.token_urlsafe(48))"
    callie_jwt_secret: SecretStr | None = None
    callie_token_ttl_seconds: int = Field(default=900, ge=30, le=3600)
    callie_users_path: Path = Path("data/callie_users.yaml")
    callie_audit_path: Path = Path("data/audit/callie.jsonl")
    # Where staff log in (the token issuer, advertised in the desk's OAuth metadata).
    callie_host: str = "127.0.0.1"
    callie_port: int = Field(default=7861, ge=1, le=65535)

    # --- Agent model ------------------------------------------------------------
    # Production is a local Ollama model. openai / groq are for testing only.
    # Measured in V0 (docs/callie_plan.md): the 4B instruct build picks the right tool
    # in 16/18 cases at ~1.4 s per turn with every layer on a 4 GB GPU; qwen3:14b gets
    # 18/18 at ~20 s. A spoken conversation needs the former.
    agent_provider: AgentProvider = "ollama"
    agent_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    agent_num_ctx: int = Field(default=4096, ge=1024)
    # Layers on the GPU. Unset = Ollama decides. On a 4 GB laptop GPU Ollama keeps 11 of
    # the 4B's 37 layers on the CPU (~3.4 s/turn); AGENT_NUM_GPU=37 fits all (~1.4 s).
    agent_num_gpu: int | None = Field(default=None, ge=0)
    agent_keep_alive: str = "30m"
    # A whole staff turn. TalkTable questions are the slow ones: 17-62 s for its pipeline,
    # more when its models and the agent's share a small GPU (108 s measured in V3).
    agent_timeout_seconds: float = Field(default=240.0, gt=0)
    # Per staff turn: model calls, tool calls, slow TalkTable questions, tickets.
    agent_max_model_calls: int = Field(default=6, ge=2)
    agent_max_tool_calls: int = Field(default=6, ge=1)
    agent_max_ask_calls: int = Field(default=2, ge=0)
    # Older turns are dropped from the prompt to stay inside the model's context.
    agent_history_tokens: int = Field(default=1800, ge=300)
    # Hosted reasoning models (Groq's gpt-oss, qwen3): "low" / "none" keeps replies fast and
    # short. Unset: the provider's default.
    agent_reasoning_effort: str | None = None
    agent_max_retries: int = Field(default=2, ge=0)  # rate limits (429) are retried with backoff
    # Models to try, in order, when the agent model fails (e.g. Groq's per-model rate limits):
    # comma-separated "name" or "name@reasoning-effort" on the same provider. Unset: none.
    agent_fallback_models: str = ""
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    groq_api_key: SecretStr | None = None
    # How Callie starts TalkTable's stdio MCP server (mcp_server.py, unchanged).
    talktable_python: str = sys.executable
    # The LLM for Callie's own TalkTable process (every step), or "default" for
    # TalkTable's .env (LLM_MODEL / LLM_QUERY_MODEL). Measured in V3 on a 4 GB GPU: with
    # qwen3:14b + qwen2.5-coder:7b a spoken TalkTable question took 124-140 s, because
    # those models and the agent's push each other out of VRAM; with the agent's own 4B
    # for everything, 8-26 s, and TalkTable's 4 known-answer live tests still pass.
    callie_talktable_llm: str = "qwen3:4b-instruct-2507-q4_K_M"

    # --- Phone line (Twilio Media Streams) -----------------------------------------
    # The gateway listens on 127.0.0.1; a Cloudflare tunnel gives Twilio its public URL.
    twilio_account_sid: str | None = None
    twilio_auth_token: SecretStr | None = None  # checks Twilio's signature on every request
    twilio_phone_number: str | None = None  # e.g. +15551234567 (only for the setup script)
    phone_public_url: str | None = None  # the tunnel's https URL; unset: taken from the request
    phone_host: str = "127.0.0.1"
    phone_port: int = Field(default=8770, ge=1, le=65535)
    phone_transfer_number: str | None = None  # "talk to a person"; unset: callers are offered a callback ticket
    phone_max_calls: int = Field(default=2, ge=1)  # at once (whisper runs on the CPU per call)
    phone_max_minutes: float = Field(default=10.0, gt=0)
    phone_calls_per_number_hour: int = Field(default=5, ge=1)
    phone_stt_model: str = "small.en"  # P0: small.en heard names and ids best on 8 kHz audio
    # "groq": Groq's hosted Whisper (fast on a CPU-only host such as the free HF Space), with
    # the local PHONE_STT_MODEL as the fallback when Groq is busy. "whisper": local only.
    phone_stt: Literal["whisper", "groq"] = "whisper"
    groq_stt_model: str = "whisper-large-v3-turbo"

    # --- Speech (both on the CPU) ---------------------------------------------------
    # small.en heard every id in the V0/V4 round trips at ~1.3 s per clip; base.en is
    # ~0.5 s but misses some. "browser" lets the page speak instead of Piper.
    callie_stt: Literal["whisper", "groq"] = "whisper"
    callie_stt_model: str = "small.en"
    callie_tts: Literal["piper", "browser", "none"] = "piper"
    callie_tts_voice: str = "en_US-lessac-medium"

    @field_validator("callie_jwt_secret")
    @classmethod
    def _long_enough(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value().encode()) < MIN_SECRET_BYTES:
            raise ValueError(f"CALLIE_JWT_SECRET must be at least {MIN_SECRET_BYTES} bytes")
        return value

    @property
    def desk_url(self) -> str:
        """The desk's MCP endpoint. Tokens are issued for exactly this URL (their `aud`)."""
        return f"http://{self.desk_host}:{self.desk_port}/mcp"

    @property
    def callie_url(self) -> str:
        return f"http://{self.callie_host}:{self.callie_port}"

    @property
    def users_file(self) -> Path:
        return _resolve(self.callie_users_path)

    @property
    def audit_file(self) -> Path:
        return _resolve(self.callie_audit_path)

    def jwt_secret(self) -> str:
        if self.callie_jwt_secret is None:
            raise SystemExit("CALLIE_JWT_SECRET is not set. Generate one with:\n"
                             "  python -c \"import secrets; print(secrets.token_urlsafe(48))\"\n"
                             "and put it in .env")
        return self.callie_jwt_secret.get_secret_value()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_callie_settings() -> CallieSettings:
    return CallieSettings()
