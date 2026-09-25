"""Gateway configuration: a YAML file with ``${VAR}`` / ``${VAR:-default}`` env interpolation.

The YAML file is the single source of truth for topology (providers, routes, limits). Secrets and
deployment-specific values are injected through environment variables referenced from the YAML,
so the same file works locally, in Docker Compose and in CI.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

DEFAULT_CONFIG_PATH = "config/gateway.yaml"
_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


class ConfigError(ValueError):
    """Raised when the gateway configuration is invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Timeouts(_Strict):
    """Per-provider timeouts, all in seconds.

    ``first_byte_s`` bounds the wait for response headers / the first streamed chunk,
    ``idle_s`` bounds the gap between two streamed chunks, and ``total_s`` bounds a whole
    non-streaming call. A stream that keeps producing tokens is never cut off by ``total_s``.
    """

    connect_s: float = Field(default=2.0, gt=0)
    first_byte_s: float = Field(default=30.0, gt=0)
    idle_s: float = Field(default=15.0, gt=0)
    total_s: float = Field(default=60.0, gt=0)


ProviderType = Literal["groq", "ollama", "mock", "openai"]


class ProviderConfig(_Strict):
    type: ProviderType
    base_url: str
    api_key: SecretStr | None = None
    enabled: bool = True
    timeouts: Timeouts = Timeouts()
    max_connections: int = Field(default=512, ge=1)

    @field_validator("base_url")
    @classmethod
    def _strip_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("api_key", mode="before")
    @classmethod
    def _empty_key_is_none(cls, value: Any) -> Any:
        return None if value in ("", None) else value


class RouteTarget(_Strict):
    provider: str
    model: str


class Route(_Strict):
    """A public model name and the ordered list of provider targets that can serve it."""

    model: str
    targets: list[RouteTarget] = Field(min_length=1)


class RetryConfig(_Strict):
    max_attempts: int = Field(default=2, ge=1, le=10)
    base_delay_s: float = Field(default=0.05, ge=0)
    max_delay_s: float = Field(default=1.0, ge=0)


class CircuitBreakerConfig(_Strict):
    window_size: int = Field(default=20, ge=1)
    min_calls: int = Field(default=10, ge=1)
    failure_rate_threshold: float = Field(default=0.5, gt=0, le=1)
    open_s: float = Field(default=10.0, gt=0)
    half_open_max_calls: int = Field(default=1, ge=1)


class ReliabilityConfig(_Strict):
    request_timeout_s: float = Field(default=60.0, gt=0)
    """End-to-end budget across all retries and failovers, up to the first streamed byte."""
    retry: RetryConfig = RetryConfig()
    circuit_breaker: CircuitBreakerConfig = CircuitBreakerConfig()


class RedisConfig(_Strict):
    url: str = "redis://localhost:6379/0"
    key_prefix: str = "sy"
    socket_timeout_s: float = Field(default=0.5, gt=0)
    max_connections: int = Field(default=256, ge=1)


class AuthConfig(_Strict):
    enabled: bool = True
    """When false, every request is anonymous and unlimited: for local experiments only."""
    default_max_tokens: int = Field(default=256, ge=1)
    """Output tokens assumed for rate limiting when the caller does not send ``max_tokens``."""


class ExactCacheConfig(_Strict):
    enabled: bool = True
    ttl_s: int = Field(default=3600, ge=1)


class SemanticCacheConfig(_Strict):
    """Two-stage semantic matching. Thresholds come from scripts/eval_semantic_threshold.py
    (results/semantic-threshold/, ADR-016); change them only together with a new evaluation."""

    enabled: bool = False
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    candidate_threshold: float = Field(default=0.80, gt=0, le=1)
    """Cosine similarity for the nearest cached prompt to become a *candidate*."""
    verifier_model: str | None = "cross-encoder/quora-distilroberta-base"
    """Cross-encoder that must confirm a candidate. ``None`` = embedding-only (not
    recommended: see ADR-016 for its measured false-hit rate)."""
    verifier_threshold: float = Field(default=0.992, gt=0, le=1)
    ttl_s: int = Field(default=3600, ge=1)
    max_prompt_chars: int = Field(default=2000, ge=1)


class CacheConfig(_Strict):
    exact: ExactCacheConfig = ExactCacheConfig()
    semantic: SemanticCacheConfig = SemanticCacheConfig()


class LoggingConfig(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class GatewayConfig(_Strict):
    logging: LoggingConfig = LoggingConfig()
    reliability: ReliabilityConfig = ReliabilityConfig()
    redis: RedisConfig = RedisConfig()
    auth: AuthConfig = AuthConfig()
    cache: CacheConfig = CacheConfig()
    providers: dict[str, ProviderConfig]
    routes: list[Route] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_references(self) -> GatewayConfig:
        seen: set[str] = set()
        for route in self.routes:
            if route.model in seen:
                raise ValueError(f"duplicate route for model {route.model!r}")
            seen.add(route.model)
            for target in route.targets:
                if target.provider not in self.providers:
                    raise ValueError(
                        f"route {route.model!r} references unknown provider {target.provider!r}"
                    )
        return self


def interpolate_env(text: str, env: dict[str, str] | None = None) -> str:
    """Replace ``${VAR}`` and ``${VAR:-default}`` with values from ``env``.

    A reference to an unset variable without a default is a configuration error; failing at
    startup is better than silently sending an empty API key upstream.
    """
    source = os.environ if env is None else env

    def replace(match: re.Match[str]) -> str:
        name, default = match.group("name"), match.group("default")
        value = source.get(name)
        if value is not None and value != "":
            return value
        if default is not None:
            return default
        if value == "":
            return ""
        raise ConfigError(f"environment variable {name} is referenced in config but not set")

    return _ENV_PATTERN.sub(replace, text)


def parse_config(text: str, env: dict[str, str] | None = None) -> GatewayConfig:
    try:
        raw = yaml.safe_load(interpolate_env(text, env))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    try:
        return GatewayConfig.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def load_config(path: str | Path | None = None) -> GatewayConfig:
    config_path = Path(path or os.environ.get("SWITCHYARD_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    return parse_config(config_path.read_text())
