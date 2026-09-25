from __future__ import annotations

from pathlib import Path

import pytest

from switchyard.config import ConfigError, interpolate_env, load_config, parse_config
from switchyard.providers import build_providers

MINIMAL = """
providers:
  groq:
    type: groq
    base_url: https://api.groq.com/openai/v1/
    api_key: ${GROQ_API_KEY:-}
  mock:
    type: mock
    base_url: ${MOCK_URL:-http://mock:9000/v1}
routes:
  - model: fast
    targets:
      - {provider: groq, model: llama}
      - {provider: mock, model: mock-1}
"""


def test_interpolation_uses_env_then_default() -> None:
    assert interpolate_env("${A:-x}", {"A": "y"}) == "y"
    assert interpolate_env("${A:-x}", {}) == "x"
    assert interpolate_env("${A:-x}", {"A": ""}) == "x"
    assert interpolate_env("a${B}c", {"B": "b"}) == "abc"


def test_interpolation_fails_on_unset_variable_without_default() -> None:
    with pytest.raises(ConfigError, match="MISSING"):
        interpolate_env("${MISSING}", {})


def test_parse_minimal_config() -> None:
    config = parse_config(MINIMAL, {"MOCK_URL": "http://localhost:9001/v1"})
    assert config.providers["mock"].base_url == "http://localhost:9001/v1"
    assert config.providers["groq"].base_url == "https://api.groq.com/openai/v1"
    assert config.providers["groq"].api_key is None
    assert [t.provider for t in config.routes[0].targets] == ["groq", "mock"]


def test_api_key_is_secret() -> None:
    config = parse_config(MINIMAL, {"GROQ_API_KEY": "gsk_secret"})
    key = config.providers["groq"].api_key
    assert key is not None
    assert key.get_secret_value() == "gsk_secret"
    assert "gsk_secret" not in repr(config)


def test_route_to_unknown_provider_is_rejected() -> None:
    bad = MINIMAL.replace("{provider: mock, model: mock-1}", "{provider: nope, model: x}")
    with pytest.raises(ConfigError, match="unknown provider 'nope'"):
        parse_config(bad, {})


def test_duplicate_route_is_rejected() -> None:
    bad = MINIMAL + "  - model: fast\n    targets: [{provider: mock, model: m}]\n"
    with pytest.raises(ConfigError, match="duplicate route"):
        parse_config(bad, {})


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ConfigError):
        parse_config(MINIMAL + "surprise: true\n", {})


def test_groq_without_key_is_skipped() -> None:
    providers = build_providers(parse_config(MINIMAL, {}))
    assert set(providers) == {"mock"}
    providers = build_providers(parse_config(MINIMAL, {"GROQ_API_KEY": "k"}))
    assert set(providers) == {"groq", "mock"}


def test_repo_config_file_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    config = load_config(Path(__file__).parents[2] / "config" / "gateway.yaml")
    assert "mock" in {r.model for r in config.routes}


def test_missing_config_file() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("/nonexistent.yaml")
