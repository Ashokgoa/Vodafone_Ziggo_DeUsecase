"""Tests for app.config.settings — proves env vars are read correctly."""

import pytest

from app.config.settings import get_settings


def test_defaults_when_env_vars_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # monkeypatch.delenv temporarily removes an env var for this test only,
    # so we can check the fallback defaults regardless of the machine's
    # actual environment.
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("ZIGGO_URL", raising=False)
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("VECTOR_STORE_DIR", raising=False)
    monkeypatch.delenv("RETRIEVAL_TOP_K", raising=False)
    monkeypatch.delenv("RETRIEVAL_MAX_DISTANCE", raising=False)
    monkeypatch.delenv("FALLBACK_MESSAGE", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)

    settings = get_settings()

    assert settings.app_env == "local"
    assert settings.log_level == "INFO"
    assert settings.target_url == "https://www.ziggo.nl/internet"
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.embedding_model == "paraphrase-multilingual-MiniLM-L12-v2"
    assert settings.vector_store_dir == "data/vectorstore"
    assert settings.retrieval_top_k == 3
    assert settings.retrieval_max_distance == 0.65
    assert settings.fallback_message  # non-empty; exact wording isn't load-bearing
    assert settings.llm_model == "claude-haiku-4-5"
    assert settings.llm_max_tokens == 500


def test_reads_overridden_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ZIGGO_URL", "https://example.com/page")
    monkeypatch.setenv("CHUNK_SIZE", "300")
    monkeypatch.setenv("CHUNK_OVERLAP", "30")
    monkeypatch.setenv("EMBEDDING_MODEL", "fake-model")
    monkeypatch.setenv("VECTOR_STORE_DIR", "/tmp/fake-store")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "5")
    monkeypatch.setenv("RETRIEVAL_MAX_DISTANCE", "0.4")
    monkeypatch.setenv("FALLBACK_MESSAGE", "Sorry, geen antwoord gevonden.")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("LLM_MAX_TOKENS", "800")

    settings = get_settings()

    assert settings.app_env == "production"
    assert settings.log_level == "DEBUG"
    assert settings.target_url == "https://example.com/page"
    assert settings.chunk_size == 300
    assert settings.chunk_overlap == 30
    assert settings.embedding_model == "fake-model"
    assert settings.vector_store_dir == "/tmp/fake-store"
    assert settings.retrieval_top_k == 5
    assert settings.retrieval_max_distance == 0.4
    assert settings.fallback_message == "Sorry, geen antwoord gevonden."
    assert settings.llm_model == "claude-sonnet-5"
    assert settings.llm_max_tokens == 800
