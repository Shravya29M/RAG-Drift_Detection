"""Settings loading was the least-covered module in the package (50%), which
is a bad place to have gaps: a silently-ignored config block or a dropped env
override changes retrieval behaviour without failing anything loudly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rag.models import IngestConfig
from rag.settings import Settings, _build, load_settings


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch, tmp_path):
    """Keep the developer's real .env and config out of these tests."""
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("DRIFT_WEBHOOK_URL", raising=False)
    monkeypatch.chdir(tmp_path)


def write_config(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# _build
# --------------------------------------------------------------------------


def test_build_returns_defaults_for_a_missing_block():
    assert _build(IngestConfig, None) == IngestConfig()


def test_build_returns_defaults_when_the_block_is_not_a_mapping():
    assert _build(IngestConfig, ["not", "a", "mapping"]) == IngestConfig()


def test_build_applies_known_keys():
    default = IngestConfig()
    built = _build(IngestConfig, {"chunk_size": default.chunk_size + 64})
    assert built.chunk_size == default.chunk_size + 64


def test_build_ignores_unknown_keys_rather_than_raising():
    """A stale key in someone's config file must not break startup."""
    built = _build(IngestConfig, {"chunk_size": 256, "not_a_real_setting": True})
    assert built.chunk_size == 256


# --------------------------------------------------------------------------
# load_settings
# --------------------------------------------------------------------------


def test_missing_config_file_falls_back_to_defaults(tmp_path):
    settings = load_settings(tmp_path / "absent.yaml")
    assert settings == Settings()


def test_empty_config_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_settings(path) == Settings()


def test_a_yaml_scalar_instead_of_a_mapping_falls_back_to_defaults(tmp_path):
    path = tmp_path / "scalar.yaml"
    path.write_text("just-a-string", encoding="utf-8")
    assert load_settings(path) == Settings()


def test_every_config_block_is_wired_up(tmp_path):
    """A block present in YAML but not threaded into Settings would be
    silently ignored, so check each one round-trips."""
    defaults = Settings()
    path = write_config(
        tmp_path,
        {
            "ingestion": {"chunk_size": defaults.ingestion.chunk_size + 1},
            "embedding": {"batch_size": defaults.embedding.batch_size + 1},
            "vector_store": {"top_k": defaults.vector_store.top_k + 1},
            "generation": {"temperature": 0.42},
            "drift": {"window_size": defaults.drift.window_size + 1},
            "scheduler": {
                "drift_check_interval_seconds": defaults.scheduler.drift_check_interval_seconds + 1
            },
            "alarm": {"webhook_url": "https://example.test/hook"},
        },
    )
    settings = load_settings(path)
    assert settings.ingestion.chunk_size == defaults.ingestion.chunk_size + 1
    assert settings.embedding.batch_size == defaults.embedding.batch_size + 1
    assert settings.vector_store.top_k == defaults.vector_store.top_k + 1
    assert settings.generation.temperature == 0.42
    assert settings.drift.window_size == defaults.drift.window_size + 1
    assert (
        settings.scheduler.drift_check_interval_seconds
        == defaults.scheduler.drift_check_interval_seconds + 1
    )
    assert settings.alarm.webhook_url == "https://example.test/hook"


def test_a_partial_block_keeps_defaults_for_the_rest(tmp_path):
    defaults = Settings()
    path = write_config(tmp_path, {"ingestion": {"chunk_size": 999}})
    settings = load_settings(path)
    assert settings.ingestion.chunk_size == 999
    assert settings.ingestion.chunk_overlap == defaults.ingestion.chunk_overlap


def test_rag_config_env_var_selects_the_config_file(tmp_path, monkeypatch):
    path = write_config(tmp_path, {"ingestion": {"chunk_size": 777}})
    monkeypatch.setenv("RAG_CONFIG", str(path))
    assert load_settings().ingestion.chunk_size == 777


def test_an_explicit_path_wins_over_the_env_var(tmp_path, monkeypatch):
    env_path = write_config(tmp_path, {"ingestion": {"chunk_size": 111}})
    monkeypatch.setenv("RAG_CONFIG", str(env_path))
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(yaml.safe_dump({"ingestion": {"chunk_size": 222}}), encoding="utf-8")
    assert load_settings(explicit).ingestion.chunk_size == 222


# --------------------------------------------------------------------------
# env overrides
# --------------------------------------------------------------------------


def test_qdrant_url_env_var_overrides_the_config_file(tmp_path, monkeypatch):
    path = write_config(tmp_path, {"vector_store": {"qdrant_url": "http://from-yaml"}})
    monkeypatch.setenv("QDRANT_URL", "http://from-env")
    assert load_settings(path).vector_store.qdrant_url == "http://from-env"


def test_drift_webhook_env_var_overrides_the_config_file(tmp_path, monkeypatch):
    path = write_config(tmp_path, {"alarm": {"webhook_url": "https://from-yaml"}})
    monkeypatch.setenv("DRIFT_WEBHOOK_URL", "https://from-env")
    assert load_settings(path).alarm.webhook_url == "https://from-env"


def test_an_empty_env_override_does_not_blank_the_config_value(tmp_path, monkeypatch):
    """An unset-but-exported variable must not wipe a configured URL."""
    path = write_config(tmp_path, {"alarm": {"webhook_url": "https://from-yaml"}})
    monkeypatch.setenv("DRIFT_WEBHOOK_URL", "")
    assert load_settings(path).alarm.webhook_url == "https://from-yaml"


def test_env_overrides_apply_even_without_a_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://from-env")
    assert load_settings(tmp_path / "absent.yaml").vector_store.qdrant_url == "http://from-env"
