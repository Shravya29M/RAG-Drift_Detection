"""Centralised settings: config/default.yaml plus .env, loaded once at startup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from rag.models import (
    AlarmConfig,
    DriftConfig,
    EmbeddingConfig,
    GenerationConfig,
    IngestConfig,
    SchedulerConfig,
    VectorStoreConfig,
)

_M = TypeVar("_M", bound=BaseModel)


class Settings(BaseModel):
    """All tunables for the pipeline, one sub-config per layer."""

    ingestion: IngestConfig = Field(default_factory=IngestConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    alarm: AlarmConfig = Field(default_factory=AlarmConfig)


def _build(model_cls: type[_M], block: object) -> _M:
    """Instantiate *model_cls* from a YAML block, ignoring unknown keys."""
    if not isinstance(block, dict):
        return model_cls()
    fields = set(model_cls.model_fields)
    return model_cls(**{k: v for k, v in block.items() if k in fields})


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from the YAML config file, applying env-var overrides.

    Also loads ``.env`` into the process environment (existing variables are
    never overwritten).  Missing config file falls back to model defaults.

    Env overrides: ``QDRANT_URL`` → ``vector_store.qdrant_url``,
    ``DRIFT_WEBHOOK_URL`` → ``alarm.webhook_url``.
    """
    load_dotenv()

    path = config_path or Path(os.environ.get("RAG_CONFIG", "config/default.yaml"))
    raw: dict[str, object] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw = loaded

    settings = Settings(
        ingestion=_build(IngestConfig, raw.get("ingestion")),
        embedding=_build(EmbeddingConfig, raw.get("embedding")),
        vector_store=_build(VectorStoreConfig, raw.get("vector_store")),
        generation=_build(GenerationConfig, raw.get("generation")),
        drift=_build(DriftConfig, raw.get("drift")),
        scheduler=_build(SchedulerConfig, raw.get("scheduler")),
        alarm=_build(AlarmConfig, raw.get("alarm")),
    )

    qdrant_url = os.environ.get("QDRANT_URL")
    if qdrant_url:
        settings.vector_store.qdrant_url = qdrant_url
    webhook_url = os.environ.get("DRIFT_WEBHOOK_URL")
    if webhook_url:
        settings.alarm.webhook_url = webhook_url

    return settings
