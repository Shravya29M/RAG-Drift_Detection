"""Shared Pydantic v2 models used across all layers of the RAG-drift pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, PositiveInt


class SourceType(StrEnum):
    """Supported document source types."""

    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    URL = "url"


class ChunkMetadata(BaseModel):
    """Provenance metadata attached to every text chunk.

    Preserved through ingestion → embedding → vector store so that retrieval
    results can be traced back to their origin document and position.
    """

    source: str = Field(description="Original filename or URL the chunk came from.")
    source_type: SourceType = Field(description="Format of the originating document.")
    page_number: int | None = Field(
        default=None,
        ge=1,
        description="1-based page number for PDF sources; None for other types.",
    )
    section_header: str | None = Field(
        default=None,
        description="Nearest section heading above the chunk, if extractable.",
    )
    chunk_index: int = Field(
        ge=0,
        description="0-based position of this chunk within its source document.",
    )
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the chunk was created during ingestion.",
    )
    file_path: Path | None = Field(
        default=None,
        description="Absolute path to the source file on disk; None for URL sources.",
    )


class Chunk(BaseModel):
    """A single text chunk ready for embedding.

    The atomic unit that flows from the ingestion layer into the embedding
    layer and then into the vector store.
    """

    id: str = Field(description="Stable unique identifier: '<source_hash>-<chunk_index>'.")
    text: str = Field(min_length=1, description="The raw text content of the chunk.")
    metadata: ChunkMetadata
    token_count: int = Field(
        gt=0,
        description="Approximate token count of the text field.",
    )


class IngestConfig(BaseModel):
    """Configuration for a single ingestion run.

    Values default to the tunables in config/default.yaml and can be
    overridden per-request via the POST /ingest endpoint.
    """

    chunk_size: PositiveInt = Field(
        default=512,
        description="Maximum number of tokens per chunk.",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        description="Number of tokens shared between consecutive chunks.",
    )
    source_type: SourceType | None = Field(
        default=None,
        description="Force a specific parser; inferred from file extension if None.",
    )
    incremental: bool = Field(
        default=False,
        description="Skip documents already present in the vector store.",
    )


class SearchResult(BaseModel):
    """A single result returned by :meth:`~rag.vector_store.base.VectorStore.search`.

    Wraps the matched chunk together with its retrieval score and rank so
    callers don't need to zip separate lists.
    """

    chunk: Chunk = Field(description="The matched chunk.")
    score: float = Field(
        description="Cosine similarity score (dot product of L2-normalised vectors).",
    )
    rank: int = Field(
        ge=0,
        description="0-based position in the result list (0 = most similar).",
    )


class RetrievalResult(BaseModel):
    """Full result of a single retrieval call, including observability fields.

    The generation layer uses ``chunks`` for prompt construction; the tracking
    module logs ``scores``, ``latency_ms``, and ``total_candidates`` to W&B.
    """

    query: str = Field(description="Original query string.")
    chunks: list[Chunk] = Field(description="Top-k chunks after filtering, ranked by score.")
    scores: list[float] = Field(
        description="Cosine similarity scores parallel to ``chunks``.",
    )
    latency_ms: float = Field(
        ge=0.0,
        description="Wall-clock retrieval time in milliseconds.",
    )
    total_candidates: int = Field(
        ge=0,
        description="Raw search result count before metadata filtering.",
    )


class VectorStoreConfig(BaseModel):
    """Configuration for the vector store layer.

    Values correspond to the ``vector_store`` block in ``config/default.yaml``.
    """

    backend: str = Field(
        default="faiss",
        description="Which backend to use: 'faiss' (local) or 'qdrant' (prod).",
    )
    faiss_index_path: Path = Field(
        default=Path("index/faiss.index"),
        description="Path where the FAISS index file is persisted.",
    )
    qdrant_url: str = Field(
        default="",
        description="Qdrant server URL; overridden by env QDRANT_URL.",
    )
    top_k: PositiveInt = Field(
        default=5,
        description="Default number of nearest neighbours to retrieve.",
    )
    score_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score to include a hit.",
    )


class GenerationConfig(BaseModel):
    """Configuration for the generation layer.

    Values correspond to the ``generation`` block in ``config/default.yaml``.
    """

    llm_provider: str = Field(
        default="openai",
        description="LLM backend: 'openai' or 'anthropic'.",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model identifier.",
    )
    anthropic_model: str = Field(
        default="claude-opus-4-6",
        description="Anthropic model identifier.",
    )
    max_tokens: PositiveInt = Field(
        default=1024,
        description="Maximum tokens to generate.",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    prompt_template: str = Field(
        default=(
            "You are a helpful assistant. Answer the question using ONLY the context "
            "provided below. If the context does not contain the answer, respond with "
            "'No information found.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {query}\n\n"
            "Answer:"
        ),
        description="Prompt template. Must contain '{context}' and '{query}' placeholders.",
    )
    no_hit_response: str = Field(
        default="No information found.",
        description="Verbatim response when no relevant chunk is retrieved.",
    )


class AlarmConfig(BaseModel):
    """Configuration for the drift alarm manager."""

    webhook_url: str = Field(
        default="",
        description="HTTP(S) URL to POST hard-alert payloads to. Empty string disables.",
    )
    webhook_timeout_s: float = Field(
        default=5.0,
        gt=0.0,
        description="Seconds before the webhook POST times out.",
    )


class DriftConfig(BaseModel):
    """Configuration for the drift detection module.

    Values correspond to the ``drift`` block in ``config/default.yaml``.
    """

    window_size: PositiveInt = Field(
        default=50,
        description="Number of query embeddings per drift evaluation window.",
    )
    pca_components: PositiveInt = Field(
        default=32,
        description="Number of PCA dimensions to reduce to before the KS test.",
    )
    threshold_alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="KS statistic threshold above which a soft alert is raised.",
    )
    hysteresis_windows: PositiveInt = Field(
        default=3,
        description="Consecutive alert windows required before triggering re-index.",
    )
    method: str = Field(
        default="ks",
        description="Detection method: 'ks' (KS test per PCA dimension, max stat).",
    )


class DriftResult(BaseModel):
    """Result of a single drift evaluation window."""

    statistic: float = Field(
        description="KS test statistic (max across PCA dimensions).",
    )
    pvalue: float = Field(
        description="p-value corresponding to the max-statistic dimension.",
    )
    drifted: bool = Field(
        description="True when statistic exceeds DriftConfig.threshold_alpha.",
    )
    window_size: int = Field(
        ge=1,
        description="Number of query embeddings in this window.",
    )
    snapshot_size: int = Field(
        ge=1,
        description="Number of chunk embeddings in the reference snapshot.",
    )
    evaluated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of this evaluation.",
    )


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding layer.

    Values correspond to the ``embedding`` block in ``config/default.yaml``.
    """

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="sentence-transformers model identifier (or OpenAI model name).",
    )
    batch_size: PositiveInt = Field(
        default=64,
        description="Number of texts encoded per forward pass.",
    )
    cache_dir: Path | None = Field(
        default=None,
        description="Directory for caching encoded embeddings. None disables the cache.",
    )
