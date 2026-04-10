"""Top-k cosine similarity retriever with optional metadata filtering."""

from __future__ import annotations

import time

import numpy as np

from rag.embedding.encoder import Encoder
from rag.models import RetrievalResult, SearchResult
from rag.vector_store.base import VectorStore

# When filters are active we over-retrieve so filtering doesn't starve the
# result set.  E.g. with k=5 and factor=4 we fetch up to 20 candidates first.
_OVERSAMPLE_FACTOR = 4

# Sentinel used by _apply_filters to detect missing metadata attributes.
_MISSING = object()


def _apply_filters(
    results: list[SearchResult],
    filters: dict[str, object],
) -> list[SearchResult]:
    """Return only those results whose chunk metadata matches every filter key.

    Each key in *filters* must be a field name on
    :class:`~rag.models.ChunkMetadata`.  Values are compared with ``==``.
    Results whose metadata lacks a requested key are excluded.

    Args:
        results: Candidate search results to filter.
        filters: Mapping of ``ChunkMetadata`` field name → expected value.

    Returns:
        Subset of *results* where every filter predicate holds, in original order.
    """
    kept: list[SearchResult] = []
    for r in results:
        meta = r.chunk.metadata
        if all(getattr(meta, key, _MISSING) == val for key, val in filters.items()):
            kept.append(r)
    return kept


class Retriever:
    """Encodes a query string and retrieves the most relevant chunks.

    Combines an :class:`~rag.embedding.encoder.Encoder` (query encoding) with a
    :class:`~rag.vector_store.base.VectorStore` (nearest-neighbour search) and
    optional Python-side metadata filtering.

    Args:
        store: Vector store to search against.
        encoder: Encoder used to embed the query string.
        score_threshold: Minimum cosine similarity score passed to the vector
            store; results below this value are discarded before filtering.
    """

    def __init__(
        self,
        store: VectorStore,
        encoder: Encoder,
        *,
        score_threshold: float = 0.0,
    ) -> None:
        self._store = store
        self._encoder = encoder
        self._score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        k: int,
        filters: dict[str, object] | None = None,
    ) -> RetrievalResult:
        """Retrieve the *k* most relevant chunks for *query*.

        Steps:
        1. Encode *query* to a unit-norm vector via the encoder.
        2. Over-sample from the vector store (``k × _OVERSAMPLE_FACTOR`` when
           filters are set) to leave headroom for post-filter drops.
        3. Apply metadata filters (Python-side equality checks on
           :class:`~rag.models.ChunkMetadata` fields).
        4. Trim to *k* and return a :class:`~rag.models.RetrievalResult`.

        Args:
            query: Natural-language query string.
            k: Maximum number of chunks to return.
            filters: Optional mapping of ``ChunkMetadata`` field name →
                expected value.  All predicates must hold (AND semantics).
                Pass ``None`` or ``{}`` to skip filtering.

        Returns:
            :class:`~rag.models.RetrievalResult` with ``chunks``, ``scores``,
            ``latency_ms``, and ``total_candidates``.
        """
        t0 = time.monotonic()

        # Encode query: encoder returns (1, dim); take the single row.
        query_vec: np.ndarray = self._encoder.encode([query])[0]

        # Over-sample when filters are active to avoid under-returning.
        oversample_k = k * _OVERSAMPLE_FACTOR if filters else k
        raw: list[SearchResult] = self._store.search(
            query_vec,
            oversample_k,
            score_threshold=self._score_threshold,
        )
        total_candidates = len(raw)

        # Apply metadata filters (no-op when filters is None or empty).
        filtered = _apply_filters(raw, filters) if filters else raw

        # Trim to k.
        top = filtered[:k]

        latency_ms = (time.monotonic() - t0) * 1000.0

        return RetrievalResult(
            query=query,
            chunks=[r.chunk for r in top],
            scores=[r.score for r in top],
            latency_ms=latency_ms,
            total_candidates=total_candidates,
        )
