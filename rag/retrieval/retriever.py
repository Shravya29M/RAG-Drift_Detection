"""Top-k cosine similarity retriever with optional metadata filtering."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from rag.cache.query_cache import CachedRetrieval, QueryCache
from rag.embedding.encoder import Encoder
from rag.metrics import RETRIEVAL_SECONDS
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
        cache: Optional shared query cache. Requires *index_version*.
        index_version: Returns the current index version; read *before* the
            search so an entry can never be filed under a newer version than
            the index that produced it.
    """

    def __init__(
        self,
        store: VectorStore,
        encoder: Encoder,
        *,
        score_threshold: float = 0.0,
        cache: QueryCache | None = None,
        index_version: Callable[[], str] | None = None,
    ) -> None:
        self._store = store
        self._encoder = encoder
        self._score_threshold = score_threshold
        self._cache = cache if index_version is not None else None
        self._index_version = index_version

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

        With a cache configured, a hit skips steps 1–4 entirely; a miss runs
        them and stores the result. Either way ``query_embedding`` is set so
        the caller never has to encode the query a second time.

        Returns:
            :class:`~rag.models.RetrievalResult` with ``chunks``, ``scores``,
            ``latency_ms``, ``total_candidates``, ``cache_hit`` and
            ``query_embedding``.
        """
        t0 = time.monotonic()

        cache_key: str | None = None
        if self._cache is not None and self._index_version is not None:
            cache_key = self._cache.key(
                self._index_version(), query, k, filters, self._score_threshold
            )
            hit = self._cache.get(cache_key)
            if hit is not None:
                latency_s = time.monotonic() - t0
                RETRIEVAL_SECONDS.labels(cache="hit").observe(latency_s)
                return RetrievalResult(
                    query=query,
                    chunks=hit.chunks,
                    scores=hit.scores,
                    latency_ms=latency_s * 1000.0,
                    total_candidates=hit.total_candidates,
                    cache_hit=True,
                    query_embedding=hit.embedding.tolist(),
                )

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

        chunks = [r.chunk for r in top]
        scores = [r.score for r in top]
        if self._cache is not None and cache_key is not None:
            self._cache.put(
                cache_key,
                CachedRetrieval(
                    embedding=np.asarray(query_vec, dtype=np.float32),
                    chunks=chunks,
                    scores=scores,
                    total_candidates=total_candidates,
                ),
            )

        latency_s = time.monotonic() - t0
        RETRIEVAL_SECONDS.labels(cache="miss" if cache_key is not None else "off").observe(
            latency_s
        )

        return RetrievalResult(
            query=query,
            chunks=chunks,
            scores=scores,
            latency_ms=latency_s * 1000.0,
            total_candidates=total_candidates,
            query_embedding=np.asarray(query_vec, dtype=np.float32).tolist(),
        )
