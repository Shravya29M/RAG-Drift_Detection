"""Redis cache for query embeddings and top-k retrieval results.

Keys are ``rag:q:{index_version}:{digest}`` where the digest covers the
normalised query, ``k``, the metadata filters and the score threshold. The
index version changes on every index mutation (ingest, delete, swap), so a
lookup can never reach an entry computed against a different index; the
writer additionally purges the old version's keys, and every entry has a TTL.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np

from rag.logging import get_logger
from rag.metrics import (
    CACHE_BYPASSED,
    CACHE_HITS,
    CACHE_INVALIDATED_KEYS,
    CACHE_MISSES,
    CACHE_OPERATION_SECONDS,
)
from rag.models import Chunk
from rag.redis_health import REDIS_FAILURES, RedisHealth

logger = get_logger(__name__)

KEY_PREFIX = "rag:q"
_WHITESPACE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """NFKC-normalise, casefold and collapse whitespace.

    Only transformations that cannot change what the encoder is asked about
    in practice: the cached embedding is for the *normalised* text's first
    occurrence, so a variant differing in case/spacing reuses it.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", query)).strip().casefold()


@dataclass(frozen=True)
class CachedRetrieval:
    """Everything needed to rebuild a retrieval result without the encoder or index."""

    embedding: np.ndarray
    chunks: list[Chunk]
    scores: list[float]
    total_candidates: int

    def dumps(self) -> bytes:
        return json.dumps(
            {
                "e": base64.b64encode(
                    np.asarray(self.embedding, dtype=np.float32).tobytes()
                ).decode(),
                "c": [c.model_dump(mode="json") for c in self.chunks],
                "s": self.scores,
                "n": self.total_candidates,
            }
        ).encode()

    @classmethod
    def loads(cls, raw: bytes) -> CachedRetrieval:
        data = json.loads(raw)
        return cls(
            embedding=np.frombuffer(base64.b64decode(data["e"]), dtype=np.float32).copy(),
            chunks=[Chunk.model_validate(c) for c in data["c"]],
            scores=[float(s) for s in data["s"]],
            total_candidates=int(data["n"]),
        )


class QueryCache:
    """Versioned, TTL-bounded retrieval cache that never raises on Redis failure.

    Args:
        health: Shared Redis health tracker (owns the client).
        ttl_s: Entry lifetime in seconds.
    """

    def __init__(self, health: RedisHealth, *, ttl_s: int = 3600) -> None:
        self._health = health
        self._client: Any = health.client
        self._ttl_s = ttl_s

    @staticmethod
    def key(
        index_version: str,
        query: str,
        k: int,
        filters: dict[str, object] | None,
        score_threshold: float,
    ) -> str:
        """Build the cache key for one retrieval request."""
        payload = json.dumps(
            {
                "q": normalize_query(query),
                "k": k,
                "f": filters or {},
                "t": score_threshold,
            },
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()[:32]
        return f"{KEY_PREFIX}:{index_version}:{digest}"

    def get(self, key: str) -> CachedRetrieval | None:
        """Return the cached entry, or ``None`` on a miss or when Redis is unusable."""
        if not self._health.available():
            CACHE_BYPASSED.inc()
            return None
        t0 = time.perf_counter()
        try:
            raw = self._client.get(key)
        except REDIS_FAILURES as exc:
            self._health.mark_down("cache", exc)
            CACHE_BYPASSED.inc()
            return None
        finally:
            CACHE_OPERATION_SECONDS.labels(op="get").observe(time.perf_counter() - t0)
        if raw is None:
            CACHE_MISSES.inc()
            return None
        try:
            entry = CachedRetrieval.loads(raw)
        except (ValueError, KeyError, TypeError):
            logger.warning("dropping undecodable cache entry %s", key)
            CACHE_MISSES.inc()
            return None
        CACHE_HITS.inc()
        return entry

    def put(self, key: str, entry: CachedRetrieval) -> None:
        """Store *entry* with the configured TTL; failures are swallowed."""
        if not self._health.available():
            return
        t0 = time.perf_counter()
        try:
            self._client.set(key, entry.dumps(), ex=self._ttl_s)
        except REDIS_FAILURES as exc:
            self._health.mark_down("cache", exc)
        finally:
            CACHE_OPERATION_SECONDS.labels(op="put").observe(time.perf_counter() - t0)

    def purge_version(self, index_version: str) -> int:
        """Delete every entry for *index_version*; returns the number removed.

        Uses ``SCAN`` + ``UNLINK`` so a large purge never blocks Redis. Entries
        for superseded versions are already unreachable; this only frees memory
        sooner than the TTL would.
        """
        if not self._health.available():
            return 0
        removed = 0
        try:
            batch: list[bytes] = []
            for key in self._client.scan_iter(match=f"{KEY_PREFIX}:{index_version}:*", count=500):
                batch.append(key)
                if len(batch) >= 500:
                    removed += int(self._client.unlink(*batch))
                    batch.clear()
            if batch:
                removed += int(self._client.unlink(*batch))
        except REDIS_FAILURES as exc:
            self._health.mark_down("cache", exc)
        CACHE_INVALIDATED_KEYS.inc(removed)
        return removed
