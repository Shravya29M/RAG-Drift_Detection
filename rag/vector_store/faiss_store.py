"""FAISS-backed VectorStore implementation for local development."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import faiss  # type: ignore[import-untyped]  # faiss has no py.typed marker or stubs
import numpy as np

from rag.models import Chunk, SearchResult, VectorStoreConfig
from rag.vector_store.base import VectorStore


@dataclass
class _Slot:
    """All mutable state for one index slot (A or B).

    Bundling these together means an atomic swap is a single attribute
    assignment rather than four separate updates under a lock.
    """

    index: Any  # faiss.IndexIDMap wrapping faiss.IndexFlatIP
    id_map: dict[str, int] = field(default_factory=dict)
    """Maps string chunk IDs → int64 FAISS IDs."""
    chunks: dict[int, Chunk] = field(default_factory=dict)
    """Maps int64 FAISS IDs → Chunk objects for result hydration."""
    vectors: dict[int, np.ndarray] = field(default_factory=dict)
    """Maps int64 FAISS IDs → embedding vectors for distribution snapshot."""
    next_id: int = 0
    """Monotonically increasing int64 ID counter for this slot."""


def _new_index(dim: int) -> Any:
    """Return a fresh ``IndexIDMap(IndexFlatIP(dim))``."""
    return faiss.IndexIDMap(faiss.IndexFlatIP(dim))


def _build_slot(dim: int, chunks: list[Chunk], embeddings: np.ndarray) -> _Slot:
    """Construct a fully-populated :class:`_Slot` from *chunks* and *embeddings*.

    Args:
        dim: Embedding dimensionality.
        chunks: Chunk objects to index.
        embeddings: L2-normalised float32 array of shape ``(len(chunks), dim)``.

    Returns:
        Populated :class:`_Slot` ready to be made active.
    """
    slot = _Slot(index=_new_index(dim))
    if not chunks:
        return slot

    int_ids = np.arange(len(chunks), dtype=np.int64)
    vecs = np.asarray(embeddings, dtype=np.float32)
    slot.index.add_with_ids(vecs, int_ids)

    for i, (chunk, vec) in enumerate(zip(chunks, embeddings, strict=True)):
        slot.id_map[chunk.id] = i
        slot.chunks[i] = chunk
        slot.vectors[i] = np.asarray(vec, dtype=np.float32)

    slot.next_id = len(chunks)
    return slot


class FAISSStore(VectorStore):
    """Thread-safe FAISS vector store with A/B slot swap for atomic re-indexing.

    Uses ``IndexIDMap(IndexFlatIP(dim))`` so individual chunks can be deleted
    by their string ID.  All stored embeddings must be L2-normalised: dot
    product on unit vectors equals cosine similarity, which is what
    ``IndexFlatIP`` computes.

    Thread safety is provided by a :class:`threading.RLock`.  The expensive
    part of :meth:`swap_index` (building the new index) runs *outside* the
    lock; the lock is held only for the final pointer swap.

    Args:
        dim: Embedding dimensionality. Must match the encoder's output.
        config: Vector store tunables (top_k, score_threshold, …).
    """

    def __init__(
        self,
        dim: int,
        config: VectorStoreConfig | None = None,
    ) -> None:
        self._dim = dim
        self._config = config or VectorStoreConfig()
        self._lock = threading.RLock()
        # Two slots: index 0 is active initially, index 1 is the standby.
        self._slots: list[_Slot] = [_Slot(index=_new_index(dim)), _Slot(index=_new_index(dim))]
        self._active: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _slot(self) -> _Slot:
        """The currently active slot (read under lock by callers)."""
        return self._slots[self._active]

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add *chunks* and their *embeddings* to the active index slot.

        Args:
            chunks: Chunk objects; each ``chunk.id`` must be unique in the store.
            embeddings: L2-normalised float32 array, shape ``(len(chunks), dim)``.

        Raises:
            ValueError: If ``len(chunks) != embeddings.shape[0]``.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks length ({len(chunks)}) does not match "
                f"embeddings rows ({embeddings.shape[0]})"
            )
        if not chunks:
            return

        vecs = np.asarray(embeddings, dtype=np.float32)

        with self._lock:
            slot = self._slot
            int_ids = np.arange(
                slot.next_id,
                slot.next_id + len(chunks),
                dtype=np.int64,
            )
            slot.index.add_with_ids(vecs, int_ids)
            for i, (chunk, vec) in enumerate(zip(chunks, vecs, strict=True)):
                fid = slot.next_id + i
                slot.id_map[chunk.id] = fid
                slot.chunks[fid] = chunk
                slot.vectors[fid] = vec
            slot.next_id += len(chunks)

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        *,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Return the nearest chunks to *query*.

        Args:
            query: L2-normalised float32 vector, shape ``(dim,)`` or ``(1, dim)``.
            top_k: Maximum results to return.
            score_threshold: Exclude results with score below this value.

        Returns:
            :class:`~rag.models.SearchResult` list, descending by score,
            length ≤ ``top_k``.
        """
        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 1:
            q = q[np.newaxis, :]  # (1, dim)

        with self._lock:
            slot = self._slot
            if slot.index.ntotal == 0:
                return []

            k = min(top_k, slot.index.ntotal)
            raw_d, raw_i = slot.index.search(q, k)

        distances: np.ndarray = np.asarray(raw_d[0])
        indices: np.ndarray = np.asarray(raw_i[0])

        results: list[SearchResult] = []
        for rank, (score_val, fid) in enumerate(zip(distances, indices, strict=True)):
            score = float(score_val)
            iid = int(fid)
            if iid == -1:  # FAISS sentinel for "no result"
                continue
            if score < score_threshold:
                continue
            with self._lock:
                chunk = self._slot.chunks.get(iid)
            if chunk is None:
                continue
            results.append(SearchResult(chunk=chunk, score=score, rank=rank))

        return results

    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by their string IDs.

        Unknown IDs are silently ignored.
        """
        with self._lock:
            slot = self._slot
            to_remove: list[int] = []
            for cid in chunk_ids:
                fid = slot.id_map.pop(cid, None)
                if fid is not None:
                    to_remove.append(fid)
                    slot.chunks.pop(fid, None)
                    slot.vectors.pop(fid, None)
            if to_remove:
                fa_ids = np.array(to_remove, dtype=np.int64)
                slot.index.remove_ids(fa_ids)

    def snapshot_distribution(self) -> np.ndarray:
        """Return all stored embeddings stacked into a single array.

        Returns:
            Float32 array of shape ``(n, dim)``, or ``(0, dim)`` if empty.
        """
        with self._lock:
            slot = self._slot
            if not slot.vectors:
                return np.empty((0, self._dim), dtype=np.float32)
            return np.stack(list(slot.vectors.values()), axis=0)

    def swap_index(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Atomically replace the entire index.

        Builds the new index in the *inactive* slot (without holding the lock),
        then acquires the lock only for the pointer swap so in-flight
        :meth:`search` calls are never blocked during the build phase.

        Args:
            chunks: Full new chunk set.
            embeddings: L2-normalised float32 array, shape ``(len(chunks), dim)``.
        """
        # Build outside the lock — this is the expensive step.
        new_slot = _build_slot(self._dim, chunks, embeddings)

        with self._lock:
            inactive = 1 - self._active
            self._slots[inactive] = new_slot
            self._active = inactive
