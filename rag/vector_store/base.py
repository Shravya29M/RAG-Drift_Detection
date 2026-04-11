"""Abstract VectorStore interface; all concrete implementations inherit from this."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from rag.models import Chunk, SearchResult


class VectorStore(ABC):
    """Abstract base for all vector store backends (FAISS, Qdrant, …).

    Consuming code (retriever, drift detector, re-index scheduler) must depend
    only on this interface — never import a concrete implementation directly.

    All implementations must be thread-safe: concurrent calls to
    :meth:`search` must not race with :meth:`add`, :meth:`delete`, or
    :meth:`swap_index`.
    """

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Persist *chunks* and their pre-computed *embeddings*.

        Args:
            chunks: Chunk objects whose IDs are used as stable identifiers.
            embeddings: Float32 array of shape ``(len(chunks), dim)``; rows
                must already be L2-normalised.

        Raises:
            ValueError: If ``len(chunks) != embeddings.shape[0]``.
        """

    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        top_k: int,
        *,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Return the *top_k* nearest chunks to *query*.

        Args:
            query: L2-normalised float32 vector of shape ``(dim,)``.
            top_k: Maximum number of results to return.
            score_threshold: Minimum cosine-similarity score; results below
                this value are excluded.

        Returns:
            List of :class:`~rag.models.SearchResult` sorted by descending
            score, length ≤ ``top_k``.
        """

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by their string IDs.

        Unknown IDs are silently ignored.
        """

    @abstractmethod
    def snapshot_distribution(self) -> np.ndarray:
        """Return all stored embedding vectors as a single array.

        Returns:
            Float32 array of shape ``(n_chunks, dim)``, or shape ``(0, dim)``
            if the store is empty.  Used by the drift detector to snapshot the
            reference distribution at index-build time.
        """

    @abstractmethod
    def swap_index(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Atomically replace the entire index with new *chunks* / *embeddings*.

        Implementations must build the replacement in the background (inactive
        slot) and then perform an atomic pointer swap so that in-flight
        :meth:`search` calls are never interrupted.

        Args:
            chunks: Full new chunk set.
            embeddings: Float32 array of shape ``(len(chunks), dim)``; rows
                must already be L2-normalised.
        """

    @abstractmethod
    def list_chunks(self) -> list[Chunk]:
        """Return all chunks currently stored in the active index slot.

        Returns:
            List of :class:`~rag.models.Chunk` objects in unspecified order.
            Empty list if the store has no chunks.
        """
