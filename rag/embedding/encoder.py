"""Pluggable abstract encoder interface for producing dense text embeddings."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sentence_transformers import SentenceTransformer


class Encoder(ABC):
    """Abstract base class for all text encoders.

    Consuming code (retriever, drift detector) must depend only on this
    interface — never import a concrete implementation directly.
    """

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into L2-normalised dense vectors.

        Args:
            texts: Non-empty list of strings to encode.

        Returns:
            Float32 array of shape ``(len(texts), embedding_dim)`` where every
            row has unit L2 norm.
        """


class SentenceTransformerEncoder(Encoder):
    """Encoder backed by a sentence-transformers model.

    Embeddings are L2-normalised after encoding so they are compatible with
    ``faiss.IndexFlatIP`` cosine-similarity search (dot product on unit vectors
    equals cosine similarity).

    Args:
        model_name: Any model identifier accepted by
            :class:`sentence_transformers.SentenceTransformer`.
        batch_size: Number of texts encoded per forward pass.
    """

    def __init__(self, model_name: str, *, batch_size: int = 64) -> None:
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* and return L2-normalised embeddings.

        Args:
            texts: Strings to encode. Must not be empty.

        Returns:
            Float32 array of shape ``(len(texts), embedding_dim)`` with unit
            L2-norm rows.
        """
        raw = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # normalise manually for explicit control
            show_progress_bar=False,
        )
        arr: np.ndarray = np.asarray(raw, dtype=np.float32)
        return _l2_normalize(arr)


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """Return a copy of *arr* with each row scaled to unit L2 norm.

    Zero vectors are left as-is (norm treated as 1 to avoid NaN).

    Args:
        arr: 2-D float array of shape ``(n, d)``.

    Returns:
        Array of the same shape and dtype with unit-norm rows.
    """
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return np.asarray(arr / safe_norms, dtype=arr.dtype)
