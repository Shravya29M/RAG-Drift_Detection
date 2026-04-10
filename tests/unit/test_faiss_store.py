"""Unit tests for rag.vector_store.faiss_store."""

from __future__ import annotations

import threading
from datetime import datetime

import numpy as np
import pytest

from rag.models import Chunk, ChunkMetadata, SearchResult, SourceType
from rag.vector_store.base import VectorStore
from rag.vector_store.faiss_store import FAISSStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8


def _unit(v: list[float]) -> np.ndarray:
    """Return *v* L2-normalised as a float32 ndarray."""
    arr = np.array(v, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


def _make_chunk(cid: str) -> Chunk:
    return Chunk(
        id=cid,
        text=f"text for {cid}",
        token_count=3,
        metadata=ChunkMetadata(
            source="test.txt",
            source_type=SourceType.TEXT,
            chunk_index=0,
            ingested_at=datetime(2026, 1, 1),
        ),
    )


def _random_unit_vecs(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return np.asarray(vecs / norms, dtype=np.float32)


def _store(*chunk_ids: str) -> tuple[FAISSStore, list[Chunk], np.ndarray]:
    """Return a FAISSStore pre-loaded with one unit vector per chunk id."""
    chunks = [_make_chunk(cid) for cid in chunk_ids]
    vecs = _random_unit_vecs(len(chunk_ids))
    store = FAISSStore(dim=DIM)
    if chunks:
        store.add(chunks, vecs)
    return store, chunks, vecs


# ---------------------------------------------------------------------------
# VectorStore ABC
# ---------------------------------------------------------------------------


class TestVectorStoreABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            VectorStore()  # type: ignore[abstract]

    def test_faiss_store_is_vector_store(self) -> None:
        assert isinstance(FAISSStore(dim=DIM), VectorStore)


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_increases_index_size(self) -> None:
        store = FAISSStore(dim=DIM)
        chunks = [_make_chunk("a"), _make_chunk("b")]
        store.add(chunks, _random_unit_vecs(2))
        assert store._slot.index.ntotal == 2

    def test_add_mismatched_length_raises(self) -> None:
        store = FAISSStore(dim=DIM)
        with pytest.raises(ValueError, match="does not match"):
            store.add([_make_chunk("a")], _random_unit_vecs(3))

    def test_add_empty_list_is_noop(self) -> None:
        store = FAISSStore(dim=DIM)
        store.add([], np.empty((0, DIM), dtype=np.float32))
        assert store._slot.index.ntotal == 0

    def test_add_multiple_batches_accumulates(self) -> None:
        store = FAISSStore(dim=DIM)
        store.add([_make_chunk("a")], _random_unit_vecs(1, seed=1))
        store.add([_make_chunk("b"), _make_chunk("c")], _random_unit_vecs(2, seed=2))
        assert store._slot.index.ntotal == 3


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_returns_list_of_search_results(self) -> None:
        store, _, vecs = _store("a", "b", "c")
        results = store.search(vecs[0], top_k=2)
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_top_k_limits_results(self) -> None:
        store, _, vecs = _store("a", "b", "c", "d")
        results = store.search(vecs[0], top_k=2)
        assert len(results) <= 2

    def test_search_returns_most_similar_first(self) -> None:
        """The query vector itself should score 1.0 and rank first."""
        store, chunks, vecs = _store("a", "b", "c")
        results = store.search(vecs[0], top_k=3)
        assert results[0].chunk.id == "a"
        assert pytest.approx(results[0].score, abs=1e-5) == 1.0

    def test_search_scores_are_descending(self) -> None:
        store, _, vecs = _store("a", "b", "c", "d")
        results = store.search(vecs[0], top_k=4)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_rank_matches_position(self) -> None:
        store, _, vecs = _store("a", "b", "c")
        results = store.search(vecs[0], top_k=3)
        for i, r in enumerate(results):
            assert r.rank == i

    def test_search_accepts_2d_query(self) -> None:
        store, _, vecs = _store("a", "b")
        results = store.search(vecs[0:1], top_k=2)  # shape (1, DIM)
        assert len(results) >= 1

    def test_search_score_threshold_filters(self) -> None:
        store, _, vecs = _store("a", "b", "c")
        results = store.search(vecs[0], top_k=3, score_threshold=0.999)
        # Only the exact match (score ≈ 1.0) should survive
        assert all(r.score >= 0.999 for r in results)

    def test_search_empty_store_returns_empty(self) -> None:
        store = FAISSStore(dim=DIM)
        results = store.search(_unit([1.0] + [0.0] * (DIM - 1)), top_k=5)
        assert results == []

    def test_search_chunk_content_matches(self) -> None:
        store, chunks, vecs = _store("alpha", "beta")
        results = store.search(vecs[0], top_k=1)
        assert results[0].chunk.id == "alpha"
        assert results[0].chunk.text == "text for alpha"


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_removes_chunk_from_results(self) -> None:
        store, _, vecs = _store("a", "b", "c")
        store.delete(["a"])
        ids = [r.chunk.id for r in store.search(vecs[0], top_k=3)]
        assert "a" not in ids

    def test_delete_reduces_index_size(self) -> None:
        store, _, _ = _store("a", "b", "c")
        store.delete(["b"])
        assert store._slot.index.ntotal == 2

    def test_delete_unknown_id_is_silent(self) -> None:
        store, _, _ = _store("a", "b")
        store.delete(["nonexistent"])  # must not raise
        assert store._slot.index.ntotal == 2

    def test_delete_multiple_ids(self) -> None:
        store, _, _ = _store("a", "b", "c", "d")
        store.delete(["a", "c"])
        assert store._slot.index.ntotal == 2

    def test_delete_cleans_up_dicts(self) -> None:
        store, _, _ = _store("x", "y")
        store.delete(["x"])
        assert "x" not in store._slot.id_map
        assert "x" not in {c.id for c in store._slot.chunks.values()}


# ---------------------------------------------------------------------------
# snapshot_distribution
# ---------------------------------------------------------------------------


class TestSnapshotDistribution:
    def test_empty_store_returns_zero_row_array(self) -> None:
        store = FAISSStore(dim=DIM)
        snap = store.snapshot_distribution()
        assert snap.shape == (0, DIM)
        assert snap.dtype == np.float32

    def test_shape_matches_added_chunks(self) -> None:
        store, _, _ = _store("a", "b", "c")
        snap = store.snapshot_distribution()
        assert snap.shape == (3, DIM)

    def test_snapshot_vectors_are_unit_norm(self) -> None:
        store, _, _ = _store("a", "b", "c")
        snap = store.snapshot_distribution()
        norms = np.linalg.norm(snap, axis=1)
        np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)

    def test_snapshot_excludes_deleted_chunks(self) -> None:
        store, _, _ = _store("a", "b", "c")
        store.delete(["b"])
        snap = store.snapshot_distribution()
        assert snap.shape == (2, DIM)


# ---------------------------------------------------------------------------
# swap_index
# ---------------------------------------------------------------------------


class TestSwapIndex:
    def test_swap_replaces_all_content(self) -> None:
        store, _, _ = _store("old_a", "old_b")
        new_chunks = [_make_chunk("new_x"), _make_chunk("new_y"), _make_chunk("new_z")]
        new_vecs = _random_unit_vecs(3, seed=99)
        store.swap_index(new_chunks, new_vecs)

        ids = {r.chunk.id for r in store.search(new_vecs[0], top_k=5)}
        assert "new_x" in ids
        assert "old_a" not in ids
        assert "old_b" not in ids

    def test_swap_index_size_matches_new_chunks(self) -> None:
        store, _, _ = _store("a", "b", "c", "d")
        new_chunks = [_make_chunk("x"), _make_chunk("y")]
        store.swap_index(new_chunks, _random_unit_vecs(2, seed=7))
        assert store._slot.index.ntotal == 2

    def test_swap_with_empty_set_clears_index(self) -> None:
        store, _, _ = _store("a", "b")
        store.swap_index([], np.empty((0, DIM), dtype=np.float32))
        assert store._slot.index.ntotal == 0

    def test_swap_uses_inactive_slot(self) -> None:
        """After swap the active pointer toggles."""
        store = FAISSStore(dim=DIM)
        before = store._active
        store.swap_index([_make_chunk("c")], _random_unit_vecs(1))
        assert store._active != before


# ---------------------------------------------------------------------------
# Thread safety (smoke test)
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_searches_do_not_crash(self) -> None:
        store, _, vecs = _store(*[f"c{i}" for i in range(20)])
        errors: list[Exception] = []

        def search_loop() -> None:
            try:
                for _ in range(50):
                    store.search(vecs[0], top_k=5)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=search_loop) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_add_and_search(self) -> None:
        store = FAISSStore(dim=DIM)
        errors: list[Exception] = []

        def add_loop() -> None:
            try:
                for i in range(20):
                    store.add([_make_chunk(f"t{i}")], _random_unit_vecs(1, seed=i))
            except Exception as exc:
                errors.append(exc)

        def search_loop(query: np.ndarray) -> None:
            try:
                for _ in range(50):
                    store.search(query, top_k=3)
            except Exception as exc:
                errors.append(exc)

        q = _random_unit_vecs(1)[0]
        threads: list[threading.Thread] = [threading.Thread(target=add_loop)]
        threads += [threading.Thread(target=search_loop, args=(q,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
