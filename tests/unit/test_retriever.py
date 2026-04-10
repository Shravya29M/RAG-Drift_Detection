"""Unit tests for rag.retrieval.retriever."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from rag.models import (
    Chunk,
    ChunkMetadata,
    RetrievalResult,
    SearchResult,
    SourceType,
)
from rag.retrieval.retriever import Retriever, _apply_filters

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 4


def _meta(**kwargs: object) -> ChunkMetadata:
    base: dict[str, object] = {
        "source": "doc.txt",
        "source_type": SourceType.TEXT,
        "chunk_index": 0,
        "ingested_at": datetime(2026, 1, 1),
    }
    base.update(kwargs)
    return ChunkMetadata(**base)  # type: ignore[arg-type]


def _chunk(cid: str, **meta_kwargs: object) -> Chunk:
    return Chunk(id=cid, text=f"text-{cid}", token_count=2, metadata=_meta(**meta_kwargs))


def _sr(chunk: Chunk, score: float = 0.9, rank: int = 0) -> SearchResult:
    return SearchResult(chunk=chunk, score=score, rank=rank)


def _unit_vec(dim: int = DIM) -> np.ndarray:
    v = np.ones(dim, dtype=np.float32)
    return v / float(np.linalg.norm(v))


def _make_retriever(
    search_results: list[SearchResult],
    *,
    score_threshold: float = 0.0,
) -> tuple[Retriever, MagicMock, MagicMock]:
    """Return (retriever, mock_store, mock_encoder) pre-wired with *search_results*."""
    enc = MagicMock()
    enc.encode.return_value = np.array([_unit_vec()], dtype=np.float32)

    store = MagicMock()
    store.search.return_value = search_results

    retriever = Retriever(store, enc, score_threshold=score_threshold)
    return retriever, store, enc


# ---------------------------------------------------------------------------
# _apply_filters (pure function)
# ---------------------------------------------------------------------------


class TestApplyFilters:
    def test_empty_filters_returns_all(self) -> None:
        results = [_sr(_chunk("a")), _sr(_chunk("b"))]
        assert _apply_filters(results, {}) == results

    def test_single_field_match(self) -> None:
        pdf = _sr(_chunk("pdf", source_type=SourceType.PDF))
        txt = _sr(_chunk("txt", source_type=SourceType.TEXT))
        out = _apply_filters([pdf, txt], {"source_type": SourceType.PDF})
        assert len(out) == 1
        assert out[0].chunk.id == "pdf"

    def test_multiple_fields_and_semantics(self) -> None:
        match = _sr(_chunk("m", source="a.pdf", source_type=SourceType.PDF))
        wrong_src = _sr(_chunk("w1", source="b.pdf", source_type=SourceType.PDF))
        wrong_type = _sr(_chunk("w2", source="a.pdf", source_type=SourceType.TEXT))
        out = _apply_filters(
            [match, wrong_src, wrong_type],
            {"source": "a.pdf", "source_type": SourceType.PDF},
        )
        assert [r.chunk.id for r in out] == ["m"]

    def test_unknown_filter_key_excludes_result(self) -> None:
        results = [_sr(_chunk("a"))]
        assert _apply_filters(results, {"nonexistent_field": "value"}) == []

    def test_preserves_original_order(self) -> None:
        chunks = [_chunk(f"c{i}", source=f"doc{i}.txt") for i in range(5)]
        results = [_sr(c) for c in chunks]
        out = _apply_filters(results, {})
        assert [r.chunk.id for r in out] == [f"c{i}" for i in range(5)]

    def test_none_value_filter(self) -> None:
        with_page = _sr(_chunk("p", page_number=1))
        without_page = _sr(_chunk("n"))  # page_number defaults to None
        out = _apply_filters([with_page, without_page], {"page_number": None})
        assert len(out) == 1
        assert out[0].chunk.id == "n"


# ---------------------------------------------------------------------------
# Retriever.retrieve — interface contracts
# ---------------------------------------------------------------------------


class TestRetrieverReturns:
    def test_returns_retrieval_result(self) -> None:
        ret, _, _ = _make_retriever([_sr(_chunk("a"))])
        result = ret.retrieve("hello", k=5)
        assert isinstance(result, RetrievalResult)

    def test_query_preserved_in_result(self) -> None:
        ret, _, _ = _make_retriever([])
        result = ret.retrieve("find me something", k=3)
        assert result.query == "find me something"

    def test_latency_ms_is_non_negative(self) -> None:
        ret, _, _ = _make_retriever([])
        result = ret.retrieve("q", k=1)
        assert result.latency_ms >= 0.0

    def test_chunks_and_scores_same_length(self) -> None:
        results = [_sr(_chunk(f"c{i}"), score=0.9 - i * 0.1) for i in range(3)]
        ret, _, _ = _make_retriever(results)
        r = ret.retrieve("q", k=5)
        assert len(r.chunks) == len(r.scores)

    def test_total_candidates_equals_raw_search_count(self) -> None:
        raw = [_sr(_chunk(f"c{i}")) for i in range(7)]
        ret, _, _ = _make_retriever(raw)
        r = ret.retrieve("q", k=3)
        assert r.total_candidates == 7


# ---------------------------------------------------------------------------
# Retriever.retrieve — encoder interaction
# ---------------------------------------------------------------------------


class TestRetrieverEncoder:
    def test_encodes_query_string(self) -> None:
        ret, _, enc = _make_retriever([])
        ret.retrieve("my query", k=3)
        enc.encode.assert_called_once_with(["my query"])

    def test_passes_first_row_of_encoded_output_to_store(self) -> None:
        query_vec = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        enc = MagicMock()
        enc.encode.return_value = query_vec
        store = MagicMock()
        store.search.return_value = []
        Retriever(store, enc).retrieve("q", k=3)
        call_vec: np.ndarray = store.search.call_args[0][0]
        np.testing.assert_array_equal(call_vec, query_vec[0])


# ---------------------------------------------------------------------------
# Retriever.retrieve — store interaction
# ---------------------------------------------------------------------------


class TestRetrieverStore:
    def test_search_called_with_correct_k_no_filters(self) -> None:
        ret, store, _ = _make_retriever([])
        ret.retrieve("q", k=5)
        store.search.assert_called_once()
        assert store.search.call_args[0][1] == 5  # no oversample without filters

    def test_search_oversamples_when_filters_set(self) -> None:
        ret, store, _ = _make_retriever([])
        ret.retrieve("q", k=5, filters={"source_type": SourceType.PDF})
        called_k = store.search.call_args[0][1]
        assert called_k == 5 * 4  # _OVERSAMPLE_FACTOR = 4

    def test_score_threshold_forwarded_to_store(self) -> None:
        ret, store, _ = _make_retriever([], score_threshold=0.75)
        ret.retrieve("q", k=3)
        _, kwargs = store.search.call_args
        assert kwargs["score_threshold"] == 0.75


# ---------------------------------------------------------------------------
# Retriever.retrieve — trimming and filtering
# ---------------------------------------------------------------------------


class TestRetrieverTrimAndFilter:
    def test_trims_to_k(self) -> None:
        raw = [_sr(_chunk(f"c{i}"), rank=i) for i in range(10)]
        ret, _, _ = _make_retriever(raw)
        r = ret.retrieve("q", k=3)
        assert len(r.chunks) == 3

    def test_fewer_results_than_k_returns_all(self) -> None:
        raw = [_sr(_chunk("only"))]
        ret, _, _ = _make_retriever(raw)
        r = ret.retrieve("q", k=10)
        assert len(r.chunks) == 1

    def test_empty_store_returns_empty_chunks(self) -> None:
        ret, _, _ = _make_retriever([])
        r = ret.retrieve("q", k=5)
        assert r.chunks == []
        assert r.scores == []

    def test_filter_excludes_non_matching(self) -> None:
        pdf = _sr(_chunk("pdf_chunk", source_type=SourceType.PDF))
        txt = _sr(_chunk("txt_chunk", source_type=SourceType.TEXT))
        ret, store, _ = _make_retriever([pdf, txt])
        r = ret.retrieve("q", k=5, filters={"source_type": SourceType.PDF})
        assert all(c.metadata.source_type == SourceType.PDF for c in r.chunks)
        assert "txt_chunk" not in [c.id for c in r.chunks]

    def test_scores_match_filtered_chunks(self) -> None:
        pdf = _sr(_chunk("p", source_type=SourceType.PDF), score=0.95)
        txt = _sr(_chunk("t", source_type=SourceType.TEXT), score=0.80)
        ret, _, _ = _make_retriever([pdf, txt])
        r = ret.retrieve("q", k=5, filters={"source_type": SourceType.PDF})
        assert r.scores == [pytest.approx(0.95)]

    def test_order_preserved_from_store(self) -> None:
        raw = [_sr(_chunk(f"c{i}"), score=1.0 - i * 0.1, rank=i) for i in range(5)]
        ret, _, _ = _make_retriever(raw)
        r = ret.retrieve("q", k=5)
        assert [c.id for c in r.chunks] == ["c0", "c1", "c2", "c3", "c4"]
