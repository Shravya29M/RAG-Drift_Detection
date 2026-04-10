"""Unit tests for rag.ingestion.chunker."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from rag.ingestion.chunker import chunk_text
from rag.models import IngestConfig, SourceType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg() -> IngestConfig:
    """Small config so tests don't need large text inputs."""
    return IngestConfig(chunk_size=5, chunk_overlap=2)


def _words(n: int) -> str:
    """Return a string of *n* space-separated word tokens: 'w0 w1 ... w{n-1}'."""
    return " ".join(f"w{i}" for i in range(n))


# ---------------------------------------------------------------------------
# Chunk count
# ---------------------------------------------------------------------------


def test_chunk_count_exact_multiple(cfg: IngestConfig) -> None:
    """Token count aligns perfectly with step boundaries."""
    # N=9, size=5, overlap=2 → step=3 → starts: 0,3,6 → 3 chunks
    chunks = chunk_text(_words(9), "doc.txt", SourceType.TEXT, cfg)
    assert len(chunks) == 3


def test_chunk_count_formula(cfg: IngestConfig) -> None:
    """chunk count == ceil((N - overlap) / step) for various N."""
    step = cfg.chunk_size - cfg.chunk_overlap
    for n in range(1, 25):
        expected = math.ceil((n - cfg.chunk_overlap) / step) if n > cfg.chunk_overlap else 1
        # Simpler: expected = ceil((n - overlap) / step), but clamp to 1 min
        # Re-derive properly:
        expected = 0
        start = 0
        while start < n:
            end = min(start + cfg.chunk_size, n)
            expected += 1
            if end == n:
                break
            start += step
        chunks = chunk_text(_words(n), "doc.txt", SourceType.TEXT, cfg)
        assert len(chunks) == expected, f"N={n}: expected {expected}, got {len(chunks)}"


def test_chunk_count_shorter_than_window(cfg: IngestConfig) -> None:
    """Text shorter than chunk_size produces exactly one chunk."""
    chunks = chunk_text(_words(3), "doc.txt", SourceType.TEXT, cfg)
    assert len(chunks) == 1


def test_chunk_count_exact_one_window(cfg: IngestConfig) -> None:
    """Text exactly equal to chunk_size produces exactly one chunk."""
    chunks = chunk_text(_words(cfg.chunk_size), "doc.txt", SourceType.TEXT, cfg)
    assert len(chunks) == 1


def test_empty_text_returns_no_chunks(cfg: IngestConfig) -> None:
    chunks = chunk_text("", "doc.txt", SourceType.TEXT, cfg)
    assert chunks == []


def test_whitespace_only_returns_no_chunks(cfg: IngestConfig) -> None:
    chunks = chunk_text("   \n\t  ", "doc.txt", SourceType.TEXT, cfg)
    assert chunks == []


# ---------------------------------------------------------------------------
# Overlap correctness
# ---------------------------------------------------------------------------


def test_overlap_tokens_match_between_consecutive_chunks(cfg: IngestConfig) -> None:
    """Last *overlap* tokens of chunk i must equal first *overlap* tokens of chunk i+1."""
    chunks = chunk_text(_words(20), "doc.txt", SourceType.TEXT, cfg)
    overlap = cfg.chunk_overlap

    for i in range(len(chunks) - 1):
        tail = chunks[i].text.split()[-overlap:]
        head = chunks[i + 1].text.split()[:overlap]
        assert tail == head, f"Overlap mismatch between chunk {i} and {i + 1}: {tail!r} != {head!r}"


def test_overlap_zero_means_no_shared_tokens() -> None:
    """With overlap=0 adjacent chunks share no tokens."""
    cfg = IngestConfig(chunk_size=4, chunk_overlap=0)
    chunks = chunk_text(_words(12), "doc.txt", SourceType.TEXT, cfg)
    assert len(chunks) == 3
    for i in range(len(chunks) - 1):
        tokens_i = set(chunks[i].text.split())
        tokens_next = set(chunks[i + 1].text.split())
        assert tokens_i.isdisjoint(tokens_next)


def test_full_text_coverage(cfg: IngestConfig) -> None:
    """Every token from the source text appears in at least one chunk."""
    n = 15
    words = [f"w{i}" for i in range(n)]
    chunks = chunk_text(" ".join(words), "doc.txt", SourceType.TEXT, cfg)
    seen: set[str] = set()
    for chunk in chunks:
        seen.update(chunk.text.split())
    assert seen == set(words)


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


def test_metadata_source_preserved(cfg: IngestConfig) -> None:
    chunks = chunk_text(_words(10), "my_doc.pdf", SourceType.PDF, cfg)
    for chunk in chunks:
        assert chunk.metadata.source == "my_doc.pdf"


def test_metadata_source_type_preserved(cfg: IngestConfig) -> None:
    chunks = chunk_text(_words(10), "readme.md", SourceType.MARKDOWN, cfg)
    for chunk in chunks:
        assert chunk.metadata.source_type == SourceType.MARKDOWN


def test_metadata_page_number_preserved(cfg: IngestConfig) -> None:
    chunks = chunk_text(_words(10), "doc.pdf", SourceType.PDF, cfg, page_number=3)
    for chunk in chunks:
        assert chunk.metadata.page_number == 3


def test_metadata_section_header_preserved(cfg: IngestConfig) -> None:
    chunks = chunk_text(
        _words(10), "doc.md", SourceType.MARKDOWN, cfg, section_header="Introduction"
    )
    for chunk in chunks:
        assert chunk.metadata.section_header == "Introduction"


def test_metadata_file_path_preserved(cfg: IngestConfig) -> None:
    p = Path("/data/docs/report.txt")
    chunks = chunk_text(_words(10), "report.txt", SourceType.TEXT, cfg, file_path=p)
    for chunk in chunks:
        assert chunk.metadata.file_path == p


def test_metadata_chunk_index_sequential(cfg: IngestConfig) -> None:
    """chunk_index must match the chunk's position in the returned list."""
    chunks = chunk_text(_words(20), "doc.txt", SourceType.TEXT, cfg)
    for i, chunk in enumerate(chunks):
        assert chunk.metadata.chunk_index == i


def test_metadata_defaults_are_none(cfg: IngestConfig) -> None:
    """Optional metadata fields default to None when not supplied."""
    chunks = chunk_text(_words(5), "doc.txt", SourceType.TEXT, cfg)
    assert chunks[0].metadata.page_number is None
    assert chunks[0].metadata.section_header is None
    assert chunks[0].metadata.file_path is None


# ---------------------------------------------------------------------------
# ID stability
# ---------------------------------------------------------------------------


def test_chunk_ids_are_unique(cfg: IngestConfig) -> None:
    chunks = chunk_text(_words(20), "doc.txt", SourceType.TEXT, cfg)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_id_format(cfg: IngestConfig) -> None:
    """ID must be '<8-char-hex>-<index>'."""
    chunks = chunk_text(_words(10), "doc.txt", SourceType.TEXT, cfg)
    for i, chunk in enumerate(chunks):
        prefix, _, suffix = chunk.id.rpartition("-")
        assert len(prefix) == 8 and all(c in "0123456789abcdef" for c in prefix)
        assert suffix == str(i)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_overlap_gte_chunk_size_raises() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        cfg = IngestConfig(chunk_size=4, chunk_overlap=4)
        chunk_text(_words(10), "doc.txt", SourceType.TEXT, cfg)


def test_token_count_matches_chunk_length(cfg: IngestConfig) -> None:
    """token_count on each Chunk must equal the actual word count of its text."""
    chunks = chunk_text(_words(17), "doc.txt", SourceType.TEXT, cfg)
    for chunk in chunks:
        assert chunk.token_count == len(chunk.text.split())
