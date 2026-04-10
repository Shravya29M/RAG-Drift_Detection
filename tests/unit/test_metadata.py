"""Unit tests for rag.ingestion.metadata."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rag.ingestion.metadata import extract_section_header, infer_source_type, section_metadata
from rag.models import SourceType


def test_extract_section_header_returns_heading_text() -> None:
    """Heading markers and whitespace are stripped; only the title text is returned."""
    assert extract_section_header("# Introduction\nSome body text.") == "Introduction"
    assert extract_section_header("## Sub-section\nMore text.") == "Sub-section"
    assert extract_section_header("###### Deep\nbody") == "Deep"
    # Non-heading first line → None
    assert extract_section_header("Just a paragraph.") is None
    assert extract_section_header("") is None


def test_section_metadata_pdf_assigns_page_numbers() -> None:
    """PDF sections get 1-based page_number; no section_header is set."""
    pages = ["Page one text.", "Page two text.", "Page three text."]
    metas = section_metadata(pages, "report.pdf", SourceType.PDF)

    assert len(metas) == 3
    for i, meta in enumerate(metas):
        assert meta.page_number == i + 1, f"section {i}: expected page {i + 1}"
        assert meta.section_header is None
        assert meta.source == "report.pdf"
        assert meta.source_type == SourceType.PDF
        assert meta.chunk_index == i


def test_section_metadata_markdown_extracts_headers_and_shared_timestamp() -> None:
    """Markdown sections get section_header from the ATX heading; no page_number.
    All sections in one batch share a single ingested_at timestamp.
    """
    fixed_time = datetime(2026, 4, 10, 12, 0, 0)
    sections = [
        "# Overview\nThis project does X.",
        "## Details\nMore info here.",
        "No heading here, just text.",
    ]
    fp = Path("/data/docs/guide.md")
    metas = section_metadata(
        sections,
        "guide.md",
        SourceType.MARKDOWN,
        file_path=fp,
        ingested_at=fixed_time,
    )

    assert len(metas) == 3
    assert metas[0].section_header == "Overview"
    assert metas[1].section_header == "Details"
    assert metas[2].section_header is None  # no heading → None
    assert all(m.page_number is None for m in metas)
    assert all(m.ingested_at == fixed_time for m in metas)
    assert all(m.file_path == fp for m in metas)


def test_infer_source_type_maps_extensions() -> None:
    """Common extensions resolve to the expected SourceType."""
    cases = {
        ".pdf": SourceType.PDF,
        ".md": SourceType.MARKDOWN,
        ".markdown": SourceType.MARKDOWN,
        ".txt": SourceType.TEXT,
        ".text": SourceType.TEXT,
    }
    for ext, expected in cases.items():
        assert infer_source_type(Path(f"file{ext}")) == expected
