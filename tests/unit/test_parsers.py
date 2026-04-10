"""Unit tests for rag.ingestion.parsers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz  # type: ignore[import-untyped]
import httpx
import pytest

from rag.ingestion.parsers import (
    _HTMLTextExtractor,
    parse,
    parse_markdown,
    parse_pdf,
    parse_text,
    parse_url,
)
from rag.models import SourceType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf(path: Path, pages: list[str]) -> None:
    """Write a minimal PDF with one text layer per page to *path*."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def _mock_http_client(html: str, status_code: int = 200) -> MagicMock:
    """Return an httpx.Client mock that responds with *html*."""
    response = MagicMock(spec=httpx.Response)
    response.text = html
    response.status_code = status_code
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="error", request=MagicMock(), response=response
        )
    else:
        response.raise_for_status.return_value = None

    client = MagicMock(spec=httpx.Client)
    client.get.return_value = response
    return client


# ---------------------------------------------------------------------------
# _HTMLTextExtractor
# ---------------------------------------------------------------------------


class TestHTMLTextExtractor:
    def test_extracts_visible_text(self) -> None:
        ex = _HTMLTextExtractor()
        ex.feed("<html><body><h1>Hello</h1><p>World</p></body></html>")
        assert "Hello" in ex.text
        assert "World" in ex.text

    def test_skips_script_content(self) -> None:
        ex = _HTMLTextExtractor()
        ex.feed("<html><body><script>var x = 1;</script><p>Visible</p></body></html>")
        assert "var x" not in ex.text
        assert "Visible" in ex.text

    def test_skips_style_content(self) -> None:
        ex = _HTMLTextExtractor()
        ex.feed("<html><head><style>body { color: red; }</style></head><body>Text</body></html>")
        assert "color" not in ex.text
        assert "Text" in ex.text

    def test_skips_noscript_content(self) -> None:
        ex = _HTMLTextExtractor()
        ex.feed("<body><noscript>Enable JS</noscript><p>Content</p></body>")
        assert "Enable JS" not in ex.text
        assert "Content" in ex.text

    def test_empty_html_returns_empty_text(self) -> None:
        ex = _HTMLTextExtractor()
        ex.feed("<html><body></body></html>")
        assert ex.text == ""

    def test_whitespace_only_nodes_ignored(self) -> None:
        ex = _HTMLTextExtractor()
        ex.feed("<p>  </p><p>Real</p>")
        assert ex.text.strip() == "Real"

    def test_multiple_paragraphs_joined(self) -> None:
        """Text from multiple elements is joined with newlines."""
        ex = _HTMLTextExtractor()
        ex.feed("<p>First</p><p>Second</p><p>Third</p>")
        lines = ex.text.splitlines()
        assert lines == ["First", "Second", "Third"]


# ---------------------------------------------------------------------------
# parse_pdf
# ---------------------------------------------------------------------------


class TestParsePdf:
    def test_returns_one_string_per_page(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        _make_pdf(pdf, ["Page one text", "Page two text", "Page three text"])
        result = parse_pdf(pdf)
        assert len(result) == 3

    def test_page_text_content(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        _make_pdf(pdf, ["Hello world"])
        result = parse_pdf(pdf)
        assert "Hello" in result[0]
        assert "world" in result[0]

    def test_all_elements_are_strings(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        _make_pdf(pdf, ["A", "B"])
        result = parse_pdf(pdf)
        assert all(isinstance(s, str) for s in result)

    def test_empty_pdf_returns_empty_strings(self, tmp_path: Path) -> None:
        """A PDF with blank pages returns empty strings, not errors."""
        doc = fitz.open()
        doc.new_page()  # blank page — no text inserted
        pdf = tmp_path / "blank.pdf"
        doc.save(str(pdf))
        doc.close()
        result = parse_pdf(pdf)
        assert len(result) == 1
        assert result[0].strip() == ""

    def test_multipage_preserves_order(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pages = [f"unique_token_page_{i}" for i in range(5)]
        _make_pdf(pdf, pages)
        result = parse_pdf(pdf)
        for i, token in enumerate(pages):
            assert token in result[i]


# ---------------------------------------------------------------------------
# parse_markdown
# ---------------------------------------------------------------------------


class TestParseMarkdown:
    def test_splits_on_h1(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Section A\nContent A\n\n# Section B\nContent B\n", encoding="utf-8")
        result = parse_markdown(md)
        assert len(result) == 2
        assert "Section A" in result[0]
        assert "Section B" in result[1]

    def test_splits_on_mixed_heading_levels(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# H1\nbody\n## H2\nbody2\n### H3\nbody3\n", encoding="utf-8")
        result = parse_markdown(md)
        assert len(result) == 3

    def test_no_headings_returns_single_section(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("Just plain text\nwith no headings.\n", encoding="utf-8")
        result = parse_markdown(md)
        assert len(result) == 1
        assert "plain text" in result[0]

    def test_section_includes_heading_and_body(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Title\nBody paragraph.\n", encoding="utf-8")
        result = parse_markdown(md)
        assert "# Title" in result[0]
        assert "Body paragraph" in result[0]

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        md = tmp_path / "empty.md"
        md.write_text("", encoding="utf-8")
        result = parse_markdown(md)
        assert result == []

    def test_whitespace_only_file_returns_empty_list(self, tmp_path: Path) -> None:
        md = tmp_path / "ws.md"
        md.write_text("   \n\n\t\n", encoding="utf-8")
        result = parse_markdown(md)
        assert result == []

    def test_content_before_first_heading_is_included(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("Preamble text.\n\n# First section\nBody.\n", encoding="utf-8")
        result = parse_markdown(md)
        # Preamble becomes its own section (no heading), then one more for # First
        assert any("Preamble" in s for s in result)
        assert any("First section" in s for s in result)


# ---------------------------------------------------------------------------
# parse_text
# ---------------------------------------------------------------------------


class TestParseText:
    def test_returns_single_element_list(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("Hello world", encoding="utf-8")
        result = parse_text(f)
        assert len(result) == 1
        assert result[0] == "Hello world"

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        assert parse_text(f) == []

    def test_whitespace_only_returns_empty_list(self, tmp_path: Path) -> None:
        f = tmp_path / "ws.txt"
        f.write_text("  \n\t  \n", encoding="utf-8")
        assert parse_text(f) == []

    def test_multiline_content_preserved(self, tmp_path: Path) -> None:
        content = "line one\nline two\nline three"
        f = tmp_path / "multi.txt"
        f.write_text(content, encoding="utf-8")
        result = parse_text(f)
        assert result[0] == content

    def test_leading_trailing_whitespace_stripped(self, tmp_path: Path) -> None:
        f = tmp_path / "padded.txt"
        f.write_text("\n  Hello  \n", encoding="utf-8")
        assert parse_text(f)[0] == "Hello"


# ---------------------------------------------------------------------------
# parse_url
# ---------------------------------------------------------------------------


class TestParseUrl:
    def test_returns_visible_text(self) -> None:
        client = _mock_http_client("<html><body><p>Hello from the web</p></body></html>")
        result = parse_url("https://example.com", client=client)
        assert len(result) == 1
        assert "Hello from the web" in result[0]

    def test_strips_script_and_style(self) -> None:
        html = (
            "<html><head><style>.x{}</style></head>"
            "<body><script>alert(1)</script><p>Clean</p></body></html>"
        )
        client = _mock_http_client(html)
        result = parse_url("https://example.com", client=client)
        assert ".x" not in result[0]
        assert "alert" not in result[0]
        assert "Clean" in result[0]

    def test_empty_page_returns_empty_list(self) -> None:
        client = _mock_http_client("<html><body></body></html>")
        result = parse_url("https://example.com", client=client)
        assert result == []

    def test_http_error_raises(self) -> None:
        client = _mock_http_client("Not Found", status_code=404)
        with pytest.raises(httpx.HTTPStatusError):
            parse_url("https://example.com/missing", client=client)

    def test_client_get_called_with_url(self) -> None:
        client = _mock_http_client("<p>Hi</p>")
        parse_url("https://example.com/page", client=client)
        client.get.assert_called_once_with("https://example.com/page")

    def test_injected_client_not_closed(self) -> None:
        """Caller-supplied clients must not be closed by parse_url."""
        client = _mock_http_client("<p>data</p>")
        parse_url("https://example.com", client=client)
        client.close.assert_not_called()

    def test_owned_client_is_closed(self) -> None:
        """When parse_url creates its own client it must close it."""
        mock_client = _mock_http_client("<p>data</p>")
        with patch("rag.ingestion.parsers.httpx.Client", return_value=mock_client):
            parse_url("https://example.com")
        mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# parse dispatch
# ---------------------------------------------------------------------------


class TestParseDispatch:
    def test_dispatches_pdf(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        _make_pdf(pdf, ["content"])
        result = parse(str(pdf), SourceType.PDF)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_dispatches_markdown(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Title\nbody\n", encoding="utf-8")
        result = parse(str(md), SourceType.MARKDOWN)
        assert len(result) == 1

    def test_dispatches_text(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("hello", encoding="utf-8")
        result = parse(str(f), SourceType.TEXT)
        assert result == ["hello"]

    def test_dispatches_url(self) -> None:
        mock_client = _mock_http_client("<p>web</p>")
        result = parse("https://example.com", SourceType.URL, http_client=mock_client)
        assert "web" in result[0]

    def test_all_results_are_list_of_str(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("words here", encoding="utf-8")
        result = parse(str(f), SourceType.TEXT)
        assert all(isinstance(s, str) for s in result)
