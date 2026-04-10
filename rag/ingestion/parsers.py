"""Document parsers: convert PDF, Markdown, plain text, and URLs to raw text."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import fitz  # type: ignore[import-untyped]  # PyMuPDF's fitz re-export has no py.typed marker
import httpx

from rag.models import SourceType

# ATX heading pattern: any line that starts with one or more '#'
_HEADING_RE = re.compile(r"(?m)^(?=#)")


class _HTMLTextExtractor(HTMLParser):
    """Strip HTML markup and collect visible text nodes."""

    _SKIP_TAGS: frozenset[str] = frozenset({"script", "style", "head", "meta", "link", "noscript"})

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    @property
    def text(self) -> str:
        """Joined visible text content from the parsed document."""
        return "\n".join(self._parts)


def parse_pdf(path: Path) -> list[str]:
    """Extract text from each page of a PDF file.

    Args:
        path: Path to the PDF file.

    Returns:
        One string per page in document order. Pages with no extractable text
        are included as empty strings so that page index aligns with page number.
    """
    doc = fitz.open(str(path))
    try:
        return [str(page.get_text()) for page in doc]
    finally:
        doc.close()


def parse_markdown(path: Path) -> list[str]:
    """Split a Markdown file into sections at ATX headings.

    Args:
        path: Path to the ``.md`` file.

    Returns:
        One string per section (heading line + body). Returns the entire file
        as a single-element list when no ATX headings are present.
    """
    content = path.read_text(encoding="utf-8")
    parts = _HEADING_RE.split(content)
    return [p.strip() for p in parts if p.strip()]


def parse_text(path: Path) -> list[str]:
    """Read a plain-text file as a single section.

    Args:
        path: Path to the ``.txt`` file.

    Returns:
        Single-element list with the full file content, or ``[]`` if the file
        is empty or whitespace-only.
    """
    content = path.read_text(encoding="utf-8").strip()
    return [content] if content else []


def parse_url(url: str, *, client: httpx.Client | None = None) -> list[str]:
    """Fetch a URL and extract its visible text content.

    Args:
        url: HTTP or HTTPS URL to retrieve.
        client: Optional pre-configured :class:`httpx.Client`. When ``None``
            a transient client is created and closed after the request. Inject
            an explicit client in tests to avoid real network calls.

    Returns:
        Single-element list with the stripped page text, or ``[]`` if the
        response body contains no visible text.

    Raises:
        httpx.HTTPStatusError: For 4xx / 5xx responses.
    """
    own_client = client is None
    if client is None:
        active: httpx.Client = httpx.Client(follow_redirects=True, timeout=10.0)
    else:
        active = client
    try:
        response = active.get(url)
        response.raise_for_status()
        extractor = _HTMLTextExtractor()
        extractor.feed(response.text)
        text = extractor.text.strip()
        return [text] if text else []
    finally:
        if own_client:
            active.close()


def parse(
    source: str,
    source_type: SourceType,
    *,
    http_client: httpx.Client | None = None,
) -> list[str]:
    """Dispatch *source* to the correct parser based on *source_type*.

    Args:
        source: File-system path (PDF / Markdown / text) or URL string.
        source_type: Determines which parser is invoked.
        http_client: Optional :class:`httpx.Client` forwarded to
            :func:`parse_url`; ignored for file-based sources.

    Returns:
        List of raw text strings, one per page or section.

    Raises:
        ValueError: If *source_type* is not a recognised :class:`SourceType`.
    """
    if source_type is SourceType.PDF:
        return parse_pdf(Path(source))
    if source_type is SourceType.MARKDOWN:
        return parse_markdown(Path(source))
    if source_type is SourceType.TEXT:
        return parse_text(Path(source))
    if source_type is SourceType.URL:
        return parse_url(source, client=http_client)
    raise ValueError(f"Unsupported source_type: {source_type!r}")  # pragma: no cover
