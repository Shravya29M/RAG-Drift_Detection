"""Metadata extraction and DocumentMeta schema for ingested document chunks."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from rag.models import ChunkMetadata, SourceType

# Matches the heading text in an ATX heading line: "## Some Title" → "Some Title"
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)")

# Maps file-system extensions to SourceType
_EXT_TO_SOURCE_TYPE: dict[str, SourceType] = {
    ".pdf": SourceType.PDF,
    ".md": SourceType.MARKDOWN,
    ".markdown": SourceType.MARKDOWN,
    ".txt": SourceType.TEXT,
    ".text": SourceType.TEXT,
}


def extract_section_header(text: str) -> str | None:
    """Return the heading text from the first non-empty line if it is an ATX heading.

    Args:
        text: Raw text of a single Markdown section (heading + body).

    Returns:
        Stripped heading content (without ``#`` prefix), or ``None`` if the
        first line is not an ATX heading.
    """
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    match = _HEADING_RE.match(first_line)
    return match.group(1).strip() if match else None


def infer_source_type(path: Path) -> SourceType:
    """Map a file-system path's extension to a :class:`~rag.models.SourceType`.

    Args:
        path: Path to the source file.

    Returns:
        Matching :class:`~rag.models.SourceType`.

    Raises:
        ValueError: If the extension is not recognised.
    """
    ext = path.suffix.lower()
    try:
        return _EXT_TO_SOURCE_TYPE[ext]
    except KeyError:
        raise ValueError(
            f"Cannot infer source type from extension {ext!r}. "
            f"Supported: {sorted(_EXT_TO_SOURCE_TYPE)}"
        ) from None


def section_metadata(
    sections: list[str],
    source: str,
    source_type: SourceType,
    *,
    file_path: Path | None = None,
    ingested_at: datetime | None = None,
) -> list[ChunkMetadata]:
    """Build one :class:`~rag.models.ChunkMetadata` per section or page.

    Intended to bridge raw parser output (``list[str]``) into typed metadata
    before the chunker splits each section into token windows.

    - **PDF**: ``page_number`` is set to ``index + 1`` (1-based).
    - **Markdown**: ``section_header`` is extracted from the first line of each
      section via :func:`extract_section_header`.
    - **Text / URL**: neither field is populated.
    - ``ingested_at`` is snapped once at call time and shared across all
      sections in the batch so that a single ingest run has a consistent stamp.
    - ``chunk_index`` is set to the section index; the chunker will assign
      final per-chunk indexes when it splits section text into windows.

    Args:
        sections: Ordered list of section/page strings from a parser function.
        source: Filename or URL identifying the originating document.
        source_type: Format that determines how metadata fields are populated.
        file_path: Absolute path on disk; ``None`` for URL sources.
        ingested_at: Override the ingest timestamp (useful in tests). Defaults
            to ``datetime.utcnow()`` at the moment of the call.

    Returns:
        List of :class:`~rag.models.ChunkMetadata`, one per section, in order.
    """
    now = ingested_at if ingested_at is not None else datetime.utcnow()
    result: list[ChunkMetadata] = []

    for i, text in enumerate(sections):
        page_number = (i + 1) if source_type is SourceType.PDF else None
        header = extract_section_header(text) if source_type is SourceType.MARKDOWN else None

        result.append(
            ChunkMetadata(
                source=source,
                source_type=source_type,
                page_number=page_number,
                section_header=header,
                chunk_index=i,
                ingested_at=now,
                file_path=file_path,
            )
        )

    return result
