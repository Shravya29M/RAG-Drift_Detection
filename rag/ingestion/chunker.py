"""Sliding-window text chunker with configurable chunk size and overlap."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rag.models import Chunk, ChunkMetadata, IngestConfig, SourceType


def _tokenize(text: str) -> list[str]:
    """Split text into whitespace-delimited tokens."""
    return text.split()


def _source_hash(source: str) -> str:
    """Return an 8-character hex digest of *source* for use in chunk IDs."""
    return hashlib.md5(source.encode(), usedforsecurity=False).hexdigest()[:8]


def chunk_text(
    text: str,
    source: str,
    source_type: SourceType,
    config: IngestConfig,
    *,
    page_number: int | None = None,
    section_header: str | None = None,
    file_path: Path | None = None,
) -> list[Chunk]:
    """Split *text* into overlapping token-window chunks.

    Args:
        text: Raw document text to split.
        source: Filename or URL that identifies the document.
        source_type: Format of the source document.
        config: Chunk size and overlap settings.
        page_number: 1-based PDF page number; ``None`` for non-PDF sources.
        section_header: Nearest heading above this text, if extractable.
        file_path: Absolute path to the source file; ``None`` for URL sources.

    Returns:
        Ordered list of :class:`~rag.models.Chunk` objects with metadata attached.

    Raises:
        ValueError: If ``config.chunk_overlap >= config.chunk_size``.
    """
    step = config.chunk_size - config.chunk_overlap
    if step <= 0:
        raise ValueError(
            f"chunk_overlap ({config.chunk_overlap}) must be less than "
            f"chunk_size ({config.chunk_size})"
        )

    tokens = _tokenize(text)
    if not tokens:
        return []

    src_hash = _source_hash(source)
    chunks: list[Chunk] = []
    start = 0

    while start < len(tokens):
        end = min(start + config.chunk_size, len(tokens))
        window = tokens[start:end]
        chunk_text_str = " ".join(window)
        index = len(chunks)

        chunks.append(
            Chunk(
                id=f"{src_hash}-{index}",
                text=chunk_text_str,
                token_count=len(window),
                metadata=ChunkMetadata(
                    source=source,
                    source_type=source_type,
                    page_number=page_number,
                    section_header=section_header,
                    chunk_index=index,
                    file_path=file_path,
                ),
            )
        )

        if end == len(tokens):
            break
        start += step

    return chunks
