"""Sliding-window text chunker sized in embedding-model tokens.

With a tokenizer (the normal path — the API passes the encoder's), chunks are
measured in the model's own tokens so none exceeds what the model can embed:
all-MiniLM-L6-v2 truncates at 256 tokens, and anything past that is silently
dropped from the embedding. Without one (encoders that expose no tokenizer,
e.g. test doubles) chunks fall back to whitespace-delimited words.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.models import Chunk, ChunkMetadata, IngestConfig, SourceType


@dataclass(frozen=True)
class _Word:
    """A pre-tokenizer word: its character span and how many tokens it costs."""

    start: int
    end: int
    token_spans: tuple[tuple[int, int], ...]

    @property
    def n_tokens(self) -> int:
        return len(self.token_spans)


def _tokenize(text: str) -> list[str]:
    """Split text into whitespace-delimited tokens (fallback path)."""
    return text.split()


def _source_hash(source: str) -> str:
    """Return an 8-character hex digest of *source* for use in chunk IDs."""
    return hashlib.md5(source.encode(), usedforsecurity=False).hexdigest()[:8]


def count_tokens(tokenizer: Any, text: str) -> int:
    """Number of model tokens in *text*, excluding special tokens."""
    return len(tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])


def _words(tokenizer: Any, text: str) -> list[_Word]:
    """Group the tokenizer's tokens into pre-tokenizer words with char offsets."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    words: list[_Word] = []
    spans: list[tuple[int, int]] = []
    current: int | None = None
    for word_id, (start, end) in zip(enc.word_ids(), enc["offset_mapping"], strict=True):
        if word_id != current and spans:
            words.append(_Word(spans[0][0], spans[-1][1], tuple(spans)))
            spans = []
        current = word_id
        spans.append((start, end))
    if spans:
        words.append(_Word(spans[0][0], spans[-1][1], tuple(spans)))
    return words


def _split_long_word(tokenizer: Any, text: str, word: _Word, size: int) -> list[tuple[str, int]]:
    """Cut a single word longer than *size* tokens on token boundaries.

    A mid-word slice can re-tokenize differently (a ``##`` continuation becomes
    a word start), so each piece is re-measured and shrunk until it fits.
    """
    pieces: list[tuple[str, int]] = []
    i = 0
    spans = word.token_spans
    while i < len(spans):
        take = min(size, len(spans) - i)
        while True:
            piece = text[spans[i][0] : spans[i + take - 1][1]]
            n = count_tokens(tokenizer, piece)
            if n <= size or take == 1:
                break
            take -= 1
        pieces.append((piece, n))
        i += take
    return pieces


def _token_windows(tokenizer: Any, text: str, size: int, overlap: int) -> list[tuple[str, int]]:
    """Pack whole words into windows of at most *size* tokens.

    Consecutive windows share whole words worth at most *overlap* tokens.
    Each window is an exact slice of *text*, so re-tokenizing it gives the
    same count.
    """
    words = _words(tokenizer, text)
    windows: list[tuple[str, int]] = []
    i = 0
    while i < len(words):
        j, tokens = i, 0
        while j < len(words) and tokens + words[j].n_tokens <= size:
            tokens += words[j].n_tokens
            j += 1
        if j == i:  # one word alone exceeds the window
            windows.extend(_split_long_word(tokenizer, text, words[i], size))
            i += 1
            continue
        windows.append((text[words[i].start : words[j - 1].end], tokens))
        if j == len(words):
            break
        k, shared = j, 0
        while k > i + 1 and shared + words[k - 1].n_tokens <= overlap:
            k -= 1
            shared += words[k].n_tokens
        i = k
    return windows


def _word_windows(text: str, size: int, overlap: int) -> list[tuple[str, int]]:
    """Whitespace-word windows (fallback when no tokenizer is available)."""
    tokens = _tokenize(text)
    windows: list[tuple[str, int]] = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        windows.append((" ".join(tokens[start:end]), end - start))
        if end == len(tokens):
            break
        start += size - overlap
    return windows


def chunk_text(
    text: str,
    source: str,
    source_type: SourceType,
    config: IngestConfig,
    *,
    page_number: int | None = None,
    section_header: str | None = None,
    file_path: Path | None = None,
    tokenizer: Any | None = None,
    max_tokens: int | None = None,
) -> list[Chunk]:
    """Split *text* into overlapping chunks of at most ``config.chunk_size`` tokens.

    Args:
        text: Raw document text to split.
        source: Filename or URL that identifies the document.
        source_type: Format of the source document.
        config: Chunk size and overlap settings, in tokenizer tokens (or
            whitespace words when *tokenizer* is ``None``).
        page_number: 1-based PDF page number; ``None`` for non-PDF sources.
        section_header: Nearest heading above this text, if extractable.
        file_path: Absolute path to the source file; ``None`` for URL sources.
        tokenizer: Hugging Face fast tokenizer of the embedding model.
        max_tokens: The model's input limit excluding special tokens; a
            ``chunk_size`` above it is rejected rather than silently truncated
            at embedding time.

    Returns:
        Ordered list of :class:`~rag.models.Chunk` objects with metadata attached.

    Raises:
        ValueError: If ``config.chunk_overlap >= config.chunk_size``, or
            ``config.chunk_size > max_tokens``.
    """
    if config.chunk_size - config.chunk_overlap <= 0:
        raise ValueError(
            f"chunk_overlap ({config.chunk_overlap}) must be less than "
            f"chunk_size ({config.chunk_size})"
        )
    if max_tokens is not None and config.chunk_size > max_tokens:
        raise ValueError(
            f"chunk_size ({config.chunk_size}) exceeds the embedding model's "
            f"input limit ({max_tokens} tokens); the excess would never be embedded"
        )

    if tokenizer is not None:
        windows = _token_windows(tokenizer, text, config.chunk_size, config.chunk_overlap)
    else:
        windows = _word_windows(text, config.chunk_size, config.chunk_overlap)

    src_hash = _source_hash(source)
    return [
        Chunk(
            id=f"{src_hash}-{index}",
            text=chunk_str,
            token_count=n_tokens,
            metadata=ChunkMetadata(
                source=source,
                source_type=source_type,
                page_number=page_number,
                section_header=section_header,
                chunk_index=index,
                file_path=file_path,
            ),
        )
        for index, (chunk_str, n_tokens) in enumerate(windows)
    ]
