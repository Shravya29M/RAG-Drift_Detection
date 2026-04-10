"""Unit tests for rag.generation.streaming."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag.generation.llm import LLMRouter
from rag.generation.streaming import stream_complete

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _router(response: str) -> LLMRouter:
    """Return a mock LLMRouter that synchronously returns *response*."""
    router = MagicMock(spec=LLMRouter)
    router.complete.return_value = response
    return router


async def _collect(router: LLMRouter, prompt: str, **kwargs: object) -> list[str]:
    """Drain *stream_complete* into a list."""
    chunks: list[str] = []
    async for token in stream_complete(router, prompt, **kwargs):  # type: ignore[arg-type]
        chunks.append(token)
    return chunks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_yields_chunks_for_multiword_response() -> None:
    """Generator must yield at least one chunk for a non-empty response."""
    router = _router("The answer is forty-two.")
    chunks = await _collect(router, "What is the answer?")
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)


@pytest.mark.anyio
async def test_reassembled_text_equals_original() -> None:
    """Joining all yielded chunks must reconstruct the full response."""
    text = "hello world this is a test"
    router = _router(text)
    chunks = await _collect(router, "q")
    assert " ".join(chunks) == text


@pytest.mark.anyio
async def test_empty_response_yields_nothing() -> None:
    router = _router("")
    chunks = await _collect(router, "q")
    assert chunks == []


@pytest.mark.anyio
async def test_single_word_response_yields_one_chunk() -> None:
    router = _router("yes")
    chunks = await _collect(router, "q")
    assert chunks == ["yes"]


@pytest.mark.anyio
async def test_chunk_size_groups_words() -> None:
    """chunk_size=2 should group pairs of words into each yielded token."""
    router = _router("one two three four")
    chunks = await _collect(router, "q", chunk_size=2)
    assert chunks == ["one two", "three four"]


@pytest.mark.anyio
async def test_chunk_size_larger_than_response_yields_one_chunk() -> None:
    router = _router("short")
    chunks = await _collect(router, "q", chunk_size=10)
    assert chunks == ["short"]


@pytest.mark.anyio
async def test_invalid_chunk_size_raises() -> None:
    router = _router("anything")
    with pytest.raises(ValueError, match="chunk_size"):
        async for _ in stream_complete(router, "q", chunk_size=0):
            pass


@pytest.mark.anyio
async def test_complete_called_with_prompt() -> None:
    """The underlying router.complete must be called with the supplied prompt."""
    router = _router("ok")
    await _collect(router, "my exact prompt")
    router.complete.assert_called_once_with("my exact prompt")  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_no_empty_string_chunks_yielded() -> None:
    """No yielded chunk should be an empty string."""
    router = _router("  spaced   out   words  ")
    chunks = await _collect(router, "q")
    assert all(c != "" for c in chunks)
