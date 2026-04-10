"""Server-Sent Events (SSE) streaming response helpers for generation output."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from rag.generation.llm import LLMRouter


async def stream_complete(
    router: LLMRouter,
    prompt: str,
    *,
    chunk_size: int = 1,
) -> AsyncGenerator[str, None]:
    """Wrap a synchronous :meth:`~rag.generation.llm.LLMRouter.complete` call as an
    async generator that yields token chunks suitable for Server-Sent Events.

    The blocking ``complete`` call is offloaded to a thread pool via
    :func:`asyncio.to_thread` so the event loop is never blocked.  The full
    response text is then split into *chunk_size*-word pieces and yielded one
    at a time, giving downstream SSE handlers a stream of incremental tokens.

    Args:
        router: Any :class:`~rag.generation.llm.LLMRouter` implementation.
        prompt: Fully-rendered prompt string to send to the LLM.
        chunk_size: Number of whitespace-delimited tokens to yield per
            iteration.  Defaults to ``1`` (word-by-word streaming).

    Yields:
        Non-empty string fragments of the completion, each ending with a
        trailing space except the final fragment.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    full_text: str = await asyncio.to_thread(router.complete, prompt)

    if not full_text:
        return

    words = full_text.split(" ")
    # Group words into chunks of chunk_size, preserving inter-word spaces.
    for i in range(0, len(words), chunk_size):
        piece = " ".join(words[i : i + chunk_size])
        if piece:
            yield piece
