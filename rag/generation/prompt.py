"""Prompt template builder: injects retrieved context chunks into the query prompt."""

from __future__ import annotations

from rag.models import Chunk, GenerationConfig


def build_prompt(
    query: str,
    chunks: list[Chunk],
    config: GenerationConfig,
) -> str:
    """Render the generation prompt by injecting *query* and *chunks* into the template.

    Chunk texts are joined with a blank-line separator and placed into the
    ``{context}`` slot of ``config.prompt_template``. When *chunks* is empty the
    context block is replaced with the literal string ``"(no context retrieved)"``.

    Args:
        query: The user's natural-language question.
        chunks: Retrieved chunks to use as context (may be empty).
        config: Generation config supplying the prompt template.

    Returns:
        Fully rendered prompt string ready to send to the LLM.

    Raises:
        KeyError: If ``config.prompt_template`` is missing the ``{context}``
            or ``{query}`` placeholder.
    """
    if chunks:
        context = "\n\n".join(
            f"[{i + 1}] (source: {c.metadata.source})\n{c.text}" for i, c in enumerate(chunks)
        )
    else:
        context = "(no context retrieved)"

    try:
        return config.prompt_template.format(context=context, query=query)
    except KeyError as exc:
        raise KeyError(
            f"Prompt template is missing the {{{exc.args[0]}}} placeholder. "
            "Template must contain both '{context}' and '{query}'."
        ) from exc
