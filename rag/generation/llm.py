"""Abstract LLMRouter interface and OpenAI/Anthropic/Groq concrete implementations."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import anthropic
import groq as groq_sdk
from openai import OpenAI

from rag.models import GenerationConfig


class LLMRouter(ABC):
    """Abstract base class for LLM completion backends.

    Consuming code (generation layer, API routes) must depend only on this
    interface — never import a concrete router directly.
    """

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the completion text.

        Args:
            prompt: Fully-rendered prompt string (system + context + question).

        Returns:
            Model-generated completion as a plain string.
        """


class OpenAIRouter(LLMRouter):
    """LLMRouter backed by the OpenAI Chat Completions API.

    API key is read from the ``OPENAI_API_KEY`` environment variable; it is
    never accepted as a constructor argument to prevent accidental hardcoding.

    Args:
        config: Generation config supplying model name, temperature, and
            max_tokens.
    """

    def __init__(self, config: GenerationConfig) -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        self._client = OpenAI(api_key=api_key)
        self._config = config

    def complete(self, prompt: str) -> str:
        """Call the OpenAI Chat Completions API with *prompt* as the user message.

        Args:
            prompt: Rendered prompt string.

        Returns:
            Completion text; empty string when the model returns no content.
        """
        response = self._client.chat.completions.create(
            model=self._config.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        content = response.choices[0].message.content
        return content if content is not None else ""


class AnthropicRouter(LLMRouter):
    """LLMRouter backed by the Anthropic Messages API.

    API key is read from the ``ANTHROPIC_API_KEY`` environment variable.

    Args:
        config: Generation config supplying model name, temperature, and
            max_tokens.
    """

    def __init__(self, config: GenerationConfig) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._config = config

    def complete(self, prompt: str) -> str:
        """Call the Anthropic Messages API with *prompt* as the user message.

        Args:
            prompt: Rendered prompt string.

        Returns:
            Text from the first TextBlock in the response; empty string when
            the response contains no text block.
        """
        response = self._client.messages.create(
            model=self._config.anthropic_model,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if isinstance(block, anthropic.types.TextBlock):
                return block.text
        return ""


class GroqRouter(LLMRouter):
    """LLMRouter backed by the Groq Chat Completions API.

    API key is read from the ``GROQ_API_KEY`` environment variable.

    Args:
        config: Generation config supplying model name, temperature, and
            max_tokens.  ``config.groq_model`` selects the model
            (default: ``llama-3.3-70b-versatile``).
    """

    def __init__(self, config: GenerationConfig) -> None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        self._client = groq_sdk.Groq(api_key=api_key)
        self._config = config

    def complete(self, prompt: str) -> str:
        """Call the Groq Chat Completions API with *prompt* as the user message.

        Args:
            prompt: Rendered prompt string.

        Returns:
            Completion text; empty string when the model returns no content.
        """
        response = self._client.chat.completions.create(
            model=self._config.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        content = response.choices[0].message.content
        return content if content is not None else ""


def make_router(config: GenerationConfig) -> LLMRouter:
    """Instantiate the appropriate :class:`LLMRouter` based on available env vars.

    Selection priority:

    1. ``OPENAI_API_KEY`` is set → :class:`OpenAIRouter`
    2. ``GROQ_API_KEY`` is set (and ``OPENAI_API_KEY`` is not) → :class:`GroqRouter`
    3. ``ANTHROPIC_API_KEY`` is set → :class:`AnthropicRouter`
    4. Fall back to :class:`OpenAIRouter` (will fail at call time if key is absent)

    Args:
        config: Generation configuration forwarded to the chosen router.

    Returns:
        A concrete :class:`LLMRouter` instance.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIRouter(config)
    if os.environ.get("GROQ_API_KEY"):
        return GroqRouter(config)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicRouter(config)
    return OpenAIRouter(config)
