"""Abstract LLMRouter interface and OpenAI/Anthropic concrete implementations."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import anthropic
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
