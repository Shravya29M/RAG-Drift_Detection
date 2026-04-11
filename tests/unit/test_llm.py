"""Unit tests for rag.generation.llm and rag.generation.prompt."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from rag.generation.llm import AnthropicRouter, GroqRouter, LLMRouter, OpenAIRouter, make_router
from rag.generation.prompt import build_prompt
from rag.models import Chunk, ChunkMetadata, GenerationConfig, SourceType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**kwargs: object) -> GenerationConfig:
    return GenerationConfig(**kwargs)  # type: ignore[arg-type]


def _chunk(cid: str, text: str = "chunk text", source: str = "doc.txt") -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        token_count=len(text.split()),
        metadata=ChunkMetadata(
            source=source,
            source_type=SourceType.TEXT,
            chunk_index=0,
            ingested_at=datetime(2026, 1, 1),
        ),
    )


# ---------------------------------------------------------------------------
# LLMRouter ABC
# ---------------------------------------------------------------------------


class TestLLMRouterABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            LLMRouter()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_complete(self) -> None:
        class Incomplete(LLMRouter):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_is_valid(self) -> None:
        class Echo(LLMRouter):
            def complete(self, prompt: str) -> str:
                return prompt

        assert isinstance(Echo(), LLMRouter)


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_contains_query(self) -> None:
        cfg = _cfg()
        prompt = build_prompt("What is X?", [_chunk("a")], cfg)
        assert "What is X?" in prompt

    def test_contains_chunk_text(self) -> None:
        cfg = _cfg()
        prompt = build_prompt("q", [_chunk("a", text="important fact")], cfg)
        assert "important fact" in prompt

    def test_contains_source_attribution(self) -> None:
        cfg = _cfg()
        prompt = build_prompt("q", [_chunk("a", source="paper.pdf")], cfg)
        assert "paper.pdf" in prompt

    def test_multiple_chunks_numbered(self) -> None:
        cfg = _cfg()
        chunks = [_chunk(f"c{i}") for i in range(3)]
        prompt = build_prompt("q", chunks, cfg)
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt

    def test_empty_chunks_uses_fallback_context(self) -> None:
        cfg = _cfg()
        prompt = build_prompt("q", [], cfg)
        assert "no context retrieved" in prompt

    def test_custom_template(self) -> None:
        cfg = _cfg(prompt_template="CTX:{context} Q:{query}")
        prompt = build_prompt("hello", [_chunk("a", text="fact")], cfg)
        assert prompt.startswith("CTX:")
        assert "Q:hello" in prompt

    def test_unknown_placeholder_raises(self) -> None:
        # Template contains {unknown} which is never passed → KeyError
        cfg = _cfg(prompt_template="{context} {query} {unknown}")
        with pytest.raises(KeyError):
            build_prompt("q", [_chunk("a")], cfg)

    def test_chunks_separated_by_blank_line(self) -> None:
        cfg = _cfg()
        chunks = [_chunk("a", text="first"), _chunk("b", text="second")]
        prompt = build_prompt("q", chunks, cfg)
        assert "first" in prompt and "second" in prompt
        # Blank-line separator between the two numbered entries
        assert "\n\n" in prompt


# ---------------------------------------------------------------------------
# OpenAIRouter
# ---------------------------------------------------------------------------


class TestOpenAIRouter:
    def _make(self, response_text: str = "answer") -> tuple[OpenAIRouter, MagicMock]:
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = response_text
        mock_client.chat.completions.create.return_value = MagicMock(choices=[choice])
        with patch("rag.generation.llm.OpenAI", return_value=mock_client):
            router = OpenAIRouter(_cfg())
        return router, mock_client

    def test_is_llm_router_subclass(self) -> None:
        router, _ = self._make()
        assert isinstance(router, LLMRouter)

    def test_complete_returns_string(self) -> None:
        router, _ = self._make("the answer")
        assert router.complete("prompt") == "the answer"

    def test_complete_calls_chat_completions(self) -> None:
        router, mock_client = self._make()
        router.complete("my prompt")
        mock_client.chat.completions.create.assert_called_once()

    def test_complete_passes_model_name(self) -> None:
        cfg = _cfg(openai_model="gpt-4-turbo")
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = "ok"
        mock_client.chat.completions.create.return_value = MagicMock(choices=[choice])
        with patch("rag.generation.llm.OpenAI", return_value=mock_client):
            router = OpenAIRouter(cfg)
        router.complete("p")
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == "gpt-4-turbo"

    def test_complete_passes_temperature_and_max_tokens(self) -> None:
        cfg = _cfg(temperature=0.5, max_tokens=256)
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = "ok"
        mock_client.chat.completions.create.return_value = MagicMock(choices=[choice])
        with patch("rag.generation.llm.OpenAI", return_value=mock_client):
            router = OpenAIRouter(cfg)
        router.complete("p")
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 256

    def test_complete_prompt_in_messages(self) -> None:
        router, mock_client = self._make()
        router.complete("hello world")
        _, kwargs = mock_client.chat.completions.create.call_args
        messages = kwargs["messages"]
        assert any(m["content"] == "hello world" for m in messages)

    def test_none_content_returns_empty_string(self) -> None:
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = None
        mock_client.chat.completions.create.return_value = MagicMock(choices=[choice])
        with patch("rag.generation.llm.OpenAI", return_value=mock_client):
            router = OpenAIRouter(_cfg())
        assert router.complete("p") == ""

    def test_api_key_read_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        captured: list[str] = []

        def fake_client(*, api_key: str) -> MagicMock:
            captured.append(api_key)
            return MagicMock()

        with patch("rag.generation.llm.OpenAI", side_effect=fake_client):
            OpenAIRouter(_cfg())
        assert captured[0] == "test-key-123"


# ---------------------------------------------------------------------------
# AnthropicRouter
# ---------------------------------------------------------------------------


class TestAnthropicRouter:
    def _make(self, response_text: str = "answer") -> tuple[AnthropicRouter, MagicMock]:
        mock_client = MagicMock()
        text_block = MagicMock(spec=anthropic.types.TextBlock)
        text_block.text = response_text
        mock_client.messages.create.return_value = MagicMock(content=[text_block])
        with patch("rag.generation.llm.anthropic.Anthropic", return_value=mock_client):
            router = AnthropicRouter(_cfg())
        return router, mock_client

    def test_is_llm_router_subclass(self) -> None:
        router, _ = self._make()
        assert isinstance(router, LLMRouter)

    def test_complete_returns_string(self) -> None:
        router, _ = self._make("anthropic answer")
        assert router.complete("p") == "anthropic answer"

    def test_complete_calls_messages_create(self) -> None:
        router, mock_client = self._make()
        router.complete("prompt text")
        mock_client.messages.create.assert_called_once()

    def test_complete_passes_model_name(self) -> None:
        cfg = _cfg(anthropic_model="claude-haiku-4-5-20251001")
        mock_client = MagicMock()
        tb = MagicMock(spec=anthropic.types.TextBlock)
        tb.text = "ok"
        mock_client.messages.create.return_value = MagicMock(content=[tb])
        with patch("rag.generation.llm.anthropic.Anthropic", return_value=mock_client):
            router = AnthropicRouter(cfg)
        router.complete("p")
        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["model"] == "claude-haiku-4-5-20251001"

    def test_complete_passes_temperature_and_max_tokens(self) -> None:
        cfg = _cfg(temperature=0.7, max_tokens=512)
        mock_client = MagicMock()
        tb = MagicMock(spec=anthropic.types.TextBlock)
        tb.text = "ok"
        mock_client.messages.create.return_value = MagicMock(content=[tb])
        with patch("rag.generation.llm.anthropic.Anthropic", return_value=mock_client):
            router = AnthropicRouter(cfg)
        router.complete("p")
        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 512

    def test_complete_prompt_in_messages(self) -> None:
        router, mock_client = self._make()
        router.complete("my question")
        _, kwargs = mock_client.messages.create.call_args
        assert any(m["content"] == "my question" for m in kwargs["messages"])

    def test_no_text_block_returns_empty_string(self) -> None:
        """Non-TextBlock content (e.g. ToolUseBlock) yields empty string."""
        mock_client = MagicMock()
        non_text = MagicMock(spec=anthropic.types.ToolUseBlock)
        mock_client.messages.create.return_value = MagicMock(content=[non_text])
        with patch("rag.generation.llm.anthropic.Anthropic", return_value=mock_client):
            router = AnthropicRouter(_cfg())
        assert router.complete("p") == ""

    def test_empty_content_returns_empty_string(self) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(content=[])
        with patch("rag.generation.llm.anthropic.Anthropic", return_value=mock_client):
            router = AnthropicRouter(_cfg())
        assert router.complete("p") == ""

    def test_api_key_read_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-key-xyz")
        captured: list[str] = []

        def fake_client(*, api_key: str) -> MagicMock:
            captured.append(api_key)
            return MagicMock()

        with patch("rag.generation.llm.anthropic.Anthropic", side_effect=fake_client):
            AnthropicRouter(_cfg())
        assert captured[0] == "anth-key-xyz"


# ---------------------------------------------------------------------------
# GroqRouter
# ---------------------------------------------------------------------------


class TestGroqRouter:
    def test_complete_returns_content(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="groq answer"))]
        )
        with patch("rag.generation.llm.groq_sdk.Groq", return_value=mock_client):
            router = GroqRouter(_cfg())
        assert router.complete("p") == "groq answer"

    def test_uses_groq_model_from_config(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )
        with patch("rag.generation.llm.groq_sdk.Groq", return_value=mock_client):
            router = GroqRouter(_cfg(groq_model="llama-3.3-70b-versatile"))
            router.complete("p")
        kwargs = mock_client.chat.completions.create.call_args[1]
        assert kwargs["model"] == "llama-3.3-70b-versatile"

    def test_none_content_returns_empty_string(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=None))]
        )
        with patch("rag.generation.llm.groq_sdk.Groq", return_value=mock_client):
            router = GroqRouter(_cfg())
        assert router.complete("p") == ""

    def test_api_key_read_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test-key")
        captured: list[str] = []

        def fake_client(*, api_key: str) -> MagicMock:
            captured.append(api_key)
            return MagicMock()

        with patch("rag.generation.llm.groq_sdk.Groq", side_effect=fake_client):
            GroqRouter(_cfg())
        assert captured[0] == "gsk-test-key"

    def test_passes_temperature_and_max_tokens(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )
        with patch("rag.generation.llm.groq_sdk.Groq", return_value=mock_client):
            router = GroqRouter(_cfg(temperature=0.1, max_tokens=512))
            router.complete("p")
        kwargs = mock_client.chat.completions.create.call_args[1]
        assert kwargs["temperature"] == pytest.approx(0.1)
        assert kwargs["max_tokens"] == 512


# ---------------------------------------------------------------------------
# make_router factory
# ---------------------------------------------------------------------------


class TestMakeRouter:
    def test_openai_key_returns_openai_router(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("rag.generation.llm.OpenAI"):
            router = make_router(_cfg())
        assert isinstance(router, OpenAIRouter)

    def test_groq_key_without_openai_returns_groq_router(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("rag.generation.llm.groq_sdk.Groq"):
            router = make_router(_cfg())
        assert isinstance(router, GroqRouter)

    def test_openai_key_takes_priority_over_groq(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("rag.generation.llm.OpenAI"):
            router = make_router(_cfg())
        assert isinstance(router, OpenAIRouter)

    def test_anthropic_key_only_returns_anthropic_router(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-test")
        with patch("rag.generation.llm.anthropic.Anthropic"):
            router = make_router(_cfg())
        assert isinstance(router, AnthropicRouter)

    def test_no_keys_falls_back_to_openai_router(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("rag.generation.llm.OpenAI"):
            router = make_router(_cfg())
        assert isinstance(router, OpenAIRouter)
