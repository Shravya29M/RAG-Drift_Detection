"""Token-sized chunking against the real all-MiniLM-L6-v2 tokenizer.

Downloads only the tokenizer files and ``sentence_bert_config.json`` (no model
weights). The guard test measures every chunk with special tokens included
against the model's own ``max_seq_length`` — the length past which the
embedding silently ignores text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from rag.ingestion.chunker import chunk_text, count_tokens
from rag.ingestion.parsers import parse_markdown
from rag.models import IngestConfig, SourceType

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLES = Path(__file__).resolve().parents[2] / "samples"


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def model_max_seq_length() -> int:
    """The embedding model's truncation length, from the model's own config."""
    cfg = json.loads(Path(hf_hub_download(MODEL, "sentence_bert_config.json")).read_text())
    return int(cfg["max_seq_length"])


def _content_limit(tokenizer: Any, model_max_seq_length: int) -> int:
    return model_max_seq_length - int(tokenizer.num_special_tokens_to_add(pair=False))


def _texts() -> list[str]:
    """Bundled docs plus adversarial inputs: code, unicode, CJK, one giant word."""
    texts = [s for p in sorted(SAMPLES.glob("*.md")) for s in parse_markdown(p)]
    texts += [
        "def f(x):\n    return {'k': [x ** 2 for x in range(10)]}\n" * 60,
        "Café naïve résumé — ‘quotes’ → arrows ≥ 3 µs " * 80,
        "日本語のテキストは空白なしで続きます。" * 120,
        "https://example.com/" + "a1b2c3d4e5" * 400,  # one ~4k-char "word"
        "word " * 5,
    ]
    return texts


def test_no_chunk_exceeds_the_models_max_length(tokenizer: Any, model_max_seq_length: int) -> None:
    """Production defaults + the model's real tokenizer: every chunk, encoded the
    way the model encodes it (with [CLS]/[SEP]), fits in max_seq_length."""
    config = IngestConfig()
    limit = _content_limit(tokenizer, model_max_seq_length)
    worst = 0
    n = 0
    for text in _texts():
        for chunk in chunk_text(
            text, "src", SourceType.MARKDOWN, config, tokenizer=tokenizer, max_tokens=limit
        ):
            with_specials = len(tokenizer(chunk.text, verbose=False)["input_ids"])
            assert with_specials <= model_max_seq_length, (with_specials, chunk.text[:80])
            assert count_tokens(tokenizer, chunk.text) == chunk.token_count
            worst = max(worst, with_specials)
            n += 1
    assert n > 20 and worst > 150  # the test really exercised near-full chunks


def test_defaults_fit_the_model(tokenizer: Any, model_max_seq_length: int) -> None:
    assert IngestConfig().chunk_size <= _content_limit(tokenizer, model_max_seq_length)


def test_chunk_size_above_the_model_limit_is_rejected(tokenizer: Any) -> None:
    with pytest.raises(ValueError, match="exceeds the embedding model's input limit"):
        chunk_text(
            "some text",
            "src",
            SourceType.TEXT,
            IngestConfig(chunk_size=300, chunk_overlap=30),
            tokenizer=tokenizer,
            max_tokens=254,
        )


def test_windows_overlap_by_whole_words_and_cover_the_text(tokenizer: Any) -> None:
    words = [f"token{i}" for i in range(400)]
    text = " ".join(words)
    chunks = chunk_text(
        text,
        "src",
        SourceType.TEXT,
        IngestConfig(chunk_size=40, chunk_overlap=10),
        tokenizer=tokenizer,
    )
    seen: list[str] = []
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        a, b = prev.text.split(), nxt.text.split()
        shared = [w for w in b if w in set(a)]
        assert shared, "consecutive chunks must overlap"
        assert count_tokens(tokenizer, " ".join(shared)) <= 10
        assert b[: len(shared)] == shared  # overlap is a prefix of the next chunk
    for c in chunks:
        assert c.token_count <= 40
        seen.extend(c.text.split())
    assert set(seen) == set(words)
    assert [c.metadata.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunks_are_exact_slices_of_the_source(tokenizer: Any) -> None:
    text = "Alpha, beta; gamma!\n\nDelta   epsilon. " * 50
    for c in chunk_text(
        text,
        "s",
        SourceType.TEXT,
        IngestConfig(chunk_size=30, chunk_overlap=5),
        tokenizer=tokenizer,
    ):
        assert c.text in text


def test_zero_overlap_and_empty_text(tokenizer: Any) -> None:
    cfg = IngestConfig(chunk_size=20, chunk_overlap=0)
    assert chunk_text("   \n ", "s", SourceType.TEXT, cfg, tokenizer=tokenizer) == []
    chunks = chunk_text(" ".join(["w"] * 100), "s", SourceType.TEXT, cfg, tokenizer=tokenizer)
    assert sum(c.token_count for c in chunks) == 100


def test_a_word_longer_than_the_window_is_split_to_fit(tokenizer: Any) -> None:
    # WordPiece maps words over 100 chars to one [UNK]; 99 chars of "7q" is 99 tokens.
    giant = "x" + "7q" * 49
    assert count_tokens(tokenizer, giant) > 25
    chunks = chunk_text(
        f"before {giant} after",
        "s",
        SourceType.TEXT,
        IngestConfig(chunk_size=25, chunk_overlap=5),
        tokenizer=tokenizer,
    )
    assert len(chunks) > 3
    assert all(count_tokens(tokenizer, c.text) <= 25 for c in chunks)
    assert chunks[0].text == "before"
    assert chunks[-1].text == "after"


def test_a_word_over_100_chars_is_one_unknown_token(tokenizer: Any) -> None:
    """Documents the tokenizer behaviour the splitter relies on not needing."""
    [chunk] = chunk_text(
        "q" * 500,
        "s",
        SourceType.TEXT,
        IngestConfig(chunk_size=5, chunk_overlap=0),
        tokenizer=tokenizer,
    )
    assert chunk.token_count == 1


def test_split_shrinks_a_piece_that_retokenizes_longer(tokenizer: Any) -> None:
    """A mid-word slice can gain tokens when re-tokenized; the splitter must
    re-measure and shrink it instead of emitting an oversized piece."""
    from rag.ingestion import chunker

    real = chunker.count_tokens
    calls = {"n": 0}

    def inflate_first(tok: Any, text: str) -> int:
        calls["n"] += 1
        return real(tok, text) + (5 if calls["n"] == 1 else 0)

    giant = "z" + "9k" * 49
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(chunker, "count_tokens", inflate_first)
        chunks = chunk_text(
            giant,
            "s",
            SourceType.TEXT,
            IngestConfig(chunk_size=20, chunk_overlap=0),
            tokenizer=tokenizer,
        )
    assert all(real(tokenizer, c.text) <= 20 for c in chunks)
    assert calls["n"] > len(chunks)  # at least one piece was re-measured
