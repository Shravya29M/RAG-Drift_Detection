"""QueryCache: keying, round-trip, TTL, version purge, and graceful degradation."""

from __future__ import annotations

from unittest.mock import patch

import fakeredis
import numpy as np
import pytest

from rag.cache.query_cache import CachedRetrieval, QueryCache, normalize_query
from rag.metrics import CACHE_BYPASSED, CACHE_HITS, CACHE_INVALIDATED_KEYS, CACHE_MISSES
from rag.models import Chunk, ChunkMetadata, SourceType
from rag.redis_health import RedisHealth


def _count(metric: object) -> float:
    return float(metric._value.get())  # type: ignore[attr-defined]


def _chunk(i: int) -> Chunk:
    return Chunk(
        id=f"c-{i}",
        text=f"text {i}",
        token_count=2,
        metadata=ChunkMetadata(source="s.md", source_type=SourceType.MARKDOWN, chunk_index=i),
    )


def _entry() -> CachedRetrieval:
    return CachedRetrieval(
        embedding=np.arange(4, dtype=np.float32),
        chunks=[_chunk(0), _chunk(1)],
        scores=[0.9, 0.5],
        total_candidates=2,
    )


@pytest.fixture()
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture()
def cache(server: fakeredis.FakeServer) -> QueryCache:
    return QueryCache(RedisHealth(fakeredis.FakeRedis(server=server)), ttl_s=120)


class TestKey:
    def test_normalisation_collapses_case_space_and_unicode_width(self) -> None:
        assert normalize_query("  What IS\tDrift?\n") == "what is drift?"
        assert normalize_query("ｄｒｉｆｔ") == "drift"  # NFKC folds full-width forms

    def test_equivalent_queries_share_a_key(self) -> None:
        a = QueryCache.key("v1", "What is drift?", 5, None, 0.0)
        b = QueryCache.key("v1", "  what   is DRIFT? ", 5, {}, 0.0)
        assert a == b
        assert a.startswith("rag:q:v1:")

    @pytest.mark.parametrize(
        "other",
        [
            ("v2", "q", 5, None, 0.0),
            ("v1", "q2", 5, None, 0.0),
            ("v1", "q", 3, None, 0.0),
            ("v1", "q", 5, {"source": "a.md"}, 0.0),
            ("v1", "q", 5, None, 0.3),
        ],
    )
    def test_every_request_dimension_changes_the_key(self, other: tuple) -> None:  # type: ignore[type-arg]
        assert QueryCache.key("v1", "q", 5, None, 0.0) != QueryCache.key(*other)

    def test_filter_order_does_not_matter(self) -> None:
        a = QueryCache.key("v", "q", 5, {"a": 1, "b": "x"}, 0.0)
        b = QueryCache.key("v", "q", 5, {"b": "x", "a": 1}, 0.0)
        assert a == b


class TestRoundTrip:
    def test_miss_then_hit(self, cache: QueryCache) -> None:
        key = QueryCache.key("v", "q", 5, None, 0.0)
        hits, misses = _count(CACHE_HITS), _count(CACHE_MISSES)
        assert cache.get(key) is None
        cache.put(key, _entry())
        got = cache.get(key)
        assert got is not None
        np.testing.assert_array_equal(got.embedding, np.arange(4, dtype=np.float32))
        assert [c.id for c in got.chunks] == ["c-0", "c-1"]
        assert got.scores == [0.9, 0.5]
        assert got.total_candidates == 2
        assert _count(CACHE_HITS) == hits + 1
        assert _count(CACHE_MISSES) == misses + 1

    def test_entries_carry_the_ttl(self, cache: QueryCache, server: fakeredis.FakeServer) -> None:
        key = QueryCache.key("v", "q", 5, None, 0.0)
        cache.put(key, _entry())
        ttl = fakeredis.FakeRedis(server=server).ttl(key)
        assert 0 < ttl <= 120

    def test_undecodable_entry_is_a_miss(
        self, cache: QueryCache, server: fakeredis.FakeServer
    ) -> None:
        fakeredis.FakeRedis(server=server).set("rag:q:v:bad", b"not json")
        with patch("rag.cache.query_cache.logger"):
            assert cache.get("rag:q:v:bad") is None


class TestPurge:
    def test_purges_only_the_given_version(
        self, cache: QueryCache, server: fakeredis.FakeServer
    ) -> None:
        for i in range(1203):  # crosses the 500-key UNLINK batch boundary twice
            cache.put(QueryCache.key("old", f"q{i}", 5, None, 0.0), _entry())
        keep = QueryCache.key("new", "q", 5, None, 0.0)
        cache.put(keep, _entry())
        before = _count(CACHE_INVALIDATED_KEYS)
        assert cache.purge_version("old") == 1203
        assert _count(CACHE_INVALIDATED_KEYS) == before + 1203
        client = fakeredis.FakeRedis(server=server)
        assert list(client.scan_iter(match="rag:q:old:*")) == []
        assert client.exists(keep)

    def test_purge_of_empty_version_is_zero(self, cache: QueryCache) -> None:
        assert cache.purge_version("nothing") == 0


class TestDegradation:
    def test_redis_error_on_get_bypasses_and_marks_down(
        self, cache: QueryCache, server: fakeredis.FakeServer
    ) -> None:
        server.connected = False
        bypassed = _count(CACHE_BYPASSED)
        with patch("rag.redis_health.logger"):
            assert cache.get("rag:q:v:x") is None
        assert _count(CACHE_BYPASSED) == bypassed + 1
        assert not cache._health.up

    def test_while_down_no_network_calls_are_made(self, cache: QueryCache) -> None:
        cache._health._up = False
        cache._health._next_probe = float("inf")
        with patch.object(cache._client, "get") as get, patch.object(cache._client, "set") as st:
            assert cache.get("k") is None
            cache.put("k", _entry())
            assert cache.purge_version("v") == 0
        get.assert_not_called()
        st.assert_not_called()

    def test_put_and_purge_swallow_errors(
        self, cache: QueryCache, server: fakeredis.FakeServer
    ) -> None:
        server.connected = False
        with patch("rag.redis_health.logger"):
            cache.put("rag:q:v:x", _entry())  # must not raise
            cache._health._up = True
            assert cache.purge_version("v") == 0
        assert not cache._health.up
