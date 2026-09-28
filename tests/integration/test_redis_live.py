"""Against a real Redis server. Skipped unless ``REDIS_URL`` is set (CI runs a
Redis service container). Uses its own logical DB and flushes it, so point it
at a disposable instance.

These cover what fakeredis cannot prove: Lua atomicity and WATCH/MULTI under
genuinely concurrent connections, real pub/sub delivery, and TTL behaviour.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Generator
from datetime import datetime

import numpy as np
import pytest
import redis

from rag.cache.query_cache import CachedRetrieval, QueryCache
from rag.drift.detector import DriftDetector
from rag.drift.redis_state import RedisDriftDetector
from rag.drift.snapshot import DistributionSnapshot
from rag.index_sync import IndexSync
from rag.models import Chunk, ChunkMetadata, DriftConfig, RemediationIncident, SourceType
from rag.redis_health import RedisHealth
from rag.remediation import RedisRemediationStore

REDIS_URL = os.environ.get("REDIS_URL", "")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="REDIS_URL not set; live Redis tests skipped")

TEST_DB = 15
DIM = 16


def _client() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, db=TEST_DB)


@pytest.fixture()
def flushed() -> Generator[None, None, None]:
    _client().flushdb()
    yield
    _client().flushdb()


def _cfg() -> DriftConfig:
    return DriftConfig(window_size=10, pca_components=4, hysteresis_windows=3)


def _snapshot() -> DistributionSnapshot:
    rng = np.random.default_rng(0)
    return DistributionSnapshot(rng.normal(size=(60, DIM)).astype(np.float32), _cfg())


@pytest.mark.usefixtures("flushed")
def test_concurrent_appends_form_exact_windows() -> None:
    """3 'replicas' x 4 threads each push 1,000 samples concurrently: every sample
    lands in exactly one window, no window is short, none is lost."""
    cfg = _cfg()
    replicas = [RedisDriftDetector(_client(), _snapshot(), cfg, "rag:drift:live") for _ in range(3)]
    per_thread = 250
    errors: list[BaseException] = []

    def push(rep: RedisDriftDetector, seed: int) -> None:
        rng = np.random.default_rng(seed)
        try:
            for _ in range(per_thread):
                rep.add_query_embeddings([(rng.normal(size=DIM).astype(np.float32), 0.5)])
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=push, args=(replicas[i % 3], i)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []

    total = 12 * per_thread
    windows = total // cfg.window_size
    # 1 calibration window + (windows - 1) evaluated windows, nothing pending.
    assert len(replicas[0].history) == windows - 1
    assert replicas[1].buffer_size == total % cfg.window_size
    assert _client().llen("rag:drift:live:pending") == 0


@pytest.mark.usefixtures("flushed")
def test_shared_detector_matches_local_detector() -> None:
    cfg = _cfg()
    rng = np.random.default_rng(3)
    traffic = [
        (
            rng.normal(size=DIM).astype(np.float32) + (3.0 if i >= 40 else 0.0),
            0.5 if i < 40 else 0.1,
        )
        for i in range(120)
    ]
    replicas = [RedisDriftDetector(_client(), _snapshot(), cfg, "rag:drift:eq") for _ in range(3)]
    shared = []
    for i, sample in enumerate(traffic):
        shared += replicas[i % 3].add_query_embeddings([sample])
    local = DriftDetector(_snapshot(), cfg).add_query_embeddings(traffic)
    assert [(o.result.drifted, o.reindex_triggered) for o in shared] == [
        (o.result.drifted, o.reindex_triggered) for o in local
    ]


@pytest.mark.usefixtures("flushed")
def test_cache_round_trip_and_ttl() -> None:
    cache = QueryCache(RedisHealth(_client()), ttl_s=1)
    chunk = Chunk(
        id="c",
        text="t",
        token_count=1,
        metadata=ChunkMetadata(source="s", source_type=SourceType.TEXT, chunk_index=0),
    )
    key = QueryCache.key("v", "q", 5, None, 0.0)
    cache.put(key, CachedRetrieval(np.ones(4, dtype=np.float32), [chunk], [0.5], 1))
    assert cache.get(key) is not None
    time.sleep(1.2)
    assert cache.get(key) is None


@pytest.mark.usefixtures("flushed")
def test_concurrent_auto_alarms_open_exactly_one_incident() -> None:
    now = datetime(2026, 9, 27)
    stores = [RedisRemediationStore(_client(), key="rag:remediations:live") for _ in range(6)]
    barrier = threading.Barrier(len(stores))

    def fire(store: RedisRemediationStore, i: int) -> None:
        barrier.wait()
        store.open_or_dedupe(
            now,
            3600,
            lambda: RemediationIncident(incident_id=f"inc-{i}", opened_at=now, updated_at=now),
        )

    threads = [threading.Thread(target=fire, args=(s, i)) for i, s in enumerate(stores)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    [incident] = stores[0].list()
    assert incident.occurrences == len(stores)


@pytest.mark.usefixtures("flushed")
def test_pubsub_announcement_reaches_another_replica() -> None:
    file = {"version": "v0"}
    follower_version = {"v": "v0"}

    def reload() -> None:
        follower_version["v"] = file["version"]

    writer = IndexSync(RedisHealth(_client()), local_version=lambda: file["version"], reload=reload)
    follower = IndexSync(
        RedisHealth(_client()),
        local_version=lambda: follower_version["v"],
        reload=reload,
        poll_interval_s=60.0,  # long poll: only pub/sub can deliver in time
    )
    follower.start()
    try:
        time.sleep(0.3)
        file["version"] = "v1"
        writer.publish("v1")
        deadline = time.monotonic() + 3.0
        while follower_version["v"] != "v1" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert follower_version["v"] == "v1"
    finally:
        follower.stop()
