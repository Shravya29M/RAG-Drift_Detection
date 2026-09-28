"""API with Redis (fakeredis): two in-process "replicas" share one Redis server
and one index file, the way the docker-compose deployment runs."""

from __future__ import annotations

import hashlib
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import fakeredis
import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag.api import (
    AppState,
    _attach_shared_services,
    _bootstrap_index,
    _connect_redis,
    _index_version,
    _new_job,
    _open_remediation,
    _run_ingest,
    _run_reindex,
    app,
)
from rag.cache.query_cache import KEY_PREFIX, QueryCache
from rag.drift.redis_state import ResilientDriftMonitor
from rag.embedding.encoder import Encoder
from rag.index_sync import VERSION_KEY, index_version
from rag.metrics import CACHE_HITS
from rag.models import (
    DriftConfig,
    GenerationConfig,
    IngestConfig,
    JobStatusEnum,
    RedisConfig,
    RemediationStatus,
    SourceType,
)
from rag.redis_health import RedisHealth
from rag.retrieval.retriever import Retriever
from rag.vector_store.faiss_store import FAISSStore

from .test_api import _MockRouter

DIM = 8
MODEL = "test-model"


class _HashEncoder(Encoder):
    """Deterministic per-text unit vectors; counts calls to prove cache hits skip it."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def dim(self) -> int:
        return DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "little")
            v = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows)


def _state(server: fakeredis.FakeServer, index_path: Path, *, cache: bool = True) -> AppState:
    store = FAISSStore(dim=DIM)
    health = RedisHealth(fakeredis.FakeRedis(server=server), retry_interval_s=0.0)
    encoder = _HashEncoder()
    query_cache = QueryCache(health) if cache else None
    state = AppState(
        encoder=encoder,
        store=store,
        retriever=Retriever(
            store,
            encoder,
            cache=query_cache,
            index_version=lambda: index_version(store, MODEL),
        ),
        llm_router=_MockRouter(),
        generation_config=GenerationConfig(),
        ingest_config=IngestConfig(chunk_size=8, chunk_overlap=2),
        drift_config=DriftConfig(window_size=4, pca_components=2, hysteresis_windows=2),
        index_path=index_path,
        model_name=MODEL,
        redis=health,
        query_cache=query_cache,
    )
    _attach_shared_services(state)
    return state


def _doc(tmp_path: Path, name: str, words: int = 40) -> tuple[str, SourceType]:
    path = tmp_path / name
    path.write_text(" ".join(f"{name}-word{i}" for i in range(words)))
    return str(path), SourceType.TEXT


def _ingest(state: AppState, *sources: tuple[str, SourceType]) -> None:
    job = _new_job(state)
    _run_ingest(state, job, list(sources), state.ingest_config, [])
    assert state.jobs[job].status is JobStatusEnum.DONE, state.jobs[job].error


@pytest.fixture()
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture()
def replicas(
    server: fakeredis.FakeServer, tmp_path: Path
) -> Generator[tuple[AppState, AppState], None, None]:
    index_path = tmp_path / "faiss.index"
    a, b = _state(server, index_path), _state(server, index_path)
    _ingest(a, _doc(tmp_path, "alpha.txt"), _doc(tmp_path, "beta.txt"))
    assert b.index_sync is not None and b.index_sync.sync()
    yield a, b
    for s in (a, b):
        if s.drift_scheduler is not None:
            s.drift_scheduler.shutdown(wait=False)


def _client(state: AppState) -> TestClient:
    app.state.app = state
    return TestClient(app)


# ---------------------------------------------------------------------------
# Query cache
# ---------------------------------------------------------------------------


class TestQueryCache:
    def test_repeat_query_is_a_cache_hit_that_still_feeds_drift(
        self, replicas: tuple[AppState, AppState]
    ) -> None:
        a, _ = replicas
        assert isinstance(a.encoder, _HashEncoder)
        hits = CACHE_HITS._value.get()
        with _client(a) as c:
            calls = a.encoder.calls
            c.post("/query", json={"query": "alpha words"})
            c.post("/query", json={"query": "  ALPHA   words "})
        assert CACHE_HITS._value.get() == hits + 1
        assert a.encoder.calls == calls + 1  # one encode total: no double-embed, no re-encode
        assert a.drift_scheduler is not None and a.drift_scheduler.queue_size == 2

    def test_cache_is_shared_between_replicas(self, replicas: tuple[AppState, AppState]) -> None:
        a, b = replicas
        assert isinstance(b.encoder, _HashEncoder)
        with _client(a) as c:
            c.post("/query", json={"query": "beta"})
        calls = b.encoder.calls
        with _client(b) as c:
            c.post("/query", json={"query": "beta"})
        assert b.encoder.calls == calls

    def test_ingest_changes_version_and_purges_old_entries(
        self, replicas: tuple[AppState, AppState], server: fakeredis.FakeServer, tmp_path: Path
    ) -> None:
        a, b = replicas
        old = _index_version(a)
        with _client(a) as c:
            c.post("/query", json={"query": "alpha"})
        client = fakeredis.FakeRedis(server=server)
        assert list(client.scan_iter(match=f"{KEY_PREFIX}:{old}:*"))

        _ingest(a, _doc(tmp_path, "gamma.txt"))
        new = _index_version(a)
        assert new != old
        assert client.get(VERSION_KEY) == new.encode()
        assert list(client.scan_iter(match=f"{KEY_PREFIX}:{old}:*")) == []

        with _client(a) as c:
            assert c.post("/query", json={"query": "alpha"}).status_code == 200
        assert list(client.scan_iter(match=f"{KEY_PREFIX}:{new}:*"))
        assert b.index_sync is not None and b.index_sync.sync()
        assert _index_version(b) == new

    def test_cache_can_be_disabled(self, server: fakeredis.FakeServer, tmp_path: Path) -> None:
        state = _state(server, tmp_path / "idx", cache=False)
        _ingest(state, _doc(tmp_path, "alpha.txt"))
        with _client(state) as c:
            c.post("/query", json={"query": "q"})
        assert list(fakeredis.FakeRedis(server=server).scan_iter(match=f"{KEY_PREFIX}:*")) == []
        if state.drift_scheduler is not None:
            state.drift_scheduler.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Index sync and bootstrap
# ---------------------------------------------------------------------------


class TestIndexSync:
    def test_replicas_converge_on_contents_and_drift_namespace(
        self, replicas: tuple[AppState, AppState]
    ) -> None:
        a, b = replicas
        assert _index_version(a) == _index_version(b)
        assert len(a.store.list_chunks()) == len(b.store.list_chunks()) > 0
        assert isinstance(a.drift_detector, ResilientDriftMonitor)
        assert isinstance(b.drift_detector, ResilientDriftMonitor)
        assert a.drift_detector.shared.namespace == b.drift_detector.shared.namespace

    def test_index_status_endpoint(self, replicas: tuple[AppState, AppState]) -> None:
        a, _ = replicas
        with _client(a) as c:
            body = c.get("/index/status").json()
        assert body["index_version"] == body["shared_version"] == _index_version(a)
        assert body["chunk_count"] == len(a.store.list_chunks())
        assert body["redis_up"] is True

    def test_bootstrap_seeds_once_and_others_load_the_file(
        self,
        server: fakeredis.FakeServer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        samples = tmp_path / "samples"
        samples.mkdir()
        (samples / "doc.md").write_text(" ".join(f"w{i}" for i in range(30)))
        monkeypatch.setenv("SAMPLES_DIR", str(samples))
        monkeypatch.setenv("SEED_SAMPLE_DATA", "true")
        index_path = tmp_path / "faiss.index"
        a, b = _state(server, index_path), _state(server, index_path)
        _bootstrap_index(a)
        seeded = a.store.snapshot_id  # type: ignore[attr-defined]
        _bootstrap_index(b)
        assert b.store.snapshot_id == seeded  # type: ignore[attr-defined]
        assert fakeredis.FakeRedis(server=server).get(VERSION_KEY) == _index_version(a).encode()
        for s in (a, b):
            if s.drift_scheduler is not None:
                s.drift_scheduler.shutdown(wait=False)

    def test_bootstrap_republishes_when_shared_version_is_stale(
        self, replicas: tuple[AppState, AppState], server: fakeredis.FakeServer
    ) -> None:
        a, _ = replicas
        client = fakeredis.FakeRedis(server=server)
        client.set(VERSION_KEY, "stale")
        with patch.object(a.index_sync, "sync", return_value=False):
            _bootstrap_index(a)
        assert client.get(VERSION_KEY) == _index_version(a).encode()

    def test_reload_without_a_file_is_a_no_op(
        self, server: fakeredis.FakeServer, tmp_path: Path
    ) -> None:
        state = _state(server, tmp_path / "missing.index")
        fakeredis.FakeRedis(server=server).set(VERSION_KEY, "other")
        assert state.index_sync is not None and state.index_sync.sync() is False
        assert state.drift_detector is None


class TestReindex:
    def test_reindex_publishes_a_new_version(self, replicas: tuple[AppState, AppState]) -> None:
        a, b = replicas
        before = _index_version(a)
        job = _new_job(a)
        _run_reindex(a, job)
        assert a.jobs[job].status is JobStatusEnum.DONE
        assert _index_version(a) != before
        assert b.index_sync is not None and b.index_sync.sync()
        assert _index_version(b) == _index_version(a)

    def test_reindex_retries_when_another_replica_changes_the_index(
        self, replicas: tuple[AppState, AppState], tmp_path: Path
    ) -> None:
        a, b = replicas
        real_encode = a.encoder.encode
        state = {"raced": False}

        def racing_encode(texts: list[str]) -> np.ndarray:
            if not state["raced"]:
                state["raced"] = True
                _ingest(b, _doc(tmp_path, "delta.txt"))  # lands while a is embedding
            return real_encode(texts)

        with patch.object(a.encoder, "encode", side_effect=racing_encode):
            job = _new_job(a)
            _run_reindex(a, job)
        assert a.jobs[job].status is JobStatusEnum.DONE
        ids = {c.id for c in a.store.list_chunks()}
        assert any(c.metadata.source.endswith("delta.txt") for c in b.store.list_chunks())
        assert ids == {c.id for c in b.store.list_chunks()}  # delta's chunks were not dropped

    def test_reindex_gives_up_if_the_index_never_settles(
        self, replicas: tuple[AppState, AppState]
    ) -> None:
        a, _ = replicas
        versions = iter(f"v{i}" for i in range(100))
        with patch("rag.api._index_version", side_effect=lambda _s: next(versions)):
            job = _new_job(a)
            _run_reindex(a, job)
        assert a.jobs[job].status is JobStatusEnum.ERROR
        assert "kept changing" in (a.jobs[job].error or "")


# ---------------------------------------------------------------------------
# Shared drift state and remediations
# ---------------------------------------------------------------------------


class TestSharedDrift:
    def test_drift_on_one_replica_is_visible_on_the_other(
        self, replicas: tuple[AppState, AppState]
    ) -> None:
        a, b = replicas
        with _client(a) as c:
            sim = c.post("/drift/simulate?windows=3").json()
            drift_a = c.get("/drift").json()
            remediations_a = c.get("/remediations").json()
        with _client(b) as c:
            drift_b = c.get("/drift").json()
            remediations_b = c.get("/remediations").json()
        # The hash encoder makes every text random noise, so whether these
        # windows drift is incidental; what matters is one shared answer.
        assert sim["history_length"] == 3
        assert len(drift_b["history"]) == 3
        assert drift_a == drift_b
        assert remediations_a == remediations_b

    def test_reset_on_one_replica_resets_all(self, replicas: tuple[AppState, AppState]) -> None:
        a, b = replicas
        with _client(a) as c:
            c.post("/drift/simulate?windows=2")
        with _client(b) as c:
            c.post("/drift/reset")
        with _client(a) as c:
            body = c.get("/drift").json()
        assert body["history"] == [] and body["baseline_ready"] is False


class TestSharedRemediations:
    def test_dedupe_and_resolve_across_replicas(self, replicas: tuple[AppState, AppState]) -> None:
        a, b = replicas
        _open_remediation(a)
        _open_remediation(b)
        with _client(b) as c:
            [incident] = c.get("/remediations").json()
            assert incident["occurrences"] == 2
            r = c.post(
                f"/remediations/{incident['incident_id']}/resolve",
                json={"resolution": "content_ingested"},
            )
            assert r.status_code == 200
        with _client(a) as c:
            [seen] = c.get("/remediations").json()
            assert seen["status"] == RemediationStatus.RESOLVED
            again = c.post(
                f"/remediations/{incident['incident_id']}/resolve",
                json={"resolution": "x"},
            )
            assert again.status_code == 409
            assert c.post("/remediations/nope/resolve", json={"resolution": "x"}).status_code == 404
        assert a.remediation_incidents == {}  # nothing leaked into the local fallback


# ---------------------------------------------------------------------------
# Redis outage: degrade, never crash
# ---------------------------------------------------------------------------


class TestRedisDown:
    def test_service_keeps_working_without_redis(
        self, replicas: tuple[AppState, AppState], server: fakeredis.FakeServer
    ) -> None:
        a, _ = replicas
        server.connected = False
        with patch("rag.redis_health.logger"), _client(a) as c:
            assert c.post("/query", json={"query": "alpha"}).status_code == 200
            assert c.get("/drift").status_code == 200
            assert c.get("/remediations").json() == []
            _open_remediation(a)  # falls back to the local store
            [local] = c.get("/remediations").json()
            assert (
                c.post(
                    f"/remediations/{local['incident_id']}/resolve", json={"resolution": "x"}
                ).status_code
                == 200
            )
            metrics = c.get("/metrics").text
            status = c.get("/index/status").json()
        assert "rag_redis_up 0.0" in metrics
        assert 'rag_redis_errors_total{component="cache"}' in metrics
        assert status["redis_up"] is False and status["shared_version"] is None

    def test_redis_failure_mid_call_falls_back(self, replicas: tuple[AppState, AppState]) -> None:
        a, _ = replicas
        broken = MagicMock()
        broken.open_or_dedupe.side_effect = ConnectionError("gone")
        broken.list.side_effect = ConnectionError("gone")
        broken.resolve.side_effect = ConnectionError("gone")
        a.remediation_store = broken
        with patch("rag.redis_health.logger"), _client(a) as c:
            _open_remediation(a)
            assert len(a.remediation_incidents) == 1
            a.redis._up = True  # type: ignore[union-attr]
            assert len(c.get("/remediations").json()) == 1
            a.redis._up = True  # type: ignore[union-attr]
            r = c.post("/remediations/missing/resolve", json={"resolution": "x"})
        assert r.status_code == 404


class TestConnect:
    def test_no_url_means_no_redis(self) -> None:
        assert _connect_redis(RedisConfig()) is None

    def test_unreachable_redis_starts_degraded(self) -> None:
        server = fakeredis.FakeServer()
        server.connected = False
        with patch("rag.redis_health.logger"):
            health = _connect_redis(
                RedisConfig(url="redis://x"),
                client_factory=lambda _c: fakeredis.FakeRedis(server=server),
            )
        assert health is not None and health.up is False

    def test_reachable_redis_starts_up(self) -> None:
        health = _connect_redis(
            RedisConfig(url="redis://x"), client_factory=lambda _c: fakeredis.FakeRedis()
        )
        assert health is not None and health.up

    def test_attach_without_redis_is_a_no_op(self, tmp_path: Path) -> None:
        state = _state(fakeredis.FakeServer(), tmp_path / "i")
        state.redis, state.index_sync, state.remediation_store = None, None, None
        _attach_shared_services(state)
        assert state.index_sync is None and state.remediation_store is None

    def test_shutdown_stops_index_sync(self) -> None:
        state = _state(fakeredis.FakeServer(), Path("unused"))
        sync = MagicMock()
        state.index_sync = sync
        with _client(state):
            pass
        sync.stop.assert_called_once()


# ---------------------------------------------------------------------------
# Jobs visible on every replica
# ---------------------------------------------------------------------------


class TestSharedJobs:
    def test_job_run_on_one_replica_is_readable_on_the_other(
        self, replicas: tuple[AppState, AppState], tmp_path: Path
    ) -> None:
        a, b = replicas
        job = _new_job(a)
        with _client(b) as c:
            assert c.get(f"/jobs/{job}").json()["status"] == JobStatusEnum.PENDING
        _run_ingest(a, job, [_doc(tmp_path, "eps.txt")], a.ingest_config, [])
        with _client(b) as c:
            body = c.get(f"/jobs/{job}").json()
        assert body["status"] == JobStatusEnum.DONE
        assert body["completed_at"] is not None
        assert job not in b.jobs  # served from Redis, not b's memory

    def test_failed_job_error_is_shared(self, replicas: tuple[AppState, AppState]) -> None:
        a, b = replicas
        job = _new_job(a)
        with patch.object(a.store, "list_chunks", side_effect=RuntimeError("disk gone")):
            _run_reindex(a, job)
        with _client(b) as c:
            body = c.get(f"/jobs/{job}").json()
        assert body["status"] == JobStatusEnum.ERROR and body["error"] == "disk gone"

    def test_unknown_job_is_404_everywhere(self, replicas: tuple[AppState, AppState]) -> None:
        for s in replicas:
            with _client(s) as c:
                assert c.get("/jobs/nope").status_code == 404

    def test_jobs_fall_back_to_local_while_redis_is_down(
        self, replicas: tuple[AppState, AppState], server: fakeredis.FakeServer
    ) -> None:
        a, b = replicas
        server.connected = False
        with patch("rag.redis_health.logger"):
            job = _new_job(a)  # write fails -> local only, never raises
            assert a.redis is not None and not a.redis.up
            with _client(a) as c:
                assert c.get(f"/jobs/{job}").json()["job_id"] == job
            with _client(b) as c:
                assert c.get(f"/jobs/{job}").status_code == 404  # documented limitation

    def test_read_failure_mid_call_falls_back(self, replicas: tuple[AppState, AppState]) -> None:
        a, _ = replicas
        job = _new_job(a)
        with (
            patch.object(a.job_store, "get", side_effect=ConnectionError("gone")),
            patch("rag.redis_health.logger"),
            _client(a) as c,
        ):
            assert c.get(f"/jobs/{job}").json()["job_id"] == job
        assert a.redis is not None and not a.redis.up

    def test_job_keys_expire(
        self, replicas: tuple[AppState, AppState], server: fakeredis.FakeServer
    ) -> None:
        a, _ = replicas
        job = _new_job(a)
        assert fakeredis.FakeRedis(server=server).ttl(f"rag:job:{job}") > 0


def test_empty_seed_env_still_seeds(
    server: fakeredis.FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Copying .env.example to .env leaves SEED_SAMPLE_DATA= empty; that must
    mean the default (seed), not "disabled"."""
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "doc.md").write_text(" ".join(f"w{i}" for i in range(30)))
    monkeypatch.setenv("SAMPLES_DIR", str(samples))
    monkeypatch.setenv("SEED_SAMPLE_DATA", "")
    state = _state(server, tmp_path / "faiss.index")
    _bootstrap_index(state)
    assert state.store.list_chunks()
    if state.drift_scheduler is not None:
        state.drift_scheduler.shutdown(wait=False)
