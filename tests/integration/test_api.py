"""Integration tests for rag/api.py using FastAPI TestClient."""

from __future__ import annotations

import json
from collections.abc import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag.api import AppState, app
from rag.drift.detector import DriftDetector
from rag.drift.scheduler import DriftScheduler
from rag.embedding.encoder import Encoder
from rag.generation.llm import LLMRouter
from rag.models import (
    Chunk,
    ChunkMetadata,
    DriftConfig,
    GenerationConfig,
    IngestConfig,
    SourceType,
)
from rag.retrieval.retriever import Retriever
from rag.vector_store.base import VectorStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM = 8


# ---------------------------------------------------------------------------
# Mock implementations
# ---------------------------------------------------------------------------


class _MockEncoder(Encoder):
    """Returns deterministic L2-normalised random vectors."""

    def encode(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(seed=len(texts))
        vecs = rng.random((len(texts), DIM)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1.0, norms)


class _MockRouter(LLMRouter):
    """Returns a static answer string."""

    def complete(self, prompt: str) -> str:
        return "Test answer."


def _make_chunk(idx: int = 0) -> Chunk:
    return Chunk(
        id=f"src-{idx}",
        text=f"chunk text {idx}",
        token_count=3,
        metadata=ChunkMetadata(
            source="test.txt",
            source_type=SourceType.TEXT,
            chunk_index=idx,
        ),
    )


def _mock_store(chunks: list[Chunk] | None = None) -> VectorStore:
    """Return a VectorStore mock pre-loaded with optional chunks."""
    store = MagicMock(spec=VectorStore)
    _chunks = chunks or [_make_chunk(0)]
    vecs = np.zeros((len(_chunks), DIM), dtype=np.float32)
    store.search.return_value = []
    store.snapshot_distribution.return_value = vecs
    store.list_chunks.return_value = _chunks
    return store


def _mock_retriever(chunks: list[Chunk] | None = None) -> Retriever:
    from rag.models import RetrievalResult

    retriever = MagicMock(spec=Retriever)
    _chunks = chunks or [_make_chunk(0)]
    retriever.retrieve.return_value = RetrievalResult(
        query="test",
        chunks=_chunks,
        scores=[0.9] * len(_chunks),
        latency_ms=5.0,
        total_candidates=len(_chunks),
    )
    return retriever


def _make_state(
    *,
    chunks: list[Chunk] | None = None,
    drift_detector: DriftDetector | None = None,
    drift_scheduler: DriftScheduler | None = None,
) -> AppState:
    return AppState(
        encoder=_MockEncoder(),
        store=_mock_store(chunks),
        retriever=_mock_retriever(chunks),
        llm_router=_MockRouter(),
        generation_config=GenerationConfig(),
        ingest_config=IngestConfig(),
        drift_config=DriftConfig(window_size=2, pca_components=2, hysteresis_windows=2),
        drift_detector=drift_detector,
        drift_scheduler=drift_scheduler,
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    app.state.app = _make_state()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /healthz
# ---------------------------------------------------------------------------


class TestHealthz:
    def test_returns_200(self, client: TestClient) -> None:
        r = client.get("/healthz")
        assert r.status_code == 200

    def test_body_is_ok(self, client: TestClient) -> None:
        r = client.get("/healthz")
        assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------


class TestIngest:
    def test_happy_path_file_upload(self, client: TestClient) -> None:
        r = client.post(
            "/ingest",
            files={"files": ("hello.txt", b"Hello world content", "text/plain")},
        )
        assert r.status_code == 200
        body = r.json()
        assert "job_id" in body
        assert body["message"] == "ingestion queued"

    def test_happy_path_url_list(self, client: TestClient) -> None:
        r = client.post(
            "/ingest",
            data={"urls": json.dumps(["https://example.com/doc"])},
        )
        assert r.status_code == 200
        assert "job_id" in r.json()

    def test_no_sources_returns_422(self, client: TestClient) -> None:
        r = client.post("/ingest")
        assert r.status_code == 422

    def test_invalid_urls_json_returns_422(self, client: TestClient) -> None:
        r = client.post("/ingest", data={"urls": "not-json"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_happy_path(self, client: TestClient) -> None:
        r = client.post("/query", json={"query": "What is drift?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "Test answer."
        assert isinstance(body["chunks"], list)
        assert isinstance(body["scores"], list)
        assert body["latency_ms"] >= 0

    def test_validates_query_model(self, client: TestClient) -> None:
        r = client.post("/query", json={"query": "", "k": 3})
        assert r.status_code == 422

    def test_enqueues_to_scheduler_when_present(self) -> None:
        scheduler = MagicMock(spec=DriftScheduler)
        app.state.app = _make_state(drift_scheduler=scheduler)
        with TestClient(app) as c:
            r = c.post("/query", json={"query": "hello"})
        assert r.status_code == 200
        scheduler.enqueue_embedding.assert_called_once()


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}
# ---------------------------------------------------------------------------


class TestJobs:
    def test_happy_path_returns_job(self, client: TestClient) -> None:
        ingest_r = client.post(
            "/ingest",
            files={"files": ("doc.txt", b"some text", "text/plain")},
        )
        job_id = ingest_r.json()["job_id"]
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200
        assert r.json()["job_id"] == job_id

    def test_unknown_id_returns_404(self, client: TestClient) -> None:
        r = client.get("/jobs/does-not-exist")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /drift
# ---------------------------------------------------------------------------


class TestDriftStatus:
    def test_no_detector_returns_empty(self, client: TestClient) -> None:
        r = client.get("/drift")
        assert r.status_code == 200
        body = r.json()
        assert body["history"] == []
        assert body["consecutive_alerts"] == 0
        assert body["reindex_triggered"] is False

    def test_with_detector_returns_state(self) -> None:

        from rag.drift.snapshot import DistributionSnapshot

        rng = np.random.default_rng(0)
        embs = rng.random((10, DIM)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / norms
        cfg = DriftConfig(window_size=2, pca_components=2, hysteresis_windows=2)
        snapshot = DistributionSnapshot(embs, cfg)
        detector = DriftDetector(snapshot, cfg)

        app.state.app = _make_state(drift_detector=detector)
        with TestClient(app) as c:
            r = c.get("/drift")
        assert r.status_code == 200
        body = r.json()
        assert "history" in body
        assert "buffer_size" in body


# ---------------------------------------------------------------------------
# POST /drift/reset
# ---------------------------------------------------------------------------


class TestDriftReset:
    def test_reset_with_no_detector_is_noop(self, client: TestClient) -> None:
        r = client.post("/drift/reset")
        assert r.status_code == 200
        assert r.json()["message"] == "drift history reset"

    def test_reset_calls_detector_reset(self) -> None:
        detector = MagicMock(spec=DriftDetector)
        app.state.app = _make_state(drift_detector=detector)
        with TestClient(app) as c:
            r = c.post("/drift/reset")
        assert r.status_code == 200
        detector.reset.assert_called_once()


# ---------------------------------------------------------------------------
# POST /reindex
# ---------------------------------------------------------------------------


class TestReindex:
    def test_returns_pending_job(self, client: TestClient) -> None:
        r = client.post("/reindex")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("pending", "running", "done", "error")
        assert "job_id" in body

    def test_job_registered(self, client: TestClient) -> None:
        r = client.post("/reindex")
        job_id = r.json()["job_id"]
        status_r = client.get(f"/jobs/{job_id}")
        assert status_r.status_code == 200


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_returns_200_plain_text(self, client: TestClient) -> None:
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]

    def test_contains_prometheus_keys(self, client: TestClient) -> None:
        r = client.get("/metrics")
        text = r.text
        assert "rag_scheduler_queue_size" in text
        assert "rag_jobs_total" in text
        assert "rag_drift_buffer_size" in text


# ---------------------------------------------------------------------------
# 503 when state not initialised
# ---------------------------------------------------------------------------


class TestUninitialised:
    def test_query_returns_503_when_no_state(self) -> None:
        app.state.app = _make_state()  # ensure lifespan guard skips init
        with TestClient(app) as c:
            app.state.app = None  # clear after lifespan startup has run
            r = c.post("/query", json={"query": "hello"})
        assert r.status_code == 503
        app.state.app = _make_state()
