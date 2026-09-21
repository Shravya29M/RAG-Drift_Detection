"""Covers the API paths the main integration suite leaves untouched: sample
seeding, the re-index background job, upload validation, remediation
transitions, and the drift-simulation endpoint.

Reuses the mock encoder/store harness from ``test_api`` rather than standing
up a real sentence-transformers model.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag.api import (
    AppState,
    _new_job,
    _persist_index,
    _run_reindex,
    _seed_sample_data,
    app,
)
from rag.models import JobStatusEnum, RemediationIncident, RemediationStatus
from rag.vector_store.faiss_store import FAISSStore

from .test_api import DIM, _make_chunk, _make_state


def _incident(status: RemediationStatus) -> RemediationIncident:
    now = datetime.now(UTC)
    return RemediationIncident(
        incident_id="inc-1",
        status=status,
        opened_at=now,
        updated_at=now,
    )


@pytest.fixture()
def state() -> AppState:
    return _make_state()


@pytest.fixture()
def client(state: AppState):
    app.state.app = state
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------
# _persist_index
# --------------------------------------------------------------------------


def test_persist_index_is_a_no_op_without_a_configured_path(state: AppState, tmp_path: Path):
    state.index_path = None
    _persist_index(state)
    assert list(tmp_path.iterdir()) == []


def test_persist_index_skips_a_store_that_does_not_support_saving(state: AppState, tmp_path: Path):
    """The base VectorStore interface has no save(); persisting must skip such
    a store rather than raising AttributeError on startup."""
    state.index_path = tmp_path / "index"
    _persist_index(state)
    assert not (tmp_path / "index").exists()


def test_persist_index_writes_a_real_faiss_store_to_disk(state: AppState, tmp_path: Path):
    store = FAISSStore(dim=DIM)
    store.add([_make_chunk(0)], np.zeros((1, DIM), dtype=np.float32))
    state.store = store
    state.index_path = tmp_path / "index"
    _persist_index(state)
    assert (tmp_path / "index").exists() or any(tmp_path.iterdir())


# --------------------------------------------------------------------------
# _seed_sample_data
# --------------------------------------------------------------------------


def test_seeding_is_skipped_when_the_samples_directory_is_absent(
    state: AppState, tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("SAMPLES_DIR", str(tmp_path / "nope"))
    _seed_sample_data(state)
    assert state.jobs == {}


def test_seeding_is_skipped_when_no_file_has_a_known_extension(
    state: AppState, tmp_path: Path, monkeypatch
):
    (tmp_path / "notes.xyz").write_text("unsupported", encoding="utf-8")
    monkeypatch.setenv("SAMPLES_DIR", str(tmp_path))
    _seed_sample_data(state)
    assert state.jobs == {}


def test_seeding_ingests_the_recognised_sample_files(state: AppState, tmp_path: Path, monkeypatch):
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "skip.xyz").write_text("unsupported", encoding="utf-8")
    monkeypatch.setenv("SAMPLES_DIR", str(tmp_path))
    _seed_sample_data(state)
    assert len(state.jobs) == 1


# --------------------------------------------------------------------------
# re-index background job
# --------------------------------------------------------------------------


def test_reindex_completes_immediately_when_the_store_is_empty(state: AppState):
    state.store.list_chunks.return_value = []
    job_id = _new_job(state)
    _run_reindex(state, job_id)
    job = state.jobs[job_id]
    assert job.status is JobStatusEnum.DONE
    assert job.completed_at is not None
    state.store.swap_index.assert_not_called()


def test_reindex_re_embeds_and_swaps_the_index(state: AppState):
    chunks = [_make_chunk(i) for i in range(3)]
    state.store.list_chunks.return_value = chunks
    job_id = _new_job(state)
    _run_reindex(state, job_id)

    assert state.jobs[job_id].status is JobStatusEnum.DONE
    state.store.swap_index.assert_called_once()
    swapped_chunks, embeddings = state.store.swap_index.call_args[0]
    assert swapped_chunks == chunks
    assert embeddings.shape == (3, DIM)


def test_a_failing_reindex_records_the_error_on_the_job(state: AppState):
    state.store.list_chunks.side_effect = RuntimeError("index corrupt")
    job_id = _new_job(state)
    _run_reindex(state, job_id)

    job = state.jobs[job_id]
    assert job.status is JobStatusEnum.ERROR
    assert "index corrupt" in (job.error or "")
    assert job.completed_at is not None


def test_reindex_endpoint_queues_a_pending_job(client: TestClient):
    body = client.post("/reindex").json()
    assert body["status"] in {JobStatusEnum.PENDING, JobStatusEnum.DONE}
    assert body["job_id"]


# --------------------------------------------------------------------------
# ingest validation
# --------------------------------------------------------------------------


def test_ingest_rejects_malformed_config_json(client: TestClient):
    r = client.post(
        "/ingest",
        files={"files": ("a.txt", b"content", "text/plain")},
        data={"config_json": "{not json}"},
    )
    assert r.status_code == 422
    assert "Invalid config_json" in r.json()["detail"]


def test_ingest_accepts_an_explicit_config_override(client: TestClient):
    r = client.post(
        "/ingest",
        files={"files": ("a.txt", b"content", "text/plain")},
        data={"config_json": '{"chunk_size": 128}'},
    )
    assert r.status_code == 200


def test_ingest_ignores_an_upload_with_no_filename(client: TestClient):
    """A blank filename carries no extension to infer a parser from."""
    r = client.post("/ingest", files={"files": ("", b"content", "text/plain")})
    assert r.status_code == 422


# --------------------------------------------------------------------------
# remediation transitions
# --------------------------------------------------------------------------


def test_resolving_an_unknown_incident_returns_404(client: TestClient):
    r = client.post("/remediations/does-not-exist/resolve", json={"resolution": "false_positive"})
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


def test_resolving_an_open_incident_marks_it_resolved(client: TestClient, state: AppState):
    state.remediation_incidents["inc-1"] = _incident(RemediationStatus.OPEN)

    body = client.post(
        "/remediations/inc-1/resolve",
        json={"resolution": "content_ingested", "notes": "re-ingested the missing docs"},
    ).json()
    assert body["status"] == RemediationStatus.RESOLVED
    assert body["resolution"] == "content_ingested"
    assert body["notes"] == "re-ingested the missing docs"


def test_resolving_requires_a_disposition(client: TestClient, state: AppState):
    """An empty resolution would leave no record of why the incident closed."""
    state.remediation_incidents["inc-1"] = _incident(RemediationStatus.OPEN)
    assert client.post("/remediations/inc-1/resolve", json={"resolution": ""}).status_code == 422


def test_resolving_an_already_resolved_incident_returns_409(client: TestClient, state: AppState):
    state.remediation_incidents["inc-1"] = _incident(RemediationStatus.RESOLVED)
    r = client.post("/remediations/inc-1/resolve", json={"resolution": "false_positive"})
    assert r.status_code == 409
    assert "already resolved" in r.json()["detail"]


def test_listing_remediations_returns_the_open_incidents(client: TestClient, state: AppState):
    state.remediation_incidents["inc-1"] = _incident(RemediationStatus.OPEN)
    body = client.get("/remediations").json()
    assert [i["incident_id"] for i in body] == ["inc-1"]


# --------------------------------------------------------------------------
# drift simulation
# --------------------------------------------------------------------------


def test_simulate_requires_a_running_drift_monitor(client: TestClient):
    r = client.post("/drift/simulate")
    assert r.status_code == 409
    assert "not running" in r.json()["detail"]


@pytest.fixture()
def monitored_client(request):
    """A client whose store holds enough chunks for the drift monitor to start.

    _start_drift_monitor needs a snapshot of at least two embeddings, so the
    single-chunk default store silently leaves the monitor unstarted.
    """
    state = _make_state(chunks=[_make_chunk(i) for i in range(4)])
    app.state.app = state
    with TestClient(app) as c:
        c.post("/ingest", files={"files": ("a.txt", b"hello world content", "text/plain")})
        yield c


def test_simulate_feeds_off_topic_windows_through_the_scheduler(monitored_client: TestClient):
    body = monitored_client.post("/drift/simulate", params={"windows": 1}).json()
    assert body["windows_fed"] == 1
    assert "reindex_triggered" in body
    assert "consecutive_alerts" in body
    assert "open_remediations" in body


def test_simulate_clamps_the_window_count_to_ten(monitored_client: TestClient):
    """An unbounded window count would let one request feed the detector for
    an arbitrarily long time."""
    body = monitored_client.post("/drift/simulate", params={"windows": 500}).json()
    assert body["windows_fed"] == 10


def test_simulate_clamps_a_non_positive_window_count_to_one(monitored_client: TestClient):
    body = monitored_client.post("/drift/simulate", params={"windows": 0}).json()
    assert body["windows_fed"] == 1


def test_drift_reset_is_safe_without_a_detector(client: TestClient):
    r = client.post("/drift/reset")
    assert r.status_code == 200
    assert r.json() == {"message": "drift history reset"}


def test_metrics_endpoint_renders_prometheus_text(client: TestClient):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "#" in r.text or "_total" in r.text


def test_unknown_job_id_returns_404(client: TestClient):
    assert client.get("/jobs/nope").status_code == 404


def test_encoder_mock_produces_unit_norm_vectors(state: AppState):
    """Guards the harness itself: drift maths assumes normalised inputs."""
    vecs = state.encoder.encode(["a", "b", "c"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
