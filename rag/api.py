"""FastAPI application: route definitions and HTTP wiring for all PRD Section 9 endpoints."""

from __future__ import annotations

import json
import tempfile
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse

from rag.drift.alarm import DriftAlarm
from rag.drift.detector import DriftDetector
from rag.drift.scheduler import DriftScheduler
from rag.drift.snapshot import DistributionSnapshot
from rag.embedding.encoder import Encoder
from rag.generation.llm import LLMRouter
from rag.generation.prompt import build_prompt
from rag.ingestion.chunker import chunk_text
from rag.ingestion.metadata import infer_source_type
from rag.ingestion.parsers import parse
from rag.models import (
    AlarmConfig,
    Chunk,
    DriftConfig,
    DriftStatus,
    GenerationConfig,
    IngestConfig,
    IngestResponse,
    JobStatus,
    JobStatusEnum,
    QueryRequest,
    QueryResponse,
    SourceType,
)
from rag.retrieval.retriever import Retriever
from rag.vector_store.base import VectorStore

# ---------------------------------------------------------------------------
# Application state container
# ---------------------------------------------------------------------------


@dataclass
class AppState:
    """All stateful dependencies injected into route handlers.

    Created once at startup (or in tests) and stored on ``app.state.app``.
    """

    encoder: Encoder
    store: VectorStore
    retriever: Retriever
    llm_router: LLMRouter
    generation_config: GenerationConfig
    ingest_config: IngestConfig
    drift_config: DriftConfig
    drift_detector: DriftDetector | None = None
    drift_scheduler: DriftScheduler | None = None
    jobs: dict[str, JobStatus] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FastAPI app + lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Shutdown the drift scheduler (if running) on app teardown."""
    yield
    state: AppState | None = getattr(app.state, "app", None)
    if state is not None and state.drift_scheduler is not None:
        state.drift_scheduler.shutdown(wait=False)


app = FastAPI(title="RAG Drift API", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


def get_state(request: Request) -> AppState:
    """Retrieve the :class:`AppState` stored on the FastAPI app.

    Raises:
        HTTPException 503: If the application state has not been initialised.
    """
    state: AppState | None = getattr(request.app.state, "app", None)
    if state is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    return state


_StateT = Annotated[AppState, get_state]


# ---------------------------------------------------------------------------
# Helper: register + run a background job
# ---------------------------------------------------------------------------


def _new_job(state: AppState) -> str:
    """Create a PENDING job in the registry and return its ID."""
    job_id = str(uuid.uuid4())
    state.jobs[job_id] = JobStatus(
        job_id=job_id,
        status=JobStatusEnum.PENDING,
        created_at=datetime.utcnow(),
    )
    return job_id


def _run_ingest(
    state: AppState,
    job_id: str,
    sources: list[tuple[str, SourceType]],
    config: IngestConfig,
    tmp_paths: list[Path],
) -> None:
    """Background task: parse, chunk, embed, and add to the vector store."""
    state.jobs[job_id].status = JobStatusEnum.RUNNING
    try:
        all_chunks: list[Chunk] = []
        for source, source_type in sources:
            sections = parse(source, source_type)
            for section in sections:
                chunks = chunk_text(section, source, source_type, config)
                all_chunks.extend(chunks)
        if all_chunks:
            texts = [c.text for c in all_chunks]
            embeddings = state.encoder.encode(texts)
            state.store.add(all_chunks, embeddings)
        state.jobs[job_id].status = JobStatusEnum.DONE
        state.jobs[job_id].completed_at = datetime.utcnow()
    except Exception as exc:  # noqa: BLE001
        state.jobs[job_id].status = JobStatusEnum.ERROR
        state.jobs[job_id].error = str(exc)
        state.jobs[job_id].completed_at = datetime.utcnow()
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)


def _run_reindex(state: AppState, job_id: str) -> None:
    """Background task: re-embed all stored chunks and atomically swap the index."""
    state.jobs[job_id].status = JobStatusEnum.RUNNING
    try:
        chunks = state.store.list_chunks()
        if not chunks:
            state.jobs[job_id].status = JobStatusEnum.DONE
            state.jobs[job_id].completed_at = datetime.utcnow()
            return
        texts = [c.text for c in chunks]
        embeddings = state.encoder.encode(texts)
        state.store.swap_index(chunks, embeddings)

        snap_embs = state.store.snapshot_distribution()
        if snap_embs.shape[0] >= state.drift_config.pca_components:
            snapshot = DistributionSnapshot(snap_embs, state.drift_config)
            detector = DriftDetector(snapshot, state.drift_config)
            alarm = DriftAlarm(AlarmConfig())
            scheduler = DriftScheduler(detector, alarm, state.drift_config)
            if state.drift_scheduler is not None:
                state.drift_scheduler.shutdown(wait=False)
            state.drift_detector = detector
            state.drift_scheduler = scheduler
            state.drift_scheduler.start()

        state.jobs[job_id].status = JobStatusEnum.DONE
        state.jobs[job_id].completed_at = datetime.utcnow()
    except Exception as exc:  # noqa: BLE001
        state.jobs[job_id].status = JobStatusEnum.ERROR
        state.jobs[job_id].error = str(exc)
        state.jobs[job_id].completed_at = datetime.utcnow()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness probe — always returns 200 if the process is alive."""
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: Request,
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File()] = [],  # noqa: B006
    urls: Annotated[str, Form()] = "[]",
    config_json: Annotated[str, Form()] = "{}",
) -> IngestResponse:
    """Ingest documents from file uploads and/or a list of URLs.

    Files are written to temporary paths and deleted after processing.
    The job runs in a background thread; poll ``GET /jobs/{job_id}`` for status.
    """
    state = get_state(request)

    try:
        url_list: list[str] = json.loads(urls)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid urls JSON: {exc}") from exc

    try:
        config = (
            IngestConfig.model_validate_json(config_json)
            if config_json != "{}"
            else state.ingest_config
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid config_json: {exc}") from exc

    sources: list[tuple[str, SourceType]] = []
    tmp_paths: list[Path] = []

    for upload in files:
        if not upload.filename:
            continue
        suffix = Path(upload.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await upload.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
        tmp_paths.append(tmp_path)
        source_type = infer_source_type(Path(upload.filename))
        sources.append((str(tmp_path), source_type))

    for url in url_list:
        sources.append((url, SourceType.URL))

    if not sources:
        raise HTTPException(status_code=422, detail="No files or URLs provided")

    job_id = _new_job(state)
    background_tasks.add_task(_run_ingest, state, job_id, sources, config, tmp_paths)
    return IngestResponse(job_id=job_id, message="ingestion queued")


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    """Query the RAG pipeline: retrieve → generate → return answer + sources.

    The query embedding is also enqueued to the drift scheduler (if running)
    so that drift detection tracks live query distributions.
    """
    state = get_state(request)

    result = state.retriever.retrieve(body.query, body.k, body.filters)
    prompt = build_prompt(body.query, result.chunks, state.generation_config)
    answer = state.llm_router.complete(prompt)

    if state.drift_scheduler is not None:
        query_vec: np.ndarray = state.encoder.encode([body.query])[0]
        state.drift_scheduler.enqueue_embedding(query_vec)

    return QueryResponse(
        answer=answer,
        chunks=result.chunks,
        scores=result.scores,
        latency_ms=result.latency_ms,
    )


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str, request: Request) -> JobStatus:
    """Return the status of an async ingest or re-index job."""
    state = get_state(request)
    job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@app.get("/drift", response_model=DriftStatus)
def drift_status(request: Request) -> DriftStatus:
    """Return current drift detection state: history, alert count, buffer size."""
    state = get_state(request)
    if state.drift_detector is None:
        return DriftStatus(
            history=[],
            consecutive_alerts=0,
            reindex_triggered=False,
            buffer_size=0,
        )
    det = state.drift_detector
    return DriftStatus(
        history=list(det.history),
        consecutive_alerts=det.consecutive_alerts,
        reindex_triggered=det.reindex_triggered,
        buffer_size=det.buffer_size,
    )


@app.post("/drift/reset")
def drift_reset(request: Request) -> dict[str, str]:
    """Reset drift detection history and hysteresis counter."""
    state = get_state(request)
    if state.drift_detector is not None:
        state.drift_detector.reset()
    return {"message": "drift history reset"}


@app.post("/reindex", response_model=JobStatus)
def reindex(request: Request, background_tasks: BackgroundTasks) -> JobStatus:
    """Trigger a manual re-index: re-embed all stored chunks and swap the index."""
    state = get_state(request)
    job_id = _new_job(state)
    background_tasks.add_task(_run_reindex, state, job_id)
    return state.jobs[job_id]


@app.get("/metrics", response_class=PlainTextResponse)
def metrics(request: Request) -> str:
    """Prometheus-compatible metrics for queue depth, drift state, and job counts."""
    state = get_state(request)

    scheduler_queue = state.drift_scheduler.queue_size if state.drift_scheduler is not None else 0
    buffer_size = state.drift_detector.buffer_size if state.drift_detector is not None else 0
    consecutive_alerts = (
        state.drift_detector.consecutive_alerts if state.drift_detector is not None else 0
    )
    reindex_triggered = (
        1 if (state.drift_detector is not None and state.drift_detector.reindex_triggered) else 0
    )

    jobs_total = len(state.jobs)
    jobs_done = sum(1 for j in state.jobs.values() if j.status == JobStatusEnum.DONE)
    jobs_error = sum(1 for j in state.jobs.values() if j.status == JobStatusEnum.ERROR)

    lines = [
        "# HELP rag_scheduler_queue_size Embeddings waiting in the drift scheduler queue",
        "# TYPE rag_scheduler_queue_size gauge",
        f"rag_scheduler_queue_size {scheduler_queue}",
        "# HELP rag_drift_buffer_size Query embeddings in the current drift window",
        "# TYPE rag_drift_buffer_size gauge",
        f"rag_drift_buffer_size {buffer_size}",
        "# HELP rag_drift_consecutive_alerts Consecutive drifted windows",
        "# TYPE rag_drift_consecutive_alerts gauge",
        f"rag_drift_consecutive_alerts {consecutive_alerts}",
        "# HELP rag_drift_reindex_triggered 1 if hysteresis threshold reached",
        "# TYPE rag_drift_reindex_triggered gauge",
        f"rag_drift_reindex_triggered {reindex_triggered}",
        "# HELP rag_jobs_total Total background jobs created",
        "# TYPE rag_jobs_total counter",
        f"rag_jobs_total {jobs_total}",
        "# HELP rag_jobs_done Completed background jobs",
        "# TYPE rag_jobs_done counter",
        f"rag_jobs_done {jobs_done}",
        "# HELP rag_jobs_error Failed background jobs",
        "# TYPE rag_jobs_error counter",
        f"rag_jobs_error {jobs_error}",
    ]
    return "\n".join(lines) + "\n"
