"""FastAPI application: route definitions and HTTP wiring for all PRD Section 9 endpoints."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest

from rag.cache.query_cache import QueryCache
from rag.drift.alarm import DriftAlarm
from rag.drift.detector import DriftDetector, DriftMonitor
from rag.drift.redis_state import RedisDriftDetector, ResilientDriftMonitor
from rag.drift.scheduler import DriftScheduler
from rag.drift.snapshot import DistributionSnapshot
from rag.embedding.encoder import Encoder, SentenceTransformerEncoder
from rag.generation.llm import LLMRouter, make_router
from rag.generation.prompt import build_prompt
from rag.index_sync import IndexSync, index_version
from rag.ingestion.chunker import chunk_text
from rag.ingestion.metadata import infer_source_type
from rag.ingestion.parsers import parse
from rag.jobs import RedisJobStore
from rag.logging import get_logger
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
    RedisConfig,
    RemediationIncident,
    RemediationStatus,
    ResolveRemediationRequest,
    SourceType,
)
from rag.redis_health import REDIS_FAILURES, RedisHealth, make_client
from rag.remediation import (
    IncidentAlreadyResolvedError,
    RedisRemediationStore,
    RemediationDecision,
    decide_remediation,
    resolved_copy,
)
from rag.retrieval.retriever import Retriever
from rag.settings import load_settings
from rag.tracking import log_event
from rag.vector_store.base import VectorStore
from rag.vector_store.faiss_store import FAISSStore

logger = get_logger(__name__)

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
    alarm_config: AlarmConfig = field(default_factory=AlarmConfig)
    drift_interval_s: float = 30.0
    search_top_k: int = 5
    index_path: Path | None = None
    model_name: str = ""
    drift_detector: DriftMonitor | None = None
    drift_scheduler: DriftScheduler | None = None
    jobs: dict[str, JobStatus] = field(default_factory=dict)
    # Local incident store: the only one without Redis, the fallback while
    # Redis is down. With Redis up, remediation_store is authoritative.
    remediation_incidents: dict[str, RemediationIncident] = field(default_factory=dict)
    remediation_lock: threading.RLock = field(default_factory=threading.RLock)
    redis_config: RedisConfig = field(default_factory=RedisConfig)
    redis: RedisHealth | None = None
    query_cache: QueryCache | None = None
    remediation_store: RedisRemediationStore | None = None
    job_store: RedisJobStore | None = None
    index_sync: IndexSync | None = None


# ---------------------------------------------------------------------------
# FastAPI app + lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise service dependencies on startup; shut down on teardown.

    Skips initialisation if ``app.state.app`` is already set (e.g. in tests).
    """
    logger.info("lifespan: started")
    if getattr(app.state, "app", None) is None:
        logger.info("lifespan: initializing")
        try:
            settings = load_settings()

            encoder = SentenceTransformerEncoder(
                settings.embedding.model_name, batch_size=settings.embedding.batch_size
            )
            store = FAISSStore(dim=encoder.dim, config=settings.vector_store)
            model_name = settings.embedding.model_name
            health = _connect_redis(settings.redis)
            cache = (
                QueryCache(health, ttl_s=settings.cache.ttl_s)
                if health is not None and settings.cache.enabled
                else None
            )
            retriever = Retriever(
                store,
                encoder,
                score_threshold=settings.vector_store.score_threshold,
                cache=cache,
                index_version=lambda: index_version(store, model_name),
            )
            router = make_router(settings.generation)

            state = AppState(
                encoder=encoder,
                store=store,
                retriever=retriever,
                llm_router=router,
                generation_config=settings.generation,
                ingest_config=settings.ingestion,
                drift_config=settings.drift,
                alarm_config=settings.alarm,
                drift_interval_s=settings.scheduler.drift_check_interval_seconds,
                search_top_k=settings.vector_store.top_k,
                index_path=settings.vector_store.faiss_index_path,
                model_name=model_name,
                redis_config=settings.redis,
                redis=health,
                query_cache=cache,
            )
            app.state.app = state

            _attach_shared_services(state)
            _bootstrap_index(state)
            _start_drift_monitor(state)
            if state.index_sync is not None:
                state.index_sync.start()
            logger.info("lifespan: complete")
        except Exception:
            logger.exception("lifespan: initialization failed")
            raise

    yield

    final_state: AppState | None = getattr(app.state, "app", None)
    if final_state is not None:
        if final_state.index_sync is not None:
            final_state.index_sync.stop()
        if final_state.drift_scheduler is not None:
            final_state.drift_scheduler.shutdown(wait=False)


app = FastAPI(title="RAG Drift API", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# ---------------------------------------------------------------------------
# Redis-backed shared services and index versioning
# ---------------------------------------------------------------------------


def _connect_redis(
    config: RedisConfig, client_factory: Callable[[RedisConfig], Any] = make_client
) -> RedisHealth | None:
    """Return a health-tracked Redis client, or ``None`` when no URL is configured.

    An unreachable Redis at startup is not fatal: the service starts in local
    fallback mode and reconnects when Redis appears.
    """
    if not config.url:
        return None
    health = RedisHealth(client_factory(config), retry_interval_s=config.retry_interval_s)
    try:
        health.client.ping()
    except REDIS_FAILURES as exc:
        health.mark_down("startup", exc)
    return health


def _attach_shared_services(state: AppState) -> None:
    """Create the Redis remediation store and index sync for *state*."""
    if state.redis is None:
        return
    state.remediation_store = RedisRemediationStore(state.redis.client)
    state.job_store = RedisJobStore(state.redis.client)
    state.index_sync = IndexSync(
        state.redis,
        local_version=lambda: _index_version(state),
        reload=lambda: _reload_index(state),
        poll_interval_s=state.redis_config.index_poll_interval_s,
    )


def _index_version(state: AppState) -> str:
    return index_version(state.store, state.model_name)


@contextmanager
def _index_writer(state: AppState) -> Iterator[None]:
    """Cross-replica write lease around an index mutation (no-op without Redis)."""
    if state.index_sync is None:
        yield
        return
    with state.index_sync.writer():
        yield


def _load_index(state: AppState) -> bool:
    """Load the persisted index into the store; ``False`` if there is none."""
    if state.index_path is None or not isinstance(state.store, FAISSStore):
        return False
    return state.store.load(state.index_path)


def _reload_index(state: AppState) -> None:
    """IndexSync callback: pick up a newer index file written by another replica."""
    if _load_index(state):
        _start_drift_monitor(state)


def _bootstrap_index(state: AppState) -> None:
    """Load or seed the index under the write lease, then announce its version.

    Holding the lease means only the first of several replicas booting
    together seeds; the rest find the file it wrote and load it.
    """
    with _index_writer(state):
        if _load_index(state):
            logger.info("lifespan: restored index from disk")
        elif (os.environ.get("SEED_SAMPLE_DATA") or "true").lower() == "true":
            _seed_sample_data(state)
        if state.index_sync is not None:
            local = _index_version(state)
            if state.index_sync.remote_version() != local:
                state.index_sync.publish(local)


def _commit_index_change(state: AppState, old_version: str) -> None:
    """After a mutation: persist, announce, drop stale cache entries, recalibrate."""
    _persist_index(state)
    new_version = _index_version(state)
    if state.index_sync is not None:
        state.index_sync.publish(new_version)
    if state.query_cache is not None and new_version != old_version:
        state.query_cache.purge_version(old_version)
    # New content changes the PCA snapshot. Restart calibration against
    # the new corpus before evaluating more drift.
    _start_drift_monitor(state)


# ---------------------------------------------------------------------------
# Drift monitor lifecycle
# ---------------------------------------------------------------------------


def _start_drift_monitor(state: AppState) -> None:
    """(Re)build the drift detector + scheduler from the current index.

    With Redis, the detector state is shared under the current index version,
    so replicas serving the same index evaluate one common window stream.
    No-ops when the store holds fewer than 2 embeddings (nothing to snapshot).
    An already-running scheduler is replaced and shut down.
    """
    snap_embs = state.store.snapshot_distribution()
    if snap_embs.shape[0] < 2:
        return

    snapshot = DistributionSnapshot(snap_embs, state.drift_config)
    local = DriftDetector(snapshot, state.drift_config)
    detector: DriftMonitor = local
    if state.redis is not None:
        shared = RedisDriftDetector(
            state.redis.client,
            snapshot,
            state.drift_config,
            namespace=f"rag:drift:{_index_version(state)}",
            state_ttl_s=state.redis_config.drift_state_ttl_s,
        )
        detector = ResilientDriftMonitor(shared, local, state.redis)
    alarm = DriftAlarm(state.alarm_config, re_index_callback=lambda: _open_remediation(state))
    scheduler = DriftScheduler(
        detector,
        alarm,
        state.drift_config,
        drift_check_interval_s=state.drift_interval_s,
    )

    old = state.drift_scheduler
    state.drift_detector = detector
    state.drift_scheduler = scheduler
    scheduler.start()
    if old is not None:
        old.shutdown(wait=False)
    logger.info("drift monitor started")


def _use_redis_remediations(state: AppState) -> bool:
    return (
        state.remediation_store is not None and state.redis is not None and state.redis.available()
    )


def _open_remediation(state: AppState) -> None:
    """Open or deduplicate a human remediation request after AUTO escalation.

    Re-embedding unchanged chunks cannot add missing knowledge, so AUTO creates
    a work item instead of pretending a re-index repaired the corpus. An open
    incident absorbs repeated alarms; after resolution, a cooldown prevents an
    immediate re-opening. With Redis the incident set is shared, so every
    replica deduplicates against the same incidents.
    """
    now = datetime.utcnow()
    detector = state.drift_detector
    history = detector.history if detector is not None else []
    latest = history[-1] if history else None
    baseline_mean = detector.baseline_mean_score if detector is not None else None
    cooldown_s = state.alarm_config.auto_remediation_cooldown_s

    def new_incident() -> RemediationIncident:
        return RemediationIncident(
            incident_id=str(uuid.uuid4()),
            opened_at=now,
            updated_at=now,
            mean_top_score=latest.mean_top_score if latest else None,
            baseline_mean_score=baseline_mean,
            quality_degraded=latest.quality_degraded if latest else True,
        )

    decision: RemediationDecision | None = None
    if _use_redis_remediations(state):
        assert state.remediation_store is not None and state.redis is not None
        try:
            decision = state.remediation_store.open_or_dedupe(now, cooldown_s, new_incident)
        except REDIS_FAILURES as exc:
            state.redis.mark_down("remediations", exc)
    if decision is None:
        with state.remediation_lock:
            decision = decide_remediation(
                state.remediation_incidents.values(), now, cooldown_s, new_incident
            )
            if decision.incident is not None:
                state.remediation_incidents[decision.incident.incident_id] = decision.incident

    if decision.action == "cooldown_suppressed":
        log_event("remediation", {"action": "cooldown_suppressed", "age_s": decision.age_s})
        return
    assert decision.incident is not None
    if decision.action == "opened":
        logger.warning("quality degradation incident opened: %s", decision.incident.incident_id)
    log_event(
        "remediation", {"action": decision.action, "incident_id": decision.incident.incident_id}
    )


def _list_remediations(state: AppState) -> list[RemediationIncident]:
    """All incidents: the shared set when Redis is up, the local set otherwise."""
    if _use_redis_remediations(state):
        assert state.remediation_store is not None and state.redis is not None
        try:
            return state.remediation_store.list()
        except REDIS_FAILURES as exc:
            state.redis.mark_down("remediations", exc)
    with state.remediation_lock:
        return list(state.remediation_incidents.values())


def _persist_index(state: AppState) -> None:
    """Save the FAISS index to disk if persistence is configured."""
    if state.index_path is not None and isinstance(state.store, FAISSStore):
        state.store.save(state.index_path)


def _seed_sample_data(state: AppState) -> None:
    """Ingest the bundled sample documents so a fresh deployment isn't empty."""
    samples_dir = Path(os.environ.get("SAMPLES_DIR", "samples"))
    if not samples_dir.is_dir():
        return
    sources: list[tuple[str, SourceType]] = []
    for p in sorted(samples_dir.iterdir()):
        try:
            sources.append((str(p), infer_source_type(p)))
        except ValueError:
            continue
    if not sources:
        return
    logger.info("seeding index with %d sample document(s)", len(sources))
    _run_ingest(state, _new_job(state), sources, state.ingest_config, [])


# ---------------------------------------------------------------------------
# Helper: register + run a background job
# ---------------------------------------------------------------------------


def _save_job(state: AppState, job: JobStatus) -> None:
    """Record *job* locally and, when Redis is up, in the shared job store."""
    state.jobs[job.job_id] = job
    if state.job_store is None or state.redis is None or not state.redis.available():
        return
    try:
        state.job_store.save(job)
    except REDIS_FAILURES as exc:
        state.redis.mark_down("jobs", exc)


def _update_job(state: AppState, job_id: str, **fields: object) -> None:
    """Apply *fields* to a job and write it through to Redis."""
    _save_job(state, state.jobs[job_id].model_copy(update=fields))


def _get_job(state: AppState, job_id: str) -> JobStatus | None:
    """Shared record first (any replica may have run the job), local as fallback."""
    if state.job_store is not None and state.redis is not None and state.redis.available():
        try:
            job = state.job_store.get(job_id)
        except REDIS_FAILURES as exc:
            state.redis.mark_down("jobs", exc)
        else:
            if job is not None:
                return job
    return state.jobs.get(job_id)


def _new_job(state: AppState) -> str:
    """Create a PENDING job in the registry and return its ID."""
    job_id = str(uuid.uuid4())
    _save_job(
        state,
        JobStatus(job_id=job_id, status=JobStatusEnum.PENDING, created_at=datetime.utcnow()),
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
    _update_job(state, job_id, status=JobStatusEnum.RUNNING)
    try:
        all_chunks: list[Chunk] = []
        for source, source_type in sources:
            sections = parse(source, source_type)
            for section in sections:
                chunks = chunk_text(
                    section,
                    source,
                    source_type,
                    config,
                    tokenizer=state.encoder.tokenizer,
                    max_tokens=state.encoder.max_input_tokens,
                )
                all_chunks.extend(chunks)
        if all_chunks:
            texts = [c.text for c in all_chunks]
            # Encode outside the write lease: it is the slow step.
            embeddings = state.encoder.encode(texts)
            with _index_writer(state):
                old_version = _index_version(state)
                state.store.add(all_chunks, embeddings)
                _commit_index_change(state, old_version)
        _update_job(state, job_id, status=JobStatusEnum.DONE, completed_at=datetime.utcnow())
    except Exception as exc:  # noqa: BLE001
        _update_job(
            state,
            job_id,
            status=JobStatusEnum.ERROR,
            error=str(exc),
            completed_at=datetime.utcnow(),
        )
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)


_REINDEX_ATTEMPTS = 3


def _run_reindex(state: AppState, job_id: str) -> None:
    """Background task: re-embed all stored chunks and atomically swap the index.

    Embedding happens outside the write lease; if another replica changed the
    index meanwhile, the swap would drop its chunks, so the job retries
    against the new contents.
    """
    _update_job(state, job_id, status=JobStatusEnum.RUNNING)
    try:
        for _ in range(_REINDEX_ATTEMPTS):
            base_version = _index_version(state)
            chunks = state.store.list_chunks()
            if not chunks:
                break
            embeddings = state.encoder.encode([c.text for c in chunks])
            with _index_writer(state):
                if _index_version(state) != base_version:
                    continue
                state.store.swap_index(chunks, embeddings)
                _commit_index_change(state, base_version)
                break
        else:
            raise RuntimeError("index kept changing during re-index; retry later")

        _update_job(state, job_id, status=JobStatusEnum.DONE, completed_at=datetime.utcnow())
    except Exception as exc:  # noqa: BLE001
        _update_job(
            state,
            job_id,
            status=JobStatusEnum.ERROR,
            error=str(exc),
            completed_at=datetime.utcnow(),
        )


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
        # Cache hits still feed the drift monitor: repeated queries are part of
        # the real query distribution. Reuse the retriever's embedding.
        query_vec: np.ndarray = (
            np.asarray(result.query_embedding, dtype=np.float32)
            if result.query_embedding is not None
            else state.encoder.encode([body.query])[0]
        )
        # 0.0 for a no-hit query is itself a quality signal, not missing data.
        top_score = float(np.mean(result.scores)) if result.scores else 0.0
        state.drift_scheduler.enqueue_embedding(query_vec, top_score=top_score)

    log_event(
        "query",
        {
            "retrieval_hits": len(result.chunks),
            "total_candidates": result.total_candidates,
            "latency_ms": result.latency_ms,
            "cache_hit": int(result.cache_hit),
            "answer_chars": len(answer),
        },
    )

    return QueryResponse(
        answer=answer,
        chunks=result.chunks,
        scores=result.scores,
        latency_ms=result.latency_ms,
    )


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str, request: Request) -> JobStatus:
    """Return the status of an async ingest or re-index job.

    With Redis this answers on every replica, not only the one running the job.
    """
    state = get_state(request)
    job = _get_job(state, job_id)
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
        baseline_ready=det.baseline_ready,
        monitor_running=state.drift_scheduler is not None and state.drift_scheduler.is_running,
        baseline_mean_score=det.baseline_mean_score,
    )


@app.get("/remediations", response_model=list[RemediationIncident])
def list_remediations(request: Request, open_only: bool = False) -> list[RemediationIncident]:
    """List quality-degradation work items created by AUTO escalation."""
    state = get_state(request)
    incidents = _list_remediations(state)
    if open_only:
        incidents = [item for item in incidents if item.status is RemediationStatus.OPEN]
    return sorted(incidents, key=lambda item: item.opened_at, reverse=True)


@app.post("/remediations/{incident_id}/resolve", response_model=RemediationIncident)
def resolve_remediation(
    incident_id: str, payload: ResolveRemediationRequest, request: Request
) -> RemediationIncident:
    """Record an operator disposition after investigating a remediation alert.

    Use ``content_ingested`` only after POST /ingest has completed successfully;
    ingestion refreshes the index snapshot and restarts drift calibration.
    """
    state = get_state(request)
    now = datetime.utcnow()
    resolved: RemediationIncident | None = None
    try:
        if _use_redis_remediations(state):
            assert state.remediation_store is not None and state.redis is not None
            try:
                resolved = state.remediation_store.resolve(
                    incident_id, payload.resolution, payload.notes, now
                )
            except REDIS_FAILURES as exc:
                state.redis.mark_down("remediations", exc)
        if resolved is None:
            with state.remediation_lock:
                incident = state.remediation_incidents.get(incident_id)
                if incident is None:
                    raise KeyError(incident_id)
                resolved = resolved_copy(incident, payload.resolution, payload.notes, now)
                state.remediation_incidents[incident_id] = resolved
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found") from exc
    except IncidentAlreadyResolvedError as exc:
        raise HTTPException(
            status_code=409, detail=f"Incident '{incident_id}' is already resolved"
        ) from exc
    log_event("remediation", {"action": "resolved", "incident_id": incident_id})
    return resolved


# Off-topic phrases used by /drift/simulate to fabricate a query-distribution
# shift. Deliberately unrelated to any technical corpus.
_OFF_TOPIC_PHRASES: tuple[str, ...] = (
    "best slow cooker recipes for beef stew",
    "how to improve my marathon training pace",
    "top rated beach resorts in the maldives",
    "acoustic guitar chords for beginners",
    "when to plant tomatoes in a home garden",
    "latest football transfer rumours and scores",
    "easy sourdough bread starter instructions",
    "reviews of mystery novels published this year",
)


@app.post("/drift/simulate")
def drift_simulate(request: Request, windows: int = 3) -> dict[str, object]:
    """Demo endpoint: push off-topic query traffic through the drift monitor.

    Feeds one on-topic calibration window first (pseudo-queries sampled from
    indexed chunks) if the baseline is not yet captured, then *windows* full
    windows of off-topic queries — enough to walk the hysteresis counter up
    and trigger the tiered alarms. Processing is synchronous so the response
    reflects the final drift state.
    """
    state = get_state(request)
    det = state.drift_detector
    sched = state.drift_scheduler
    if det is None or sched is None:
        raise HTTPException(status_code=409, detail="Drift monitor not running; ingest first.")
    windows = max(1, min(windows, 10))
    window_size = int(state.drift_config.window_size)

    def _feed(texts: list[str]) -> None:
        # Run real retrieval per query so the quality gate sees genuine scores:
        # off-topic queries score low against the corpus, on-topic ones high.
        for vec in state.encoder.encode(texts):
            hits = state.store.search(vec, state.search_top_k)
            score = float(np.mean([h.score for h in hits])) if hits else 0.0
            sched.enqueue_embedding(vec, top_score=score)
        sched.process_now()

    if not det.baseline_ready:
        chunks = state.store.list_chunks()
        on_topic = [
            " ".join(chunks[i % len(chunks)].text.split()[:12])
            for i in range(window_size - det.buffer_size)
        ]
        _feed(on_topic)

    off_topic = [
        f"{_OFF_TOPIC_PHRASES[i % len(_OFF_TOPIC_PHRASES)]} variation {i}"
        for i in range(windows * window_size)
    ]
    _feed(off_topic)

    det = state.drift_detector
    return {
        "windows_fed": windows,
        "consecutive_alerts": det.consecutive_alerts if det else 0,
        "reindex_triggered": det.reindex_triggered if det else False,
        "open_remediations": sum(
            incident.status is RemediationStatus.OPEN for incident in _list_remediations(state)
        ),
        "history_length": len(det.history) if det else 0,
    }


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


@app.get("/index/status")
def index_status(request: Request) -> dict[str, object]:
    """This replica's index version and size, plus the shared version if any."""
    state = get_state(request)
    return {
        "index_version": _index_version(state),
        "shared_version": (
            state.index_sync.remote_version() if state.index_sync is not None else None
        ),
        "chunk_count": len(state.store.list_chunks()),
        "redis_up": state.redis.up if state.redis is not None else None,
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics(request: Request) -> str:
    """Prometheus metrics for this replica: drift state, jobs, cache, and Redis health.

    Values are per replica — scrape each replica, not the load balancer.
    """
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
    open_remediations = sum(
        1 for incident in _list_remediations(state) if incident.status is RemediationStatus.OPEN
    )

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
        "# HELP rag_jobs_total Background jobs created on this replica",
        "# TYPE rag_jobs_total counter",
        f"rag_jobs_total {jobs_total}",
        "# HELP rag_jobs_done Completed background jobs on this replica",
        "# TYPE rag_jobs_done counter",
        f"rag_jobs_done {jobs_done}",
        "# HELP rag_jobs_error Failed background jobs on this replica",
        "# TYPE rag_jobs_error counter",
        f"rag_jobs_error {jobs_error}",
        "# HELP rag_remediations_open Open quality-degradation remediation incidents",
        "# TYPE rag_remediations_open gauge",
        f"rag_remediations_open {open_remediations}",
    ]
    return "\n".join(lines) + "\n" + generate_latest().decode()
