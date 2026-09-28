"""Redis-backed job records so ``GET /jobs/{id}`` answers on every replica.

One key per job (``rag:job:<id>``) holding the ``JobStatus`` JSON, rewritten on
every status change and expiring ``ttl_s`` after the last one, so finished jobs
do not accumulate forever.
"""

from __future__ import annotations

from typing import Any

from rag.models import JobStatus

JOB_PREFIX = "rag:job"


class RedisJobStore:
    """Write-through job store; callers handle Redis failures (see api.py)."""

    def __init__(self, client: Any, *, ttl_s: int = 7 * 24 * 3600) -> None:
        self._client = client
        self._ttl_s = ttl_s

    def save(self, job: JobStatus) -> None:
        self._client.set(f"{JOB_PREFIX}:{job.job_id}", job.model_dump_json(), ex=self._ttl_s)

    def get(self, job_id: str) -> JobStatus | None:
        raw = self._client.get(f"{JOB_PREFIX}:{job_id}")
        return None if raw is None else JobStatus.model_validate_json(raw)
