"""Remediation incident rules plus the Redis store that shares them across replicas.

The dedupe/cooldown decision is a pure function so the in-process dict (used
when Redis is absent or down) and the Redis hash apply identical rules.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import redis

from rag.models import RemediationIncident, RemediationStatus

REMEDIATIONS_KEY = "rag:remediations"


class IncidentAlreadyResolvedError(Exception):
    """Raised when resolving an incident that is already resolved."""


@dataclass(frozen=True)
class RemediationDecision:
    """What an AUTO escalation did: dedupe into an open incident, stay quiet, or open one."""

    action: str
    """``"deduplicated"``, ``"cooldown_suppressed"`` or ``"opened"``."""
    incident: RemediationIncident | None = None
    """The incident to write back (``None`` when suppressed)."""
    age_s: float | None = None
    """Seconds since the last incident closed, for ``cooldown_suppressed``."""


def decide_remediation(
    incidents: Iterable[RemediationIncident],
    now: datetime,
    cooldown_s: float,
    new_incident: Callable[[], RemediationIncident],
) -> RemediationDecision:
    """Apply the dedupe and cooldown rules to the current incident set.

    An open incident absorbs repeated alarms; after resolution, a cooldown
    prevents an immediate re-opening.
    """
    existing = list(incidents)
    open_incidents = [i for i in existing if i.status is RemediationStatus.OPEN]
    if open_incidents:
        latest_open = max(open_incidents, key=lambda item: item.updated_at)
        return RemediationDecision(
            "deduplicated",
            latest_open.model_copy(
                update={"occurrences": latest_open.occurrences + 1, "updated_at": now}
            ),
        )
    latest_closed = max(existing, key=lambda item: item.updated_at, default=None)
    if latest_closed is not None:
        age_s = (now - latest_closed.updated_at).total_seconds()
        if age_s < cooldown_s:
            return RemediationDecision("cooldown_suppressed", age_s=age_s)
    return RemediationDecision("opened", new_incident())


def resolved_copy(
    incident: RemediationIncident, resolution: str, notes: str | None, now: datetime
) -> RemediationIncident:
    """Return *incident* marked resolved, or raise if it already is."""
    if incident.status is RemediationStatus.RESOLVED:
        raise IncidentAlreadyResolvedError(incident.incident_id)
    return incident.model_copy(
        update={
            "status": RemediationStatus.RESOLVED,
            "resolution": resolution,
            "notes": notes,
            "updated_at": now,
        }
    )


class RedisRemediationStore:
    """Remediation incidents in one Redis hash, updated with optimistic locking.

    Every read-modify-write runs under ``WATCH``; a concurrent change from
    another replica aborts the transaction and the decision is recomputed.
    """

    def __init__(self, client: Any, key: str = REMEDIATIONS_KEY) -> None:
        self._client = client
        self._key = key

    def list(self) -> list[RemediationIncident]:
        raw: list[bytes] = self._client.hvals(self._key)
        return [RemediationIncident.model_validate_json(v) for v in raw]

    def open_or_dedupe(
        self,
        now: datetime,
        cooldown_s: float,
        new_incident: Callable[[], RemediationIncident],
    ) -> RemediationDecision:
        with self._client.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(self._key)
                    incidents = [
                        RemediationIncident.model_validate_json(v) for v in pipe.hvals(self._key)
                    ]
                    decision = decide_remediation(incidents, now, cooldown_s, new_incident)
                    pipe.multi()
                    if decision.incident is not None:
                        pipe.hset(
                            self._key,
                            decision.incident.incident_id,
                            decision.incident.model_dump_json(),
                        )
                    pipe.execute()
                    return decision
                except redis.WatchError:
                    continue

    def resolve(
        self, incident_id: str, resolution: str, notes: str | None, now: datetime
    ) -> RemediationIncident:
        """Resolve one incident.

        Raises:
            KeyError: Unknown incident.
            IncidentAlreadyResolvedError: Already resolved.
        """
        with self._client.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(self._key)
                    raw = pipe.hget(self._key, incident_id)
                    if raw is None:
                        raise KeyError(incident_id)
                    updated = resolved_copy(
                        RemediationIncident.model_validate_json(raw), resolution, notes, now
                    )
                    pipe.multi()
                    pipe.hset(self._key, incident_id, updated.model_dump_json())
                    pipe.execute()
                    return updated
                except redis.WatchError:
                    continue
