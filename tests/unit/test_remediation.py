"""Remediation rules and the Redis-backed shared incident store."""

from __future__ import annotations

from datetime import datetime, timedelta

import fakeredis
import pytest

from rag.models import RemediationIncident, RemediationStatus
from rag.remediation import (
    REMEDIATIONS_KEY,
    IncidentAlreadyResolvedError,
    RedisRemediationStore,
    decide_remediation,
    resolved_copy,
)

NOW = datetime(2026, 9, 27, 12, 0, 0)


def _incident(
    incident_id: str = "inc-1",
    *,
    status: RemediationStatus = RemediationStatus.OPEN,
    updated_at: datetime = NOW,
) -> RemediationIncident:
    return RemediationIncident(
        incident_id=incident_id, status=status, opened_at=updated_at, updated_at=updated_at
    )


def _new() -> RemediationIncident:
    return _incident("fresh")


class TestDecide:
    def test_opens_when_nothing_exists(self) -> None:
        d = decide_remediation([], NOW, 3600, _new)
        assert d.action == "opened" and d.incident is not None
        assert d.incident.incident_id == "fresh"

    def test_dedupes_into_the_most_recent_open_incident(self) -> None:
        older = _incident("a", updated_at=NOW - timedelta(hours=2))
        newer = _incident("b", updated_at=NOW - timedelta(hours=1))
        d = decide_remediation([older, newer], NOW, 3600, _new)
        assert d.action == "deduplicated"
        assert d.incident is not None
        assert d.incident.incident_id == "b"
        assert d.incident.occurrences == 2
        assert d.incident.updated_at == NOW
        assert newer.occurrences == 1  # input not mutated

    def test_cooldown_suppresses_after_recent_resolution(self) -> None:
        closed = _incident(status=RemediationStatus.RESOLVED, updated_at=NOW - timedelta(minutes=5))
        d = decide_remediation([closed], NOW, 3600, _new)
        assert d.action == "cooldown_suppressed"
        assert d.incident is None
        assert d.age_s == pytest.approx(300)

    def test_opens_after_cooldown(self) -> None:
        closed = _incident(status=RemediationStatus.RESOLVED, updated_at=NOW - timedelta(hours=2))
        assert decide_remediation([closed], NOW, 3600, _new).action == "opened"

    def test_resolved_copy_rejects_double_resolution(self) -> None:
        done = resolved_copy(_incident(), "content_ingested", "n", NOW)
        assert done.status is RemediationStatus.RESOLVED
        assert done.resolution == "content_ingested" and done.notes == "n"
        with pytest.raises(IncidentAlreadyResolvedError):
            resolved_copy(done, "again", None, NOW)


@pytest.fixture()
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


def _store(server: fakeredis.FakeServer) -> RedisRemediationStore:
    return RedisRemediationStore(fakeredis.FakeRedis(server=server))


class TestRedisStore:
    def test_open_then_dedupe_across_replicas(self, server: fakeredis.FakeServer) -> None:
        a, b = _store(server), _store(server)
        assert a.open_or_dedupe(NOW, 3600, _new).action == "opened"
        d = b.open_or_dedupe(NOW + timedelta(seconds=1), 3600, _new)
        assert d.action == "deduplicated"
        [only] = a.list()
        assert only.occurrences == 2
        assert a.list() == b.list()

    def test_cooldown_writes_nothing(self, server: fakeredis.FakeServer) -> None:
        store = _store(server)
        store.open_or_dedupe(NOW, 3600, _new)
        store.resolve("fresh", "false_positive", None, NOW)
        d = store.open_or_dedupe(NOW + timedelta(seconds=10), 3600, _new)
        assert d.action == "cooldown_suppressed"
        assert len(store.list()) == 1

    def test_concurrent_write_forces_a_retry(self, server: fakeredis.FakeServer) -> None:
        """A replica writing between WATCH and EXEC must abort and recompute,
        so two simultaneous AUTO alarms never open two incidents."""
        store = _store(server)
        other = fakeredis.FakeRedis(server=server)
        calls = {"n": 0}

        def racing_new() -> RemediationIncident:
            calls["n"] += 1
            if calls["n"] == 1:
                # Another replica opens its incident while we are deciding.
                other.hset(REMEDIATIONS_KEY, "theirs", _incident("theirs").model_dump_json())
            return _incident(f"ours-{calls['n']}")

        d = store.open_or_dedupe(NOW, 3600, racing_new)
        assert d.action == "deduplicated"
        assert d.incident is not None and d.incident.incident_id == "theirs"
        assert [i.incident_id for i in store.list()] == ["theirs"]

    def test_resolve_unknown_and_twice(self, server: fakeredis.FakeServer) -> None:
        store = _store(server)
        with pytest.raises(KeyError):
            store.resolve("missing", "x", None, NOW)
        store.open_or_dedupe(NOW, 3600, _new)
        done = store.resolve("fresh", "content_ingested", "added docs", NOW)
        assert done.status is RemediationStatus.RESOLVED
        with pytest.raises(IncidentAlreadyResolvedError):
            store.resolve("fresh", "again", None, NOW)

    def test_resolve_retries_on_concurrent_write(self, server: fakeredis.FakeServer) -> None:
        store = _store(server)
        store.open_or_dedupe(NOW, 3600, _new)
        other = fakeredis.FakeRedis(server=server)
        real_hget = fakeredis.FakeRedis.hget
        calls = {"n": 0}

        def racing_hget(self: object, *args: object) -> object:
            calls["n"] += 1
            if calls["n"] == 1:
                other.hset(REMEDIATIONS_KEY, "unrelated", _incident("unrelated").model_dump_json())
            return real_hget(self, *args)  # type: ignore[arg-type]

        from unittest.mock import patch

        with patch("redis.client.Pipeline.hget", racing_hget):
            done = store.resolve("fresh", "content_ingested", None, NOW)
        assert calls["n"] == 2
        assert done.status is RemediationStatus.RESOLVED
