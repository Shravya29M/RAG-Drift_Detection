"""Shared (Redis) drift state: equivalence with the local detector, atomic
windowing across replicas, lease fencing, and local fallback."""

from __future__ import annotations

from unittest.mock import patch

import fakeredis
import numpy as np
import pytest

from rag.drift.detector import DriftDetector
from rag.drift.redis_state import (
    HISTORY_LIMIT,
    RedisDriftDetector,
    ResilientDriftMonitor,
    pack_sample,
    unpack_window,
)
from rag.drift.snapshot import DistributionSnapshot
from rag.models import DriftConfig
from rag.redis_health import RedisHealth

DIM = 16
NS = "rag:drift:test"


def _cfg(window: int = 10) -> DriftConfig:
    return DriftConfig(window_size=window, pca_components=4, hysteresis_windows=3)


def _snapshot(cfg: DriftConfig) -> DistributionSnapshot:
    rng = np.random.default_rng(0)
    return DistributionSnapshot(rng.normal(size=(60, DIM)).astype(np.float32), cfg)


def _traffic(n: int, *, shift_at: int, seed: int = 1) -> list[tuple[np.ndarray, float | None]]:
    """On-topic traffic, then a shifted distribution with degraded scores."""
    rng = np.random.default_rng(seed)
    out: list[tuple[np.ndarray, float | None]] = []
    for i in range(n):
        shifted = i >= shift_at
        vec = rng.normal(size=DIM).astype(np.float32) + (3.0 if shifted else 0.0)
        out.append((vec, 0.1 if shifted else 0.5))
    return out


def _outcomes(outs: list) -> list[tuple[bool, bool, bool]]:  # type: ignore[type-arg]
    return [(o.result.drifted, o.result.recalibrated, o.reindex_triggered) for o in outs]


@pytest.fixture()
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


def _shared(server: fakeredis.FakeServer, cfg: DriftConfig, **kw: float) -> RedisDriftDetector:
    return RedisDriftDetector(fakeredis.FakeRedis(server=server), _snapshot(cfg), cfg, NS, **kw)


class TestPacking:
    def test_round_trip_with_and_without_score(self) -> None:
        a = np.arange(DIM, dtype=np.float32)
        blob = pack_sample(a, 0.25) + pack_sample(a * 2, None)
        (e1, s1), (e2, s2) = unpack_window(blob, 2)
        np.testing.assert_array_equal(e1, a)
        np.testing.assert_array_equal(e2, a * 2)
        assert s1 == pytest.approx(0.25)
        assert s2 is None


class TestEquivalence:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_three_interleaved_replicas_match_one_local_detector(
        self, server: fakeredis.FakeServer, seed: int
    ) -> None:
        """Same samples, same order: shared state across 3 replicas must produce
        exactly the local detector's KS/hysteresis/quality-gate outcomes."""
        cfg = _cfg()
        traffic = _traffic(120, shift_at=40, seed=seed)
        replicas = [_shared(server, cfg) for _ in range(3)]
        local = DriftDetector(_snapshot(cfg), cfg)

        rng = np.random.default_rng(seed)
        shared_outs = []
        i = 0
        while i < len(traffic):
            size = int(rng.integers(1, 7))  # uneven batches, as scheduler ticks produce
            replica = replicas[int(rng.integers(0, 3))]
            shared_outs += replica.add_query_embeddings(traffic[i : i + size])
            i += size
        local_outs = local.add_query_embeddings(traffic)

        assert _outcomes(shared_outs) == _outcomes(local_outs)
        assert [r.statistic for r in replicas[0].history] == [r.statistic for r in local.history]
        for r in replicas:
            assert r.consecutive_alerts == local.consecutive_alerts
            assert r.reindex_triggered == local.reindex_triggered is True
            assert r.baseline_mean_score == pytest.approx(local.baseline_mean_score)
            assert r.buffer_size == local.buffer_size

    def test_benign_shift_recalibrates_like_local(self, server: fakeredis.FakeServer) -> None:
        cfg = _cfg()
        rng = np.random.default_rng(5)
        traffic = [
            (rng.normal(size=DIM).astype(np.float32) + (3.0 if i >= 20 else 0.0), 0.5)
            for i in range(80)
        ]
        shared = _shared(server, cfg)
        local = DriftDetector(_snapshot(cfg), cfg)
        shared_outs = shared.add_query_embeddings(traffic)
        assert _outcomes(shared_outs) == _outcomes(local.add_query_embeddings(traffic))
        assert any(o.result.recalibrated for o in shared_outs)
        assert shared.reindex_triggered is False

    def test_no_score_samples_fall_back_to_drift_only(self, server: fakeredis.FakeServer) -> None:
        cfg = _cfg()
        traffic = [(v, None) for v, _ in _traffic(60, shift_at=10)]
        shared = _shared(server, cfg)
        local = DriftDetector(_snapshot(cfg), cfg)
        assert _outcomes(shared.add_query_embeddings(traffic)) == _outcomes(
            local.add_query_embeddings(traffic)
        )
        assert shared.baseline_mean_score is None


class TestWindowing:
    def test_baseline_window_yields_no_result(self, server: fakeredis.FakeServer) -> None:
        shared = _shared(server, _cfg())
        assert not shared.baseline_ready
        assert shared.add_query_embeddings(_traffic(10, shift_at=99)) == []
        assert shared.baseline_ready
        assert shared.buffer_size == 0

    def test_partial_window_stays_buffered(self, server: fakeredis.FakeServer) -> None:
        shared = _shared(server, _cfg())
        shared.add_query_embeddings(_traffic(7, shift_at=99))
        assert shared.buffer_size == 7
        assert _shared(server, _cfg()).buffer_size == 7  # visible to other replicas

    def test_empty_batch_only_drains(self, server: fakeredis.FakeServer) -> None:
        shared = _shared(server, _cfg())
        assert shared.add_query_embeddings([]) == []
        assert shared.buffer_size == 0

    def test_reset_clears_shared_state(self, server: fakeredis.FakeServer) -> None:
        shared = _shared(server, _cfg())
        shared.add_query_embeddings(_traffic(45, shift_at=10))
        other = _shared(server, _cfg())
        other.reset()
        assert shared.history == []
        assert not shared.baseline_ready
        assert shared.buffer_size == 0
        assert shared.consecutive_alerts == 0

    def test_history_is_capped(self, server: fakeredis.FakeServer) -> None:
        cfg = _cfg(window=2)
        shared = _shared(server, cfg)
        with patch("rag.drift.redis_state.HISTORY_LIMIT", 3):
            shared.add_query_embeddings(_traffic(20, shift_at=99))
        assert len(shared.history) == 3
        assert HISTORY_LIMIT == 1000

    def test_keys_expire(self, server: fakeredis.FakeServer) -> None:
        shared = _shared(server, _cfg(), state_ttl_s=60.0)
        shared.add_query_embeddings(_traffic(25, shift_at=99))
        client = fakeredis.FakeRedis(server=server)
        for suffix in ("window", "state", "history"):
            assert 0 < client.pttl(f"{NS}:{suffix}") <= 60_000


class TestLease:
    def test_busy_lease_leaves_windows_pending_for_the_next_caller(
        self, server: fakeredis.FakeServer
    ) -> None:
        cfg = _cfg()
        a = _shared(server, cfg, lock_wait_s=0.0)
        client = fakeredis.FakeRedis(server=server)
        client.set(f"{NS}:lock", "someone-else", px=60_000)
        assert a.add_query_embeddings(_traffic(20, shift_at=99)) == []
        assert client.llen(f"{NS}:pending") == 2
        client.delete(f"{NS}:lock")
        b = _shared(server, cfg)
        b.add_query_embeddings([])  # drains what a claimed
        assert client.llen(f"{NS}:pending") == 0
        assert b.baseline_ready and len(b.history) == 1

    def test_commit_is_rejected_after_losing_the_lease(self, server: fakeredis.FakeServer) -> None:
        cfg = _cfg()
        shared = _shared(server, cfg)
        client = fakeredis.FakeRedis(server=server)
        real_state = shared._state

        def steal_lease() -> object:
            client.set(f"{NS}:lock", "thief")  # lease expired and was re-acquired
            return real_state()

        with (
            patch.object(shared, "_state", side_effect=steal_lease),
            patch("rag.drift.redis_state.logger") as log,
        ):
            assert shared.add_query_embeddings(_traffic(10, shift_at=99)) == []
        log.warning.assert_called_once()
        assert not shared.baseline_ready  # the fenced write never landed
        assert client.get(f"{NS}:lock") == b"thief"  # and we did not release their lease


class TestResilientMonitor:
    def _monitor(
        self, server: fakeredis.FakeServer, cfg: DriftConfig
    ) -> tuple[ResilientDriftMonitor, RedisHealth]:
        health = RedisHealth(fakeredis.FakeRedis(server=server), retry_interval_s=0.0)
        shared = RedisDriftDetector(health.client, _snapshot(cfg), cfg, NS)
        return ResilientDriftMonitor(shared, DriftDetector(_snapshot(cfg), cfg), health), health

    def test_uses_shared_state_while_redis_is_up(self, server: fakeredis.FakeServer) -> None:
        cfg = _cfg()
        mon, _ = self._monitor(server, cfg)
        mon.add_query_embeddings(_traffic(10, shift_at=99))
        assert mon.baseline_ready
        assert mon.local.baseline_ready is False
        assert mon.shared.baseline_ready

    def test_falls_back_to_local_windows_and_never_raises(
        self, server: fakeredis.FakeServer
    ) -> None:
        cfg = _cfg()
        mon, health = self._monitor(server, cfg)
        server.connected = False
        with patch("rag.redis_health.logger") as log:
            outs = mon.add_query_embeddings(_traffic(60, shift_at=10))
            # reads also degrade to the local detector
            assert mon.history == mon.local.history
            assert mon.consecutive_alerts == mon.local.consecutive_alerts
            assert mon.reindex_triggered == mon.local.reindex_triggered
            assert mon.buffer_size == mon.local.buffer_size
            assert mon.baseline_ready is True
            assert mon.baseline_mean_score == mon.local.baseline_mean_score
            mon.reset()
        assert len(outs) == 5
        assert not health.up
        assert log.warning.call_count == 1

    def test_read_failure_switches_to_local(self, server: fakeredis.FakeServer) -> None:
        mon, health = self._monitor(server, _cfg())
        mon.local.add_query_embeddings(_traffic(3, shift_at=99))
        server.connected = False
        with patch("rag.redis_health.logger"):
            assert mon.buffer_size == 3
        assert not health.up

    def test_reset_failure_is_swallowed(self, server: fakeredis.FakeServer) -> None:
        mon, health = self._monitor(server, _cfg())
        with (
            patch.object(mon.shared, "reset", side_effect=ConnectionError("down")),
            patch("rag.redis_health.logger"),
        ):
            mon.reset()
        assert not health.up

    def test_recovery_discards_local_partial_window(self, server: fakeredis.FakeServer) -> None:
        cfg = _cfg()
        mon, health = self._monitor(server, cfg)
        server.connected = False
        with patch("rag.redis_health.logger"):
            mon.add_query_embeddings(_traffic(4, shift_at=99))
        assert mon.local.buffer_size == 4
        server.connected = True
        with patch("rag.redis_health.logger"), patch("rag.drift.redis_state.logger") as log:
            mon.add_query_embeddings(_traffic(2, shift_at=99))
        assert health.up
        assert mon.local.buffer_size == 0
        assert mon.shared.buffer_size == 2
        log.warning.assert_called_once()

    def test_recovery_with_empty_local_buffer_is_quiet(self, server: fakeredis.FakeServer) -> None:
        mon, _ = self._monitor(server, _cfg())
        server.connected = False
        with patch("rag.redis_health.logger"):
            assert mon.buffer_size == 0
        server.connected = True
        with patch("rag.redis_health.logger"), patch("rag.drift.redis_state.logger") as log:
            assert mon.buffer_size == 0
        log.warning.assert_not_called()
