"""Integration test: DriftScheduler runs background job and calls detector."""

from __future__ import annotations

import time
from datetime import datetime
from typing import cast
from unittest.mock import MagicMock

import numpy as np

from rag.drift.alarm import DriftAlarm
from rag.drift.detector import DriftDetector
from rag.drift.scheduler import DriftScheduler
from rag.models import DriftConfig, DriftResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8
TICK_S = 0.05  # fast tick so tests don't need to wait long


def _cfg() -> DriftConfig:
    return DriftConfig(window_size=2, pca_components=4, threshold_alpha=0.2, hysteresis_windows=3)


def _drift_result(*, drifted: bool = False) -> DriftResult:
    return DriftResult(
        statistic=0.1,
        pvalue=0.9,
        drifted=drifted,
        window_size=2,
        snapshot_size=50,
        evaluated_at=datetime(2026, 1, 1),
    )


def _vec() -> np.ndarray:
    v = np.ones(DIM, dtype=np.float32)
    return v / float(np.linalg.norm(v))


def _mock_detector(return_val: DriftResult | None = None) -> DriftDetector:
    """Return a DriftDetector mock with add_query_embedding pre-wired.

    We use a plain MagicMock (no spec) so that mypy doesn't enforce the real
    property types; the scheduler only calls add_query_embedding and reads
    reindex_triggered, both of which we set explicitly.
    """
    det = MagicMock()
    det.add_query_embedding.return_value = return_val
    det.reindex_triggered = False
    return cast(DriftDetector, det)


def _mock_alarm() -> DriftAlarm:
    return cast(DriftAlarm, MagicMock())


# ---------------------------------------------------------------------------
# Lifecycle tests (no real sleep needed)
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_not_running_before_start(self) -> None:
        sched = DriftScheduler(_mock_detector(), _mock_alarm(), _cfg(), drift_check_interval_s=1.0)
        assert not sched.is_running

    def test_running_after_start(self) -> None:
        sched = DriftScheduler(_mock_detector(), _mock_alarm(), _cfg(), drift_check_interval_s=1.0)
        sched.start()
        try:
            assert sched.is_running
        finally:
            sched.shutdown(wait=False)

    def test_not_running_after_shutdown(self) -> None:
        sched = DriftScheduler(_mock_detector(), _mock_alarm(), _cfg(), drift_check_interval_s=1.0)
        sched.start()
        sched.shutdown(wait=True)
        assert not sched.is_running

    def test_double_start_is_noop(self) -> None:
        sched = DriftScheduler(_mock_detector(), _mock_alarm(), _cfg(), drift_check_interval_s=1.0)
        sched.start()
        try:
            sched.start()  # must not raise or add duplicate job
            assert sched.is_running
        finally:
            sched.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Queue interface
# ---------------------------------------------------------------------------


class TestQueue:
    def test_enqueue_increases_queue_size(self) -> None:
        sched = DriftScheduler(_mock_detector(), _mock_alarm(), _cfg(), drift_check_interval_s=60.0)
        sched.enqueue_embedding(_vec())
        sched.enqueue_embedding(_vec())
        assert sched.queue_size == 2

    def test_queue_drained_after_tick(self) -> None:
        detector = _mock_detector(return_val=None)
        sched = DriftScheduler(detector, _mock_alarm(), _cfg(), drift_check_interval_s=60.0)
        sched.enqueue_embedding(_vec())
        sched.enqueue_embedding(_vec())
        sched._tick()  # invoke directly
        assert sched.queue_size == 0


# ---------------------------------------------------------------------------
# _tick logic (unit-level, no real scheduler thread)
# ---------------------------------------------------------------------------


class TestTickLogic:
    def test_tick_calls_detector_for_each_queued_embedding(self) -> None:
        detector = _mock_detector(return_val=None)
        sched = DriftScheduler(detector, _mock_alarm(), _cfg(), drift_check_interval_s=60.0)
        for _ in range(3):
            sched.enqueue_embedding(_vec())
        sched._tick()
        assert cast(MagicMock, detector).add_query_embedding.call_count == 3

    def test_tick_no_alarm_when_detector_returns_none(self) -> None:
        detector = _mock_detector(return_val=None)
        alarm = _mock_alarm()
        sched = DriftScheduler(detector, alarm, _cfg(), drift_check_interval_s=60.0)
        sched.enqueue_embedding(_vec())
        sched._tick()
        cast(MagicMock, alarm).fire.assert_not_called()

    def test_tick_fires_soft_alarm_on_drift_result(self) -> None:
        from rag.drift.alarm import AlarmLevel

        result = _drift_result(drifted=True)
        detector = _mock_detector(return_val=result)
        cast(MagicMock, detector).reindex_triggered = False
        alarm = _mock_alarm()
        sched = DriftScheduler(detector, alarm, _cfg(), drift_check_interval_s=60.0)
        sched.enqueue_embedding(_vec())
        sched._tick()
        cast(MagicMock, alarm).fire.assert_called_once_with(result, AlarmLevel.SOFT)

    def test_tick_fires_auto_alarm_when_reindex_triggered(self) -> None:
        from rag.drift.alarm import AlarmLevel

        result = _drift_result(drifted=True)
        detector = _mock_detector(return_val=result)
        cast(MagicMock, detector).reindex_triggered = True
        alarm = _mock_alarm()
        sched = DriftScheduler(detector, alarm, _cfg(), drift_check_interval_s=60.0)
        sched.enqueue_embedding(_vec())
        sched._tick()
        cast(MagicMock, alarm).fire.assert_called_once_with(result, AlarmLevel.AUTO)

    def test_tick_empty_queue_is_noop(self) -> None:
        detector = _mock_detector(return_val=None)
        sched = DriftScheduler(detector, _mock_alarm(), _cfg(), drift_check_interval_s=60.0)
        sched._tick()  # no embeddings queued
        cast(MagicMock, detector).add_query_embedding.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: scheduler runs for 2 seconds, job executes multiple times
# ---------------------------------------------------------------------------


class TestSchedulerRunsForTwoSeconds:
    def test_detector_called_by_background_thread(self) -> None:
        """Run the scheduler for ~2 s with a fast tick; confirm the detector
        receives all enqueued embeddings via the background thread."""
        detector = _mock_detector(return_val=None)
        alarm = _mock_alarm()
        sched = DriftScheduler(detector, alarm, _cfg(), drift_check_interval_s=TICK_S)

        n_embeddings = 10
        sched.start()
        try:
            for _ in range(n_embeddings):
                sched.enqueue_embedding(_vec())
            # Wait long enough for multiple ticks to drain the queue
            deadline = time.monotonic() + 2.0
            while sched.queue_size > 0 and time.monotonic() < deadline:
                time.sleep(0.05)
        finally:
            sched.shutdown(wait=True)

        # All embeddings must have been processed
        assert sched.queue_size == 0
        assert cast(MagicMock, detector).add_query_embedding.call_count == n_embeddings
