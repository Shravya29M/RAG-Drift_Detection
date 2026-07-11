"""Unit tests for rag.drift.detector.DriftDetector."""

from __future__ import annotations

from datetime import datetime
from typing import cast
from unittest.mock import MagicMock

import numpy as np

from rag.drift.detector import DriftDetector
from rag.drift.snapshot import DistributionSnapshot
from rag.models import DriftConfig, DriftResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8
WINDOW = 3


def _cfg(**kwargs: object) -> DriftConfig:
    defaults: dict[str, object] = {
        "window_size": WINDOW,
        "pca_components": 4,
        "threshold_alpha": 0.2,
        "hysteresis_windows": 3,
    }
    defaults.update(kwargs)
    return DriftConfig(**defaults)  # type: ignore[arg-type]


def _vec(val: float = 0.0) -> np.ndarray:
    """Return a unit-norm float32 vector with *val* filled in each slot."""
    v = np.full(DIM, val, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def _drift_result(*, drifted: bool, statistic: float = 0.5) -> DriftResult:
    return DriftResult(
        statistic=statistic,
        pvalue=0.01 if drifted else 0.9,
        drifted=drifted,
        window_size=WINDOW,
        snapshot_size=50,
        evaluated_at=datetime(2026, 1, 1),
    )


def _mock_snapshot(return_values: list[DriftResult]) -> DistributionSnapshot:
    """Return a snapshot mock whose compare() cycles through *return_values*."""
    snap = MagicMock(spec=DistributionSnapshot)
    snap.compare.side_effect = return_values
    return snap


def _detector(return_values: list[DriftResult], **cfg_kwargs: object) -> DriftDetector:
    return DriftDetector(_mock_snapshot(return_values), _cfg(**cfg_kwargs))


def _calibrate(det: DriftDetector) -> None:
    """Feed one full window so the detector captures its baseline."""
    for _ in range(WINDOW):
        assert det.add_query_embedding(_vec()) is None


def _feed_windows(det: DriftDetector, n: int) -> list[DriftResult | None]:
    """Feed *n* full post-calibration windows, returning the flush results."""
    results: list[DriftResult | None] = []
    for _ in range(n):
        for _ in range(WINDOW - 1):
            det.add_query_embedding(_vec())
        results.append(det.add_query_embedding(_vec()))
    return results


# ---------------------------------------------------------------------------
# Baseline calibration
# ---------------------------------------------------------------------------


class TestBaselineCalibration:
    def test_baseline_not_ready_initially(self) -> None:
        det = _detector([])
        assert not det.baseline_ready

    def test_first_full_window_becomes_baseline(self) -> None:
        det = _detector([])
        _calibrate(det)
        assert det.baseline_ready

    def test_calibration_window_returns_none(self) -> None:
        det = _detector([])
        results = [det.add_query_embedding(_vec()) for _ in range(WINDOW)]
        assert results == [None, None, None]

    def test_calibration_window_produces_no_history(self) -> None:
        det = _detector([])
        _calibrate(det)
        assert det.history == []

    def test_compare_not_called_for_calibration_window(self) -> None:
        snap = _mock_snapshot([])
        det = DriftDetector(snap, _cfg())
        _calibrate(det)
        cast(MagicMock, snap.compare).assert_not_called()

    def test_baseline_passed_as_reference_to_compare(self) -> None:
        snap = _mock_snapshot([_drift_result(drifted=False)])
        det = DriftDetector(snap, _cfg())
        _calibrate(det)
        _feed_windows(det, 1)
        kwargs = cast(MagicMock, snap.compare).call_args.kwargs
        reference: np.ndarray = kwargs["reference"]
        assert reference.shape == (WINDOW, DIM)

    def test_reset_clears_baseline(self) -> None:
        det = _detector([])
        _calibrate(det)
        det.reset()
        assert not det.baseline_ready


# ---------------------------------------------------------------------------
# Window accumulation
# ---------------------------------------------------------------------------


class TestWindowAccumulation:
    def test_returns_none_when_window_not_full(self) -> None:
        det = _detector([_drift_result(drifted=False)])
        _calibrate(det)
        assert det.add_query_embedding(_vec(1.0)) is None
        assert det.add_query_embedding(_vec(2.0)) is None

    def test_returns_drift_result_when_window_completes(self) -> None:
        det = _detector([_drift_result(drifted=False)])
        _calibrate(det)
        (result,) = _feed_windows(det, 1)
        assert isinstance(result, DriftResult)

    def test_buffer_size_tracks_accumulation(self) -> None:
        det = _detector([_drift_result(drifted=False)])
        assert det.buffer_size == 0
        det.add_query_embedding(_vec())
        assert det.buffer_size == 1
        det.add_query_embedding(_vec())
        assert det.buffer_size == 2

    def test_buffer_cleared_after_window_flush(self) -> None:
        det = _detector([_drift_result(drifted=False)] * 2)
        _calibrate(det)
        _feed_windows(det, 1)
        assert det.buffer_size == 0

    def test_second_window_starts_fresh(self) -> None:
        det = _detector([_drift_result(drifted=False)] * 2)
        _calibrate(det)
        _feed_windows(det, 1)
        assert det.add_query_embedding(_vec()) is None
        assert det.add_query_embedding(_vec()) is None

    def test_snapshot_compare_not_called_before_window_full(self) -> None:
        snap = _mock_snapshot([_drift_result(drifted=False)])
        det = DriftDetector(snap, _cfg())
        _calibrate(det)
        det.add_query_embedding(_vec())
        det.add_query_embedding(_vec())
        cast(MagicMock, snap.compare).assert_not_called()

    def test_snapshot_compare_called_on_full_window(self) -> None:
        snap = _mock_snapshot([_drift_result(drifted=False)])
        det = DriftDetector(snap, _cfg())
        _calibrate(det)
        _feed_windows(det, 1)
        cast(MagicMock, snap.compare).assert_called_once()

    def test_window_passed_to_compare_has_correct_shape(self) -> None:
        snap = _mock_snapshot([_drift_result(drifted=False)])
        det = DriftDetector(snap, _cfg())
        _calibrate(det)
        _feed_windows(det, 1)
        passed_array: np.ndarray = cast(MagicMock, snap.compare).call_args[0][0]
        assert passed_array.shape == (WINDOW, DIM)


# ---------------------------------------------------------------------------
# Drift below threshold (no alarm)
# ---------------------------------------------------------------------------


class TestNoDrift:
    def test_no_drift_history_empty_before_window(self) -> None:
        det = _detector([_drift_result(drifted=False)])
        det.add_query_embedding(_vec())
        assert det.history == []

    def test_no_drift_result_appended_to_history(self) -> None:
        det = _detector([_drift_result(drifted=False)])
        _calibrate(det)
        _feed_windows(det, 1)
        assert len(det.history) == 1
        assert not det.history[0].drifted

    def test_no_drift_consecutive_alerts_stays_zero(self) -> None:
        det = _detector([_drift_result(drifted=False)] * 3)
        _calibrate(det)
        _feed_windows(det, 3)
        assert det.consecutive_alerts == 0

    def test_no_drift_reindex_not_triggered(self) -> None:
        det = _detector([_drift_result(drifted=False)] * 3)
        _calibrate(det)
        _feed_windows(det, 3)
        assert not det.reindex_triggered

    def test_alert_resets_on_clean_window(self) -> None:
        """Two drifted windows then one clean window must reset counter to 0."""
        results = [
            _drift_result(drifted=True),
            _drift_result(drifted=True),
            _drift_result(drifted=False),
        ]
        det = _detector(results, hysteresis_windows=5)  # high threshold so reindex doesn't fire
        _calibrate(det)
        _feed_windows(det, 3)
        assert det.consecutive_alerts == 0


# ---------------------------------------------------------------------------
# Hysteresis — reindex triggers on 3rd consecutive drifted window
# ---------------------------------------------------------------------------


class TestHysteresis:
    def test_reindex_not_triggered_after_one_drifted_window(self) -> None:
        det = _detector([_drift_result(drifted=True)])
        _calibrate(det)
        _feed_windows(det, 1)
        assert not det.reindex_triggered

    def test_reindex_not_triggered_after_two_drifted_windows(self) -> None:
        det = _detector([_drift_result(drifted=True)] * 2)
        _calibrate(det)
        _feed_windows(det, 2)
        assert not det.reindex_triggered

    def test_reindex_triggered_after_three_drifted_windows(self) -> None:
        det = _detector([_drift_result(drifted=True)] * 3)
        _calibrate(det)
        _feed_windows(det, 3)
        assert det.reindex_triggered

    def test_consecutive_alerts_increments_per_drifted_window(self) -> None:
        det = _detector([_drift_result(drifted=True)] * 3, hysteresis_windows=10)
        _calibrate(det)
        for i in range(3):
            _feed_windows(det, 1)
            assert det.consecutive_alerts == i + 1

    def test_history_has_all_window_results(self) -> None:
        det = _detector([_drift_result(drifted=True)] * 3)
        _calibrate(det)
        _feed_windows(det, 3)
        assert len(det.history) == 3

    def test_reindex_stays_true_after_trigger(self) -> None:
        """Flag must persist across further windows once set."""
        results = [_drift_result(drifted=True)] * 5
        det = _detector(results)
        _calibrate(det)
        _feed_windows(det, 5)
        assert det.reindex_triggered

    def test_hysteresis_window_count_is_configurable(self) -> None:
        """With hysteresis_windows=1 a single drifted window should trigger."""
        det = _detector([_drift_result(drifted=True)], hysteresis_windows=1)
        _calibrate(det)
        _feed_windows(det, 1)
        assert det.reindex_triggered

    def test_interleaved_does_not_trigger(self) -> None:
        """drift/clean/drift must NOT trigger because streak is broken."""
        det = _detector(
            [_drift_result(drifted=True), _drift_result(drifted=False), _drift_result(drifted=True)]
        )
        _calibrate(det)
        _feed_windows(det, 3)
        assert not det.reindex_triggered


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_buffer(self) -> None:
        det = _detector([_drift_result(drifted=False)])
        det.add_query_embedding(_vec())
        det.reset()
        assert det.buffer_size == 0

    def test_reset_clears_history(self) -> None:
        det = _detector([_drift_result(drifted=False)])
        _calibrate(det)
        _feed_windows(det, 1)
        det.reset()
        assert det.history == []

    def test_reset_clears_consecutive_alerts(self) -> None:
        det = _detector([_drift_result(drifted=True)] * 2, hysteresis_windows=10)
        _calibrate(det)
        _feed_windows(det, 2)
        det.reset()
        assert det.consecutive_alerts == 0

    def test_reset_clears_reindex_flag(self) -> None:
        det = _detector([_drift_result(drifted=True)] * 3)
        _calibrate(det)
        _feed_windows(det, 3)
        assert det.reindex_triggered
        det.reset()
        assert not det.reindex_triggered
