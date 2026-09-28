"""Drift detector: rolling query window, KS test via snapshot, hysteresis alarm."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from rag.drift.snapshot import DistributionSnapshot
from rag.models import DriftConfig, DriftResult


@dataclass(frozen=True)
class HysteresisState:
    """Everything a drift detector carries between windows.

    Kept separate from the rolling buffer so the same transition function
    drives both the in-process :class:`DriftDetector` and the Redis-backed
    shared detector used when several replicas run together.
    """

    baseline: np.ndarray | None = None
    baseline_mean_score: float | None = None
    consecutive_alerts: int = 0
    reindex_triggered: bool = False


@dataclass(frozen=True)
class WindowOutcome:
    """One evaluated window plus the ``reindex_triggered`` flag *after* it."""

    result: DriftResult
    reindex_triggered: bool


def summarize_window(
    items: Sequence[tuple[np.ndarray, float | None]],
) -> tuple[np.ndarray, float | None]:
    """Stack a full window's embeddings and average its known retrieval scores."""
    window: np.ndarray = np.stack([e for e, _ in items], axis=0)
    scores = [s for _, s in items if s is not None]
    mean_score = float(np.mean(scores)) if scores else None
    return window, mean_score


def close_window(
    snapshot: DistributionSnapshot,
    config: DriftConfig,
    state: HysteresisState,
    window: np.ndarray,
    mean_score: float | None,
) -> tuple[HysteresisState, DriftResult | None]:
    """Apply one completed window to *state*.

    The first window becomes the calibration baseline (no result). Every later
    window is KS-tested against the baseline, then the hysteresis counter and
    quality gate decide between no action, benign recalibration, and
    ``reindex_triggered`` — see :class:`DriftDetector` for the full semantics.

    Returns:
        The new state and the window's :class:`~rag.models.DriftResult`
        (``None`` for the calibration window).
    """
    if state.baseline is None:
        return replace(state, baseline=window, baseline_mean_score=mean_score), None

    result = snapshot.compare(window, reference=state.baseline)

    quality_known = mean_score is not None and state.baseline_mean_score is not None
    degraded = False
    if mean_score is not None and state.baseline_mean_score is not None:
        degraded = mean_score < config.quality_drop_ratio * state.baseline_mean_score
    recalibrated = False

    if result.drifted:
        state = replace(state, consecutive_alerts=state.consecutive_alerts + 1)
        if state.consecutive_alerts >= int(config.hysteresis_windows):
            if quality_known and not degraded:
                # Sustained drift but retrieval still healthy: users moved
                # to a topic the corpus answers fine. Adopt the new query
                # distribution as baseline; the index is not stale.
                state = replace(
                    state,
                    baseline=window,
                    baseline_mean_score=mean_score,
                    consecutive_alerts=0,
                )
                recalibrated = True
            else:
                # Degraded scores (or no quality signal): genuine staleness.
                state = replace(state, reindex_triggered=True)
    else:
        state = replace(state, consecutive_alerts=0)

    result = result.model_copy(
        update={
            "mean_top_score": mean_score,
            "quality_degraded": degraded,
            "recalibrated": recalibrated,
        }
    )
    return state, result


class DriftMonitor(Protocol):
    """What the scheduler and API need from a drift detector.

    Implemented by the in-process :class:`DriftDetector` and by the shared,
    Redis-backed monitor in :mod:`rag.drift.redis_state`.
    """

    @property
    def history(self) -> list[DriftResult]: ...

    @property
    def consecutive_alerts(self) -> int: ...

    @property
    def reindex_triggered(self) -> bool: ...

    @property
    def buffer_size(self) -> int: ...

    @property
    def baseline_ready(self) -> bool: ...

    @property
    def baseline_mean_score(self) -> float | None: ...

    def add_query_embeddings(
        self, batch: Sequence[tuple[np.ndarray, float | None]]
    ) -> list[WindowOutcome]: ...

    def reset(self) -> None: ...


class DriftDetector:
    """Maintains a rolling window of query embeddings and evaluates drift on
    every completed window.

    State machine
    -------------
    * Each call to :meth:`add_query_embedding` appends one vector to the
      current window buffer.
    * The **first** completed window becomes the *baseline*: a calibration
      sample of real query traffic against which later windows are compared.
      Queries and document chunks occupy different regions of embedding space
      even in a healthy system, so comparing queries to queries is what makes
      the KS test meaningful.  No drift result is emitted for the baseline
      window.
    * When the buffer reaches ``config.window_size`` vectors it is flushed:
      - :meth:`~rag.drift.snapshot.DistributionSnapshot.compare` is called to
        produce a :class:`~rag.models.DriftResult`.
      - The result is appended to :attr:`history`.
      - The *consecutive alert counter* is incremented when ``result.drifted``
        is ``True``, or reset to 0 otherwise (hysteresis).
      - When the counter reaches ``config.hysteresis_windows`` the response is
        **quality-gated**: drift alone does not prove the index is stale — it
        may just mean users started asking about a different topic the corpus
        still answers well.  If the window's mean retrieval score is healthy
        (≥ ``quality_drop_ratio`` × the baseline mean), the detector adopts
        the window as its new baseline (``result.recalibrated``) instead of
        triggering a re-index.  Only drift *combined with* degraded retrieval
        scores — or drift with no score data at all (fallback) — sets
        :attr:`reindex_triggered`.
      - The buffer is cleared for the next window.
    * Returns are ``None`` when the window is not yet full, and a
      :class:`~rag.models.DriftResult` once a window is evaluated.

    Args:
        snapshot: Pre-fitted reference distribution snapshot.
        config: Drift detection configuration.

    Notes:
        This class is **not** a singleton. The APScheduler job
        holds the instance; do not store it as a module-level global.
    """

    def __init__(self, snapshot: DistributionSnapshot, config: DriftConfig) -> None:
        self._snapshot = snapshot
        self._config = config
        self._buffer: deque[tuple[np.ndarray, float | None]] = deque()
        self._state = HysteresisState()
        self._history: list[DriftResult] = []

    # ------------------------------------------------------------------
    # Read-only properties (tests and scheduler inspect these)
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[DriftResult]:
        """All evaluated :class:`~rag.models.DriftResult` objects, oldest first."""
        return list(self._history)

    @property
    def consecutive_alerts(self) -> int:
        """Current count of consecutive drifted windows (resets on clean window)."""
        return self._state.consecutive_alerts

    @property
    def reindex_triggered(self) -> bool:
        """``True`` after ``hysteresis_windows`` consecutive drifted windows."""
        return self._state.reindex_triggered

    @property
    def buffer_size(self) -> int:
        """Number of embeddings currently accumulated in the rolling buffer."""
        return len(self._buffer)

    @property
    def baseline_ready(self) -> bool:
        """``True`` once the calibration window has been captured."""
        return self._state.baseline is not None

    @property
    def baseline_mean_score(self) -> float | None:
        """Mean retrieval score of the calibration baseline; ``None`` if untracked."""
        return self._state.baseline_mean_score

    # ------------------------------------------------------------------
    # Public mutating interface
    # ------------------------------------------------------------------

    def add_query_embedding(
        self,
        embedding: np.ndarray,
        top_score: float | None = None,
    ) -> DriftResult | None:
        """Append one query embedding (and its retrieval score) to the buffer.

        When the buffer fills (``len == config.window_size``), a drift
        evaluation is triggered and the buffer is cleared.  The first full
        window is consumed as the calibration baseline and returns ``None``.

        Args:
            embedding: L2-normalised float32 vector of shape ``(dim,)``.
            top_score: Mean retrieval score for this query (e.g. mean of the
                top-k cosine scores; 0.0 for a no-hit query).  ``None`` when
                retrieval quality is not tracked — the quality gate then falls
                back to drift-only behaviour.

        Returns:
            A :class:`~rag.models.DriftResult` when this call completes a
            post-baseline window; ``None`` while the window is still
            accumulating or while calibrating.
        """
        self._buffer.append((np.asarray(embedding, dtype=np.float32), top_score))

        if len(self._buffer) < int(self._config.window_size):
            return None

        window, mean_score = summarize_window(self._buffer)
        self._buffer.clear()

        self._state, result = close_window(
            self._snapshot, self._config, self._state, window, mean_score
        )
        if result is not None:
            self._history.append(result)
        return result

    def add_query_embeddings(
        self, batch: Sequence[tuple[np.ndarray, float | None]]
    ) -> list[WindowOutcome]:
        """Feed a batch through :meth:`add_query_embedding`.

        Returns:
            One :class:`WindowOutcome` per window evaluated, carrying the
            ``reindex_triggered`` flag as it stood right after that window.
        """
        outcomes: list[WindowOutcome] = []
        for embedding, top_score in batch:
            result = self.add_query_embedding(embedding, top_score)
            if result is not None:
                outcomes.append(WindowOutcome(result, self._state.reindex_triggered))
        return outcomes

    def discard_buffer(self) -> None:
        """Drop the partially filled window, keeping baseline and hysteresis state."""
        self._buffer.clear()

    def reset(self) -> None:
        """Clear the buffer, history, and all alarm state.

        Call this after a successful re-index so the detector starts fresh
        against the new snapshot.  The calibration baseline is also cleared,
        so the next full window recalibrates.
        """
        self._buffer.clear()
        self._state = HysteresisState()
        self._history.clear()
