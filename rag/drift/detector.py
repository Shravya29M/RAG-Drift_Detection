"""Drift detector: rolling query window, KS test via snapshot, hysteresis alarm."""

from __future__ import annotations

from collections import deque

import numpy as np

from rag.drift.snapshot import DistributionSnapshot
from rag.models import DriftConfig, DriftResult


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
      - If the counter reaches ``config.hysteresis_windows`` the hard-alarm
        flag is set and :attr:`reindex_triggered` becomes ``True``.
      - The buffer is cleared for the next window.
    * Returns are ``None`` when the window is not yet full, and a
      :class:`~rag.models.DriftResult` once a window is evaluated.

    Args:
        snapshot: Pre-fitted reference distribution snapshot.
        config: Drift detection configuration.

    Notes:
        Per CLAUDE.md this class is **not** a singleton. The APScheduler job
        holds the instance; do not store it as a module-level global.
    """

    def __init__(self, snapshot: DistributionSnapshot, config: DriftConfig) -> None:
        self._snapshot = snapshot
        self._config = config
        self._buffer: deque[np.ndarray] = deque()
        self._baseline: np.ndarray | None = None
        self._consecutive_alerts: int = 0
        self._history: list[DriftResult] = []
        self._reindex_triggered: bool = False

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
        return self._consecutive_alerts

    @property
    def reindex_triggered(self) -> bool:
        """``True`` after ``hysteresis_windows`` consecutive drifted windows."""
        return self._reindex_triggered

    @property
    def buffer_size(self) -> int:
        """Number of embeddings currently accumulated in the rolling buffer."""
        return len(self._buffer)

    @property
    def baseline_ready(self) -> bool:
        """``True`` once the calibration window has been captured."""
        return self._baseline is not None

    # ------------------------------------------------------------------
    # Public mutating interface
    # ------------------------------------------------------------------

    def add_query_embedding(self, embedding: np.ndarray) -> DriftResult | None:
        """Append one query embedding to the rolling buffer.

        When the buffer fills (``len == config.window_size``), a drift
        evaluation is triggered and the buffer is cleared.  The first full
        window is consumed as the calibration baseline and returns ``None``.

        Args:
            embedding: L2-normalised float32 vector of shape ``(dim,)``.

        Returns:
            A :class:`~rag.models.DriftResult` when this call completes a
            post-baseline window; ``None`` while the window is still
            accumulating or while calibrating.
        """
        self._buffer.append(np.asarray(embedding, dtype=np.float32))

        if len(self._buffer) < int(self._config.window_size):
            return None

        window: np.ndarray = np.stack(list(self._buffer), axis=0)
        self._buffer.clear()

        if self._baseline is None:
            self._baseline = window
            return None

        return self._evaluate_window(window)

    def reset(self) -> None:
        """Clear the buffer, history, and all alarm state.

        Call this after a successful re-index so the detector starts fresh
        against the new snapshot.  The calibration baseline is also cleared,
        so the next full window recalibrates.
        """
        self._buffer.clear()
        self._baseline = None
        self._consecutive_alerts = 0
        self._history.clear()
        self._reindex_triggered = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_window(self, window: np.ndarray) -> DriftResult:
        """Run the KS test for *window*, update hysteresis, return the result."""
        result = self._snapshot.compare(window, reference=self._baseline)
        self._history.append(result)

        if result.drifted:
            self._consecutive_alerts += 1
            if self._consecutive_alerts >= int(self._config.hysteresis_windows):
                self._reindex_triggered = True
        else:
            self._consecutive_alerts = 0

        return result
