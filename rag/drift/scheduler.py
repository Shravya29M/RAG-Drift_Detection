"""APScheduler background jobs for periodic drift checks and re-index runs."""

from __future__ import annotations

import queue

import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore[import-untyped]
from apscheduler.triggers.interval import IntervalTrigger  # type: ignore[import-untyped]

from rag.drift.alarm import AlarmLevel, DriftAlarm
from rag.drift.detector import DriftDetector
from rag.models import DriftConfig


class DriftScheduler:
    """Runs a background APScheduler job that processes query embeddings and
    fires drift alarms on a fixed interval.

    Design
    ------
    * Callers (API routes, tests) push embeddings into the scheduler via
      :meth:`enqueue_embedding`.  The method is non-blocking and thread-safe.
    * A single APScheduler ``IntervalTrigger`` job wakes every
      ``config.drift_check_interval_s`` seconds, drains the queue, and feeds
      each vector to the :class:`~rag.drift.detector.DriftDetector`.
    * When the detector returns a :class:`~rag.models.DriftResult` (window
      full), the alarm level is chosen:

      - ``AUTO``  when ``detector.reindex_triggered`` is ``True``
      - ``SOFT``  otherwise (single drifted window or clean window)

    * Per CLAUDE.md APScheduler jobs run in a thread pool — the detector and
      alarm must therefore be thread-safe for external read access, but this
      scheduler is the *only* writer, so no additional locking is needed here.

    Args:
        detector: Pre-configured :class:`~rag.drift.detector.DriftDetector`.
        alarm: :class:`~rag.drift.alarm.DriftAlarm` used to fire alerts.
        config: Drift configuration; ``drift_check_interval_s`` controls the
            scheduler tick rate.
        drift_check_interval_s: How often (in seconds) the background job runs
            and drains the queue.  Defaults to 30 s.
    """

    def __init__(
        self,
        detector: DriftDetector,
        alarm: DriftAlarm,
        config: DriftConfig,
        *,
        drift_check_interval_s: float = 30.0,
    ) -> None:
        self._detector = detector
        self._alarm = alarm
        self._config = config
        self._interval_s = drift_check_interval_s
        self._embedding_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._scheduler = BackgroundScheduler()
        self._job_id = "drift_check"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the APScheduler background thread and schedule the drift job.

        Calling :meth:`start` more than once is a no-op if the scheduler is
        already running.
        """
        if self._scheduler.running:
            return
        self._scheduler.add_job(
            self._tick,
            trigger=IntervalTrigger(seconds=self._interval_s),
            id=self._job_id,
            replace_existing=True,
            max_instances=1,  # never overlap if a tick takes longer than interval
        )
        self._scheduler.start()

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop the background scheduler.

        Args:
            wait: If ``True`` (default), block until the running job finishes.
                  Pass ``False`` in tests to avoid hanging on a slow job.
        """
        if self._scheduler.running:
            self._scheduler.shutdown(wait=wait)

    # ------------------------------------------------------------------
    # Public interface used by API / ingestion layer
    # ------------------------------------------------------------------

    def enqueue_embedding(self, embedding: np.ndarray) -> None:
        """Push one query embedding into the processing queue.

        Non-blocking and thread-safe.  The embedding will be consumed by the
        next scheduler tick.

        Args:
            embedding: L2-normalised float32 vector of shape ``(dim,)``.
        """
        self._embedding_queue.put(np.asarray(embedding, dtype=np.float32))

    @property
    def queue_size(self) -> int:
        """Approximate number of embeddings waiting in the queue."""
        return self._embedding_queue.qsize()

    @property
    def is_running(self) -> bool:
        """``True`` while the background scheduler is active."""
        return bool(self._scheduler.running)

    # ------------------------------------------------------------------
    # Internal job
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Drain the embedding queue and feed vectors to the detector.

        Called by APScheduler on every interval tick.  Any :class:`~rag.models.DriftResult`
        returned by the detector triggers alarm selection:

        - ``AUTO`` when hysteresis threshold reached (``detector.reindex_triggered``)
        - ``SOFT`` for the first drifted window or a clean window
        """
        while True:
            try:
                embedding = self._embedding_queue.get_nowait()
            except queue.Empty:
                break

            result = self._detector.add_query_embedding(embedding)
            if result is None:
                continue

            level = AlarmLevel.AUTO if self._detector.reindex_triggered else AlarmLevel.SOFT
            self._alarm.fire(result, level)
