"""Alarm manager: soft/hard alerts and auto re-index trigger with hysteresis."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

import httpx

from rag.models import AlarmConfig, DriftResult
from rag.tracking import log_event


class AlarmLevel(StrEnum):
    """Severity of a drift alarm."""

    SOFT = "soft"
    """Log to W&B; no external side-effects."""
    HARD = "hard"
    """POST to webhook URL in addition to soft actions."""
    AUTO = "auto"
    """Call the re-index callback in addition to hard actions."""


class DriftAlarm:
    """Evaluates a :class:`~rag.models.DriftResult` and fires the appropriate
    alarm level.

    Alarm levels
    ------------
    * **Soft** — calls :func:`~rag.tracking.log_event` to record the drift
      score in W&B.  Always executed for any drifted result.
    * **Hard** — POSTs a JSON payload to ``config.webhook_url`` (Slack, PagerDuty,
      email relay, …).  Skipped when ``webhook_url`` is empty.
    * **Auto** — invokes *re_index_callback* to kick off a re-index run.

    The level is determined by the caller (typically
    :class:`~rag.drift.detector.DriftDetector`) based on hysteresis state:
    pass ``AlarmLevel.SOFT`` for a single drifted window, ``AlarmLevel.AUTO``
    once hysteresis fires.

    Args:
        config: Alarm configuration (webhook URL, timeout).
        re_index_callback: Zero-argument callable invoked on ``AUTO`` level.
            May be ``None`` if auto re-index is not configured.
    """

    def __init__(
        self,
        config: AlarmConfig,
        re_index_callback: Callable[[], None] | None = None,
    ) -> None:
        self._config = config
        self._re_index_callback = re_index_callback

    def fire(self, result: DriftResult, level: AlarmLevel) -> None:
        """Execute all actions appropriate for *level*.

        Actions are additive: AUTO runs HARD steps then the callback; HARD runs
        SOFT steps then the webhook; SOFT logs to W&B.

        Args:
            result: The drift evaluation result that triggered this alarm.
            level: Minimum alarm level to execute.
        """
        # Soft is always included.
        self._soft(result)

        if level in (AlarmLevel.HARD, AlarmLevel.AUTO):
            self._hard(result)

        if level is AlarmLevel.AUTO:
            self._auto(result)

    # ------------------------------------------------------------------
    # Level implementations
    # ------------------------------------------------------------------

    def _soft(self, result: DriftResult) -> None:
        """Log drift metrics to W&B."""
        log_event(
            "drift_window",
            {
                "statistic": result.statistic,
                "pvalue": result.pvalue,
                "drifted": int(result.drifted),
                "window_size": result.window_size,
                "snapshot_size": result.snapshot_size,
            },
        )

    def _hard(self, result: DriftResult) -> None:
        """POST a JSON alert payload to the configured webhook URL.

        Silently skips when ``config.webhook_url`` is empty.  Network errors
        are caught and suppressed so a failing webhook never blocks the
        detector pipeline.
        """
        if not self._config.webhook_url:
            return
        payload = {
            "alert": "drift_detected",
            "statistic": result.statistic,
            "pvalue": result.pvalue,
            "window_size": result.window_size,
            "snapshot_size": result.snapshot_size,
        }
        try:
            with httpx.Client(timeout=self._config.webhook_timeout_s) as client:
                response = client.post(self._config.webhook_url, json=payload)
                response.raise_for_status()
        except httpx.HTTPError:
            # Webhook failure must not crash the detector.
            pass

    def _auto(self, result: DriftResult) -> None:
        """Invoke the re-index callback if one is registered."""
        if self._re_index_callback is not None:
            self._re_index_callback()
