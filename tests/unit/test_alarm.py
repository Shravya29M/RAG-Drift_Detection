"""Unit tests for rag.drift.alarm and rag.tracking."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest

from rag.drift.alarm import AlarmLevel, DriftAlarm
from rag.models import AlarmConfig, DriftResult
from rag.tracking import log_event

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(webhook_url: str = "") -> AlarmConfig:
    return AlarmConfig(webhook_url=webhook_url)


def _result(*, drifted: bool = True, statistic: float = 0.6) -> DriftResult:
    return DriftResult(
        statistic=statistic,
        pvalue=0.01 if drifted else 0.9,
        drifted=drifted,
        window_size=50,
        snapshot_size=200,
        evaluated_at=datetime(2026, 1, 1),
    )


def _ok_response() -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# log_event / rag.tracking
# ---------------------------------------------------------------------------


class TestLogEvent:
    def test_calls_wandb_log_when_run_active(self) -> None:
        with (
            patch("rag.tracking.wandb.run", new=MagicMock()),
            patch("rag.tracking.wandb.log") as mock_log,
            patch.dict("os.environ", {}, clear=False),
        ):
            log_event("my_event", {"score": 0.5})
            mock_log.assert_called_once_with({"my_event/score": 0.5})

    def test_prefixes_keys_with_event_name(self) -> None:
        with (
            patch("rag.tracking.wandb.run", new=MagicMock()),
            patch("rag.tracking.wandb.log") as mock_log,
        ):
            log_event("drift_window", {"statistic": 0.3, "pvalue": 0.1})
            logged = mock_log.call_args[0][0]
            assert "drift_window/statistic" in logged
            assert "drift_window/pvalue" in logged

    def test_noop_when_wandb_disabled_env(self) -> None:
        with (
            patch("rag.tracking.wandb.run", new=MagicMock()),
            patch("rag.tracking.wandb.log") as mock_log,
            patch.dict("os.environ", {"WANDB_DISABLED": "true"}),
        ):
            log_event("ev", {"x": 1})
            mock_log.assert_not_called()

    def test_noop_when_wandb_disabled_case_insensitive(self) -> None:
        with (
            patch("rag.tracking.wandb.run", new=MagicMock()),
            patch("rag.tracking.wandb.log") as mock_log,
            patch.dict("os.environ", {"WANDB_DISABLED": "TRUE"}),
        ):
            log_event("ev", {"x": 1})
            mock_log.assert_not_called()

    def test_noop_when_no_active_run(self) -> None:
        with (
            patch("rag.tracking.wandb.run", new=None),
            patch("rag.tracking.wandb.log") as mock_log,
        ):
            log_event("ev", {"x": 1})
            mock_log.assert_not_called()

    def test_empty_data_still_calls_log(self) -> None:
        with (
            patch("rag.tracking.wandb.run", new=MagicMock()),
            patch("rag.tracking.wandb.log") as mock_log,
        ):
            log_event("ev", {})
            mock_log.assert_called_once_with({})


# ---------------------------------------------------------------------------
# DriftAlarm — soft level
# ---------------------------------------------------------------------------


class TestSoftAlarm:
    def test_soft_calls_log_event(self) -> None:
        alarm = DriftAlarm(_cfg())
        with patch("rag.drift.alarm.log_event") as mock_log:
            alarm.fire(_result(), AlarmLevel.SOFT)
            mock_log.assert_called_once()

    def test_soft_logs_correct_event_name(self) -> None:
        alarm = DriftAlarm(_cfg())
        with patch("rag.drift.alarm.log_event") as mock_log:
            alarm.fire(_result(), AlarmLevel.SOFT)
            assert mock_log.call_args[0][0] == "drift_window"

    def test_soft_payload_contains_statistic(self) -> None:
        alarm = DriftAlarm(_cfg())
        with patch("rag.drift.alarm.log_event") as mock_log:
            alarm.fire(_result(statistic=0.75), AlarmLevel.SOFT)
            data: dict[str, object] = mock_log.call_args[0][1]
            assert data["statistic"] == pytest.approx(0.75)

    def test_soft_does_not_post_webhook(self) -> None:
        alarm = DriftAlarm(_cfg(webhook_url="https://example.com/hook"))
        with (
            patch("rag.drift.alarm.log_event"),
            patch("rag.drift.alarm.httpx.Client") as mock_client_cls,
        ):
            alarm.fire(_result(), AlarmLevel.SOFT)
            mock_client_cls.assert_not_called()

    def test_soft_does_not_call_reindex_callback(self) -> None:
        cb = MagicMock()
        alarm = DriftAlarm(_cfg(), re_index_callback=cb)
        with patch("rag.drift.alarm.log_event"):
            alarm.fire(_result(), AlarmLevel.SOFT)
            cb.assert_not_called()


# ---------------------------------------------------------------------------
# DriftAlarm — hard level
# ---------------------------------------------------------------------------


class TestHardAlarm:
    def test_hard_also_calls_log_event(self) -> None:
        alarm = DriftAlarm(_cfg(webhook_url="https://example.com/hook"))
        with (
            patch("rag.drift.alarm.log_event") as mock_log,
            patch("rag.drift.alarm.httpx.Client") as mock_client_cls,
        ):
            mock_client_cls.return_value.__enter__.return_value.post.return_value = _ok_response()
            alarm.fire(_result(), AlarmLevel.HARD)
            mock_log.assert_called_once()

    def test_hard_posts_to_webhook(self) -> None:
        alarm = DriftAlarm(_cfg(webhook_url="https://hooks.example.com/alert"))
        mock_client = MagicMock()
        mock_client.post.return_value = _ok_response()
        with (
            patch("rag.drift.alarm.log_event"),
            patch("rag.drift.alarm.httpx.Client") as mock_client_cls,
        ):
            mock_client_cls.return_value.__enter__.return_value = mock_client
            alarm.fire(_result(), AlarmLevel.HARD)
            mock_client.post.assert_called_once()
            assert mock_client.post.call_args[0][0] == "https://hooks.example.com/alert"

    def test_hard_payload_is_json(self) -> None:
        alarm = DriftAlarm(_cfg(webhook_url="https://example.com/hook"))
        mock_client = MagicMock()
        mock_client.post.return_value = _ok_response()
        with (
            patch("rag.drift.alarm.log_event"),
            patch("rag.drift.alarm.httpx.Client") as mock_client_cls,
        ):
            mock_client_cls.return_value.__enter__.return_value = mock_client
            alarm.fire(_result(statistic=0.9), AlarmLevel.HARD)
            kwargs = mock_client.post.call_args[1]
            assert "json" in kwargs
            assert kwargs["json"]["statistic"] == pytest.approx(0.9)
            assert kwargs["json"]["alert"] == "drift_detected"

    def test_hard_skips_webhook_when_url_empty(self) -> None:
        alarm = DriftAlarm(_cfg(webhook_url=""))
        with (
            patch("rag.drift.alarm.log_event"),
            patch("rag.drift.alarm.httpx.Client") as mock_client_cls,
        ):
            alarm.fire(_result(), AlarmLevel.HARD)
            mock_client_cls.assert_not_called()

    def test_hard_webhook_failure_does_not_raise(self) -> None:
        alarm = DriftAlarm(_cfg(webhook_url="https://example.com/hook"))
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("refused")
        with (
            patch("rag.drift.alarm.log_event"),
            patch("rag.drift.alarm.httpx.Client") as mock_client_cls,
        ):
            mock_client_cls.return_value.__enter__.return_value = mock_client
            alarm.fire(_result(), AlarmLevel.HARD)  # must not raise

    def test_hard_does_not_call_reindex_callback(self) -> None:
        cb = MagicMock()
        alarm = DriftAlarm(_cfg(webhook_url="https://example.com/hook"), re_index_callback=cb)
        mock_client = MagicMock()
        mock_client.post.return_value = _ok_response()
        with (
            patch("rag.drift.alarm.log_event"),
            patch("rag.drift.alarm.httpx.Client") as mock_client_cls,
        ):
            mock_client_cls.return_value.__enter__.return_value = mock_client
            alarm.fire(_result(), AlarmLevel.HARD)
            cb.assert_not_called()


# ---------------------------------------------------------------------------
# DriftAlarm — auto level
# ---------------------------------------------------------------------------


class TestAutoAlarm:
    def _auto_alarm(self, cb: MagicMock, webhook: str = "") -> DriftAlarm:
        return DriftAlarm(_cfg(webhook_url=webhook), re_index_callback=cb)

    def test_auto_calls_log_event(self) -> None:
        alarm = self._auto_alarm(MagicMock())
        with patch("rag.drift.alarm.log_event") as mock_log:
            alarm.fire(_result(), AlarmLevel.AUTO)
            mock_log.assert_called_once()

    def test_auto_calls_reindex_callback(self) -> None:
        cb = MagicMock()
        alarm = self._auto_alarm(cb)
        with patch("rag.drift.alarm.log_event"):
            alarm.fire(_result(), AlarmLevel.AUTO)
            cb.assert_called_once()

    def test_auto_also_posts_webhook_when_url_set(self) -> None:
        cb = MagicMock()
        alarm = self._auto_alarm(cb, webhook="https://example.com/hook")
        mock_client = MagicMock()
        mock_client.post.return_value = _ok_response()
        with (
            patch("rag.drift.alarm.log_event"),
            patch("rag.drift.alarm.httpx.Client") as mock_client_cls,
        ):
            mock_client_cls.return_value.__enter__.return_value = mock_client
            alarm.fire(_result(), AlarmLevel.AUTO)
            mock_client.post.assert_called_once()
            cb.assert_called_once()

    def test_auto_no_callback_registered_does_not_raise(self) -> None:
        alarm = DriftAlarm(_cfg())
        with patch("rag.drift.alarm.log_event"):
            alarm.fire(_result(), AlarmLevel.AUTO)  # must not raise

    def test_auto_callback_called_exactly_once(self) -> None:
        cb = MagicMock()
        alarm = self._auto_alarm(cb)
        with patch("rag.drift.alarm.log_event"):
            alarm.fire(_result(), AlarmLevel.AUTO)
            assert cb.call_count == 1
