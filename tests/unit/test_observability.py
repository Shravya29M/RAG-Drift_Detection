"""Logging and W&B tracking: thin wrappers, but every other module routes
through them, so a regression here is silent and repo-wide."""

from __future__ import annotations

import json
import logging

import pytest

from rag.logging import _JsonFormatter, get_logger
from rag.tracking import log_event


def make_record(**kwargs) -> logging.LogRecord:
    defaults = {
        "name": "rag.test",
        "level": logging.INFO,
        "pathname": __file__,
        "lineno": 1,
        "msg": "hello",
        "args": (),
        "exc_info": None,
    }
    defaults.update(kwargs)
    return logging.LogRecord(**defaults)


# --------------------------------------------------------------------------
# JSON formatting
# --------------------------------------------------------------------------


def test_records_render_as_single_line_json():
    text = _JsonFormatter().format(make_record())
    assert "\n" not in text
    assert json.loads(text)["message"] == "hello"


def test_formatted_record_carries_level_name_and_time():
    payload = json.loads(_JsonFormatter().format(make_record()))
    assert payload["level"] == "INFO"
    assert payload["name"] == "rag.test"
    assert payload["time"]


def test_message_args_are_interpolated():
    record = make_record(msg="got %d hits", args=(3,))
    assert json.loads(_JsonFormatter().format(record))["message"] == "got 3 hits"


def test_a_record_without_an_exception_omits_exc_info():
    assert "exc_info" not in json.loads(_JsonFormatter().format(make_record()))


def test_an_exception_record_includes_the_traceback():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = make_record(exc_info=sys.exc_info())
    payload = json.loads(_JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exc_info"]


# --------------------------------------------------------------------------
# get_logger
# --------------------------------------------------------------------------


def test_get_logger_returns_a_configured_logger():
    logger = get_logger("rag.test.configured")
    assert logger.handlers
    assert isinstance(logger.handlers[0].formatter, _JsonFormatter)


def test_get_logger_does_not_stack_handlers_on_repeated_calls():
    name = "rag.test.idempotent"
    first = get_logger(name)
    count = len(first.handlers)
    assert get_logger(name) is first
    assert len(first.handlers) == count


# --------------------------------------------------------------------------
# tracking
# --------------------------------------------------------------------------


def test_log_event_no_ops_when_wandb_is_disabled(monkeypatch):
    import rag.tracking as tracking

    monkeypatch.setenv("WANDB_DISABLED", "true")

    def fail(*_args, **_kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("wandb.log must not be called when disabled")

    monkeypatch.setattr(tracking.wandb, "log", fail)
    log_event("drift_window", {"score": 1.0})


def test_log_event_no_ops_when_no_run_is_active(monkeypatch):
    import rag.tracking as tracking

    monkeypatch.setenv("WANDB_DISABLED", "false")
    monkeypatch.setattr(tracking.wandb, "run", None)

    def fail(*_args, **_kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("wandb.log must not be called without a run")

    monkeypatch.setattr(tracking.wandb, "log", fail)
    log_event("drift_window", {"score": 1.0})


def test_log_event_prefixes_keys_with_the_event_name(monkeypatch):
    import rag.tracking as tracking

    monkeypatch.setenv("WANDB_DISABLED", "false")
    monkeypatch.setattr(tracking.wandb, "run", object())
    captured: dict[str, object] = {}
    monkeypatch.setattr(tracking.wandb, "log", captured.update)

    log_event("drift_window", {"score": 1.0, "window_size": 32})
    assert captured == {"drift_window/score": 1.0, "drift_window/window_size": 32}


@pytest.mark.parametrize("value", ["TRUE", "True", "true"])
def test_wandb_disabled_check_is_case_insensitive(monkeypatch, value):
    import rag.tracking as tracking

    monkeypatch.setenv("WANDB_DISABLED", value)
    monkeypatch.setattr(
        tracking.wandb,
        "log",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("called")),
    )
    log_event("e", {"k": 1})
