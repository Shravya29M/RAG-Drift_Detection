"""RedisHealth: degrade on failure, probe at most once per interval, recover."""

from __future__ import annotations

from unittest.mock import patch

import fakeredis
import redis

from rag.metrics import REDIS_ERRORS, REDIS_UP
from rag.models import RedisConfig
from rag.redis_health import RedisHealth, make_client


def _gauge(metric: object) -> float:
    return float(metric._value.get())  # type: ignore[attr-defined]


def test_starts_up_and_available() -> None:
    health = RedisHealth(fakeredis.FakeRedis())
    assert health.up and health.available()
    assert _gauge(REDIS_UP) == 1


def test_mark_down_flips_state_counts_error_and_warns_once() -> None:
    health = RedisHealth(fakeredis.FakeRedis(), retry_interval_s=60)
    before = REDIS_ERRORS.labels(component="cache")._value.get()
    with patch("rag.redis_health.logger") as log:
        health.mark_down("cache", redis.ConnectionError("boom"))
        health.mark_down("cache", redis.ConnectionError("boom again"))
    assert not health.up
    assert _gauge(REDIS_UP) == 0
    assert REDIS_ERRORS.labels(component="cache")._value.get() == before + 2
    assert log.warning.call_count == 1  # one warning per transition, not per call


def test_no_probe_before_retry_interval() -> None:
    server = fakeredis.FakeServer()
    health = RedisHealth(fakeredis.FakeRedis(server=server), retry_interval_s=60)
    health.mark_down("drift", redis.ConnectionError("x"))
    with patch.object(health.client, "ping") as ping:
        assert health.available() is False
        ping.assert_not_called()


def test_recovers_after_successful_probe() -> None:
    server = fakeredis.FakeServer()
    health = RedisHealth(fakeredis.FakeRedis(server=server), retry_interval_s=0.0)
    health.mark_down("drift", redis.ConnectionError("x"))
    with patch("rag.redis_health.logger") as log:
        assert health.available() is True
    assert health.up
    assert _gauge(REDIS_UP) == 1
    log.warning.assert_called_once()


def test_failed_probe_stays_down() -> None:
    server = fakeredis.FakeServer()
    server.connected = False
    health = RedisHealth(fakeredis.FakeRedis(server=server), retry_interval_s=0.0)
    health.mark_down("drift", redis.ConnectionError("x"))
    assert health.available() is False
    assert not health.up


def test_concurrent_recovery_is_reported_once() -> None:
    """If another thread already flipped the state back up, the probe result is a no-op."""
    health = RedisHealth(fakeredis.FakeRedis(), retry_interval_s=0.0)
    health.mark_down("drift", redis.ConnectionError("x"))

    def ping_and_recover_elsewhere() -> bool:
        health._up = True  # simulates a racing thread finishing its probe first
        return True

    with (
        patch.object(health.client, "ping", side_effect=ping_and_recover_elsewhere),
        patch("rag.redis_health.logger") as log,
    ):
        assert health.available() is True
    log.warning.assert_not_called()


def test_available_short_circuits_when_recovered_while_waiting() -> None:
    health = RedisHealth(fakeredis.FakeRedis(), retry_interval_s=60)
    health._up = False
    real_lock = health._lock

    class _FlipOnEnter:
        def __enter__(self) -> None:
            real_lock.acquire()
            health._up = True

        def __exit__(self, *exc: object) -> None:
            real_lock.release()

    health._lock = _FlipOnEnter()  # type: ignore[assignment]
    assert health.available() is True


def test_make_client_uses_configured_timeouts() -> None:
    client = make_client(RedisConfig(url="redis://localhost:6399/0", socket_timeout_s=0.25))
    kwargs = client.connection_pool.connection_kwargs
    assert kwargs["socket_timeout"] == 0.25
    assert kwargs["socket_connect_timeout"] == 0.25
    assert kwargs["port"] == 6399
