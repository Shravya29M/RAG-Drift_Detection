"""Shared Redis client plus up/down tracking for graceful degradation.

Every Redis-backed component (query cache, shared drift state, remediation
incidents, index sync) goes through one :class:`RedisHealth`. When a call
fails the component reports it via :meth:`RedisHealth.mark_down` and falls
back to its local behaviour; :meth:`RedisHealth.available` then answers
``False`` without touching the network until the next reconnection probe, so
a dead Redis costs one timeout per retry interval rather than one per request.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import redis

from rag.logging import get_logger
from rag.metrics import REDIS_ERRORS, REDIS_UP
from rag.models import RedisConfig

logger = get_logger(__name__)

# Exceptions that mean "Redis is unusable right now". redis-py wraps socket
# failures in ConnectionError/TimeoutError, both RedisError subclasses; OSError
# covers anything that escapes the wrapping.
REDIS_FAILURES: tuple[type[BaseException], ...] = (redis.RedisError, OSError)


def make_client(config: RedisConfig) -> redis.Redis:
    """Build a binary-safe client with short timeouts from *config*."""
    client: redis.Redis = redis.Redis.from_url(
        config.url,
        socket_timeout=config.socket_timeout_s,
        socket_connect_timeout=config.socket_timeout_s,
        health_check_interval=30,
    )
    return client


class RedisHealth:
    """Owns the Redis client and tracks whether it is currently usable.

    Args:
        client: A redis-py (or fakeredis) client with ``decode_responses=False``.
        retry_interval_s: Minimum seconds between reconnection probes.
    """

    def __init__(self, client: Any, *, retry_interval_s: float = 5.0) -> None:
        self.client = client
        self._retry_interval_s = retry_interval_s
        self._lock = threading.Lock()
        self._up = True
        self._next_probe = 0.0
        REDIS_UP.set(1)

    @property
    def up(self) -> bool:
        """Last known state, without probing."""
        return self._up

    def available(self) -> bool:
        """Return ``True`` if Redis should be used for this call.

        While marked down, pings at most once per retry interval; a
        successful ping flips the state back to up.
        """
        if self._up:
            return True
        with self._lock:
            now = time.monotonic()
            if self._up:
                return True
            if now < self._next_probe:
                return False
            self._next_probe = now + self._retry_interval_s
        try:
            self.client.ping()
        except REDIS_FAILURES:
            return False
        with self._lock:
            if not self._up:
                self._up = True
                REDIS_UP.set(1)
                logger.warning("redis reachable again; leaving local fallback mode")
        return True

    def mark_down(self, component: str, exc: BaseException) -> None:
        """Record a failure from *component* and switch to local fallback mode.

        Logs one warning per up→down transition, not one per failed call.
        """
        REDIS_ERRORS.labels(component=component).inc()
        with self._lock:
            if not self._up:
                return
            self._up = False
            self._next_probe = time.monotonic() + self._retry_interval_s
            REDIS_UP.set(0)
        logger.warning(
            "redis unavailable (%s: %s); degrading to local mode — no query cache, "
            "per-replica drift windows",
            component,
            exc,
        )
