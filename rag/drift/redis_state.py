"""Redis-backed drift state shared by every replica, plus a local fallback.

Layout (``ns`` = ``rag:drift:<index version>``, so replicas share state only
when they serve the same index):

* ``{ns}:window``  — list of packed query samples, the rolling window
* ``{ns}:pending`` — list of full windows claimed but not yet evaluated
* ``{ns}:state``   — hash: baseline, baseline mean score, hysteresis counter,
  ``reindex_triggered``
* ``{ns}:history`` — list of ``DriftResult`` JSON, oldest first
* ``{ns}:lock``    — evaluation lease (token-fenced)

Appending is one Lua script: push the samples, and while the window holds at
least ``window_size`` samples move exactly that many into ``pending`` as one
blob. Every sample therefore lands in exactly one window no matter how many
replicas push at once. Evaluation runs in Python (the KS test needs numpy)
under a lease; each window is popped, run through the same
:func:`~rag.drift.detector.close_window` the local detector uses, and
committed by a second script that checks the lease token first — a replica
whose lease expired mid-evaluation cannot overwrite a newer state.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from typing import Any

import numpy as np

from rag.drift.detector import (
    DriftDetector,
    HysteresisState,
    WindowOutcome,
    close_window,
    summarize_window,
)
from rag.drift.snapshot import DistributionSnapshot
from rag.logging import get_logger
from rag.metrics import DRIFT_WINDOWS_EVALUATED
from rag.models import DriftConfig, DriftResult
from rag.redis_health import REDIS_FAILURES, RedisHealth

logger = get_logger(__name__)

# Most recent results kept in shared history (the local detector is unbounded,
# but shared memory is not free).
HISTORY_LIMIT = 1000

_APPEND_LUA = """
local size = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
for i = 3, #ARGV do
  redis.call('RPUSH', KEYS[1], ARGV[i])
end
local claimed = 0
while redis.call('LLEN', KEYS[1]) >= size do
  local items = redis.call('LRANGE', KEYS[1], 0, size - 1)
  redis.call('LTRIM', KEYS[1], size, -1)
  redis.call('RPUSH', KEYS[2], table.concat(items))
  claimed = claimed + 1
end
redis.call('PEXPIRE', KEYS[1], ttl)
if claimed > 0 then redis.call('PEXPIRE', KEYS[2], ttl) end
return claimed
"""

# KEYS: lock, state, history. ARGV: token, ttl_ms, history_limit, result_json
# ('' for the calibration window), then state field/value pairs.
_COMMIT_LUA = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
  return 0
end
redis.call('HSET', KEYS[2], unpack(ARGV, 5))
redis.call('PEXPIRE', KEYS[2], tonumber(ARGV[2]))
if ARGV[4] ~= '' then
  redis.call('RPUSH', KEYS[3], ARGV[4])
  redis.call('LTRIM', KEYS[3], -tonumber(ARGV[3]), -1)
  redis.call('PEXPIRE', KEYS[3], tonumber(ARGV[2]))
end
return 1
"""

_RELEASE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


def pack_sample(embedding: np.ndarray, top_score: float | None) -> bytes:
    """Encode one query sample as ``[has_score, score, *embedding]`` float32 bytes."""
    head = np.array(
        [0.0 if top_score is None else 1.0, 0.0 if top_score is None else top_score],
        dtype=np.float32,
    )
    return head.tobytes() + np.asarray(embedding, dtype=np.float32).tobytes()


def unpack_window(blob: bytes, window_size: int) -> list[tuple[np.ndarray, float | None]]:
    """Split a claimed window blob back into ``(embedding, score)`` samples."""
    rows = np.frombuffer(blob, dtype=np.float32).reshape(window_size, -1)
    return [(row[2:].copy(), float(row[1]) if row[0] else None) for row in rows]


def _encode_state(state: HysteresisState) -> list[bytes | str]:
    baseline = b"" if state.baseline is None else state.baseline.astype(np.float32).tobytes()
    rows = 0 if state.baseline is None else int(state.baseline.shape[0])
    mean = "" if state.baseline_mean_score is None else repr(state.baseline_mean_score)
    return [
        "baseline",
        baseline,
        "baseline_rows",
        str(rows),
        "baseline_mean_score",
        mean,
        "consecutive_alerts",
        str(state.consecutive_alerts),
        "reindex_triggered",
        "1" if state.reindex_triggered else "0",
    ]


def _decode_state(raw: dict[bytes, bytes]) -> HysteresisState:
    if not raw:
        return HysteresisState()
    rows = int(raw.get(b"baseline_rows", b"0"))
    baseline = None
    if rows:
        baseline = np.frombuffer(raw[b"baseline"], dtype=np.float32).reshape(rows, -1).copy()
    mean_raw = raw.get(b"baseline_mean_score", b"")
    return HysteresisState(
        baseline=baseline,
        baseline_mean_score=float(mean_raw) if mean_raw else None,
        consecutive_alerts=int(raw.get(b"consecutive_alerts", b"0")),
        reindex_triggered=raw.get(b"reindex_triggered") == b"1",
    )


class RedisDriftDetector:
    """A :class:`~rag.drift.detector.DriftMonitor` whose state lives in Redis.

    Every method may raise a Redis exception; :class:`ResilientDriftMonitor`
    is the component that catches them and falls back.

    Args:
        client: redis-py/fakeredis client (binary responses).
        snapshot: PCA basis fitted on this replica's (identical) corpus.
        config: Drift configuration; window size and hysteresis must match
            across replicas.
        namespace: Key prefix, normally ``rag:drift:<index version>``.
        state_ttl_s: Idle expiry refreshed on every write.
        lock_timeout_s: Evaluation lease length.
        lock_wait_s: How long to wait for another replica's lease before
            leaving pending windows for the next caller.
    """

    def __init__(
        self,
        client: Any,
        snapshot: DistributionSnapshot,
        config: DriftConfig,
        namespace: str,
        *,
        state_ttl_s: float = 7 * 24 * 3600.0,
        lock_timeout_s: float = 10.0,
        lock_wait_s: float = 5.0,
    ) -> None:
        self._client = client
        self._snapshot = snapshot
        self._config = config
        self._ttl_ms = int(state_ttl_s * 1000)
        self._lock_timeout_ms = int(lock_timeout_s * 1000)
        self._lock_wait_s = lock_wait_s
        self.namespace = namespace
        self._window_key = f"{namespace}:window"
        self._pending_key = f"{namespace}:pending"
        self._state_key = f"{namespace}:state"
        self._history_key = f"{namespace}:history"
        self._lock_key = f"{namespace}:lock"
        self._append = client.register_script(_APPEND_LUA)
        self._commit = client.register_script(_COMMIT_LUA)
        self._release = client.register_script(_RELEASE_LUA)

    # ------------------------------------------------------------------
    # DriftMonitor read interface
    # ------------------------------------------------------------------

    def _state(self) -> HysteresisState:
        return _decode_state(self._client.hgetall(self._state_key))

    @property
    def history(self) -> list[DriftResult]:
        raw: list[bytes] = self._client.lrange(self._history_key, 0, -1)
        return [DriftResult.model_validate_json(item) for item in raw]

    @property
    def consecutive_alerts(self) -> int:
        return int(self._client.hget(self._state_key, "consecutive_alerts") or 0)

    @property
    def reindex_triggered(self) -> bool:
        return bool(self._client.hget(self._state_key, "reindex_triggered") == b"1")

    @property
    def buffer_size(self) -> int:
        return int(self._client.llen(self._window_key))

    @property
    def baseline_ready(self) -> bool:
        return int(self._client.hget(self._state_key, "baseline_rows") or 0) > 0

    @property
    def baseline_mean_score(self) -> float | None:
        raw = self._client.hget(self._state_key, "baseline_mean_score")
        return float(raw) if raw else None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_query_embeddings(
        self, batch: Sequence[tuple[np.ndarray, float | None]]
    ) -> list[WindowOutcome]:
        """Append *batch* atomically, then evaluate any windows it completed.

        Also drains windows other replicas claimed but left pending, so no
        window is stranded when a lease holder dies.
        """
        if batch:
            samples = [pack_sample(e, s) for e, s in batch]
            self._append(
                keys=[self._window_key, self._pending_key],
                args=[int(self._config.window_size), self._ttl_ms, *samples],
            )
        return self._drain()

    def reset(self) -> None:
        """Delete all shared state for this namespace; the next window recalibrates."""
        self._client.delete(self._window_key, self._pending_key, self._state_key, self._history_key)

    def _acquire(self, token: str) -> bool:
        deadline = time.monotonic() + self._lock_wait_s
        while True:
            if self._client.set(self._lock_key, token, nx=True, px=self._lock_timeout_ms):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)

    def _drain(self) -> list[WindowOutcome]:
        outcomes: list[WindowOutcome] = []
        while self._client.llen(self._pending_key) > 0:
            token = uuid.uuid4().hex
            if not self._acquire(token):
                # Another replica holds the lease and will drain what is pending.
                break
            try:
                while True:
                    blob = self._client.lpop(self._pending_key)
                    if blob is None:
                        break
                    outcome = self._evaluate(blob, token)
                    if outcome is not None:
                        outcomes.append(outcome)
            finally:
                self._release(keys=[self._lock_key], args=[token])
        return outcomes

    def _evaluate(self, blob: bytes, token: str) -> WindowOutcome | None:
        window_size = int(self._config.window_size)
        window, mean_score = summarize_window(unpack_window(blob, window_size))
        new_state, result = close_window(
            self._snapshot, self._config, self._state(), window, mean_score
        )
        committed = self._commit(
            keys=[self._lock_key, self._state_key, self._history_key],
            args=[
                token,
                self._ttl_ms,
                HISTORY_LIMIT,
                "" if result is None else result.model_dump_json(),
                *_encode_state(new_state),
            ],
        )
        if not committed:
            logger.warning("drift lease expired mid-evaluation; window discarded")
            return None
        if result is None:
            return None
        DRIFT_WINDOWS_EVALUATED.labels(mode="shared").inc()
        return WindowOutcome(result, new_state.reindex_triggered)


class ResilientDriftMonitor:
    """Uses the shared Redis detector while Redis is up, a local one otherwise.

    On failover the local detector carries on with whatever calibration it
    had. When Redis returns, the local partial window is discarded (it was
    never part of any shared window) and shared state resumes.

    Args:
        shared: The Redis-backed detector.
        local: This replica's in-process detector.
        health: Shared Redis health tracker.
    """

    def __init__(
        self, shared: RedisDriftDetector, local: DriftDetector, health: RedisHealth
    ) -> None:
        self.shared = shared
        self.local = local
        self._health = health
        self._using_local = False

    def _use_shared(self) -> bool:
        if not self._health.available():
            self._using_local = True
            return False
        if self._using_local:
            self._using_local = False
            if self.local.buffer_size:
                logger.warning(
                    "redis back; discarding %d locally buffered drift samples",
                    self.local.buffer_size,
                )
            self.local.discard_buffer()
        return True

    def _read(self, name: str) -> Any:
        if self._use_shared():
            try:
                return getattr(self.shared, name)
            except REDIS_FAILURES as exc:
                self._health.mark_down("drift", exc)
                self._using_local = True
        return getattr(self.local, name)

    @property
    def history(self) -> list[DriftResult]:
        result: list[DriftResult] = self._read("history")
        return result

    @property
    def consecutive_alerts(self) -> int:
        return int(self._read("consecutive_alerts"))

    @property
    def reindex_triggered(self) -> bool:
        return bool(self._read("reindex_triggered"))

    @property
    def buffer_size(self) -> int:
        return int(self._read("buffer_size"))

    @property
    def baseline_ready(self) -> bool:
        return bool(self._read("baseline_ready"))

    @property
    def baseline_mean_score(self) -> float | None:
        value: float | None = self._read("baseline_mean_score")
        return value

    def add_query_embeddings(
        self, batch: Sequence[tuple[np.ndarray, float | None]]
    ) -> list[WindowOutcome]:
        if self._use_shared():
            try:
                return self.shared.add_query_embeddings(batch)
            except REDIS_FAILURES as exc:
                self._health.mark_down("drift", exc)
                self._using_local = True
        outcomes = self.local.add_query_embeddings(batch)
        if outcomes:
            DRIFT_WINDOWS_EVALUATED.labels(mode="local").inc(len(outcomes))
        return outcomes

    def reset(self) -> None:
        self.local.reset()
        if self._use_shared():
            try:
                self.shared.reset()
            except REDIS_FAILURES as exc:
                self._health.mark_down("drift", exc)
                self._using_local = True
