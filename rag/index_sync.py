"""Keeps every replica on the same FAISS index.

The index file on the shared volume is the source of truth for contents;
``rag:index:version`` in Redis says which version is current. A writer (ingest,
re-index, first-boot seeding) holds ``rag:index:lock``, catches up, mutates,
saves the file, then sets the version key and publishes on
``rag:index:updates``. Other replicas reload on the pub/sub message, and also
compare versions every ``poll_interval_s`` so a missed message (subscriber
reconnecting, replica restarting) is caught up anyway. Reloads are idempotent:
a replica whose local version already equals the shared one does nothing.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from rag.logging import get_logger
from rag.metrics import INDEX_RELOADS
from rag.redis_health import REDIS_FAILURES, RedisHealth
from rag.vector_store.base import VectorStore

logger = get_logger(__name__)

VERSION_KEY = "rag:index:version"
CHANNEL = "rag:index:updates"
LOCK_KEY = "rag:index:lock"

_MAX_WAIT_S = 0.5

_RELEASE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


@functools.lru_cache(maxsize=64)
def _version_of(model_name: str, fingerprint: str) -> str:
    return hashlib.sha256(f"{model_name}\n{fingerprint}".encode()).hexdigest()[:16]


def index_version(store: VectorStore, model_name: str) -> str:
    """Content-derived index version: model name + the store's fingerprint.

    The FAISS store's fingerprint covers its snapshot ID and chunk IDs, never
    the vectors, and is memoised per generation, so this is cheap per query.
    """
    return _version_of(model_name, str(store.content_fingerprint()))


class IndexSync:
    """Pub/sub + polling reconciliation of the shared index version.

    Args:
        health: Shared Redis health tracker (owns the client).
        local_version: Returns this replica's current index version.
        reload: Reloads the index from the shared file (and restarts the
            drift monitor); must be safe to call repeatedly.
        poll_interval_s: Version check period; also the pub/sub read timeout.
        lock_timeout_s: Writer lease; bounds how long a crashed writer blocks others.
        lock_wait_s: How long a writer waits for the lease before proceeding
            without it (logged).
    """

    def __init__(
        self,
        health: RedisHealth,
        *,
        local_version: Callable[[], str],
        reload: Callable[[], None],
        poll_interval_s: float = 5.0,
        lock_timeout_s: float = 120.0,
        lock_wait_s: float = 120.0,
    ) -> None:
        self._health = health
        self._client: Any = health.client
        self._local_version = local_version
        self._reload = reload
        self._poll_interval_s = poll_interval_s
        self._lock_timeout_ms = int(lock_timeout_s * 1000)
        self._lock_wait_s = lock_wait_s
        # Serialises reloads with local writes: a poll must never reload the
        # old file between a writer's mutation and its publish.
        self._local_lock = threading.RLock()
        self._writer_depth = threading.local()
        self._release = self._client.register_script(_RELEASE_LUA)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Version reconciliation
    # ------------------------------------------------------------------

    def remote_version(self) -> str | None:
        """The shared version, or ``None`` if unset or Redis is unavailable."""
        if not self._health.available():
            return None
        try:
            raw = self._client.get(VERSION_KEY)
        except REDIS_FAILURES as exc:
            self._health.mark_down("index_sync", exc)
            return None
        return raw.decode() if raw is not None else None

    def sync(self) -> bool:
        """Reload if the shared version differs from ours.

        Returns:
            ``True`` if a reload brought this replica to the shared version.
        """
        with self._local_lock:
            remote = self.remote_version()
            if remote is None or remote == self._local_version():
                return False
            self._reload()
            if self._local_version() != remote:
                # The file was rewritten again after that version was
                # published; the next poll will see the newer version.
                logger.info("index file ahead of announced version %s; will re-check", remote)
                return False
            INDEX_RELOADS.inc()
            logger.info("index reloaded to shared version %s", remote)
            return True

    def publish(self, version: str) -> None:
        """Announce *version* as current. Call after the file is saved."""
        if not self._health.available():
            return
        try:
            self._client.set(VERSION_KEY, version)
            self._client.publish(CHANNEL, version)
        except REDIS_FAILURES as exc:
            self._health.mark_down("index_sync", exc)

    # ------------------------------------------------------------------
    # Writer lease
    # ------------------------------------------------------------------

    @contextmanager
    def writer(self) -> Iterator[None]:
        """Hold the cross-replica write lease and catch up before mutating.

        Re-entrant per thread (seeding runs an ingest inside startup's lease).
        Without Redis the body still runs — local-only, logged by health.
        """
        depth = getattr(self._writer_depth, "value", 0)
        with self._local_lock:
            if depth:
                self._writer_depth.value = depth + 1
                try:
                    yield
                finally:
                    self._writer_depth.value = depth
                return
            token = self._acquire()
            self._writer_depth.value = 1
            try:
                self.sync()
                yield
            finally:
                self._writer_depth.value = 0
                if token is not None:
                    try:
                        self._release(keys=[LOCK_KEY], args=[token])
                    except REDIS_FAILURES as exc:
                        self._health.mark_down("index_sync", exc)

    def _acquire(self) -> str | None:
        token = uuid.uuid4().hex
        deadline = time.monotonic() + self._lock_wait_s
        while self._health.available():
            try:
                if self._client.set(LOCK_KEY, token, nx=True, px=self._lock_timeout_ms):
                    return token
            except REDIS_FAILURES as exc:
                self._health.mark_down("index_sync", exc)
                break
            if time.monotonic() >= deadline:
                logger.warning(
                    "index write lease busy for %.0fs; writing anyway", self._lock_wait_s
                )
                break
            time.sleep(0.05)
        return None

    # ------------------------------------------------------------------
    # Background listener
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the listener thread (idempotent)."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="index-sync", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the listener thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_MAX_WAIT_S + self._poll_interval_s)
            self._thread = None

    def _run(self) -> None:
        pubsub: Any = None
        next_poll = 0.0
        while not self._stop.is_set():
            try:
                if pubsub is None and self._health.available():
                    pubsub = self._client.pubsub(ignore_subscribe_messages=True)
                    pubsub.subscribe(CHANNEL)
                # Wake at least every 0.5 s so stop() stays responsive.
                wait_s = max(0.0, min(next_poll - time.monotonic(), _MAX_WAIT_S))
                message = None
                if pubsub is not None:
                    message = pubsub.get_message(timeout=wait_s)
                else:
                    self._stop.wait(wait_s)
                # Reconcile on an announcement, and on every poll interval to
                # catch announcements this replica missed. sync() is a no-op
                # when already current.
                if message is not None or time.monotonic() >= next_poll:
                    next_poll = time.monotonic() + self._poll_interval_s
                    self.sync()
            except REDIS_FAILURES as exc:
                self._health.mark_down("index_sync", exc)
                self._close(pubsub)
                pubsub = None
                self._stop.wait(self._poll_interval_s)
            except Exception:  # noqa: BLE001 — the listener must never die
                logger.exception("index sync iteration failed")
                self._stop.wait(self._poll_interval_s)
        self._close(pubsub)

    @staticmethod
    def _close(pubsub: Any) -> None:
        if pubsub is not None:
            with contextlib.suppress(*REDIS_FAILURES):
                pubsub.close()
