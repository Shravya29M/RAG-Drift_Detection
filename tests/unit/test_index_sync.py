"""IndexSync: version reconciliation, pub/sub + polling catch-up, writer lease."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import fakeredis
import numpy as np
import pytest
import redis

from rag.index_sync import CHANNEL, LOCK_KEY, VERSION_KEY, IndexSync, index_version
from rag.metrics import INDEX_RELOADS
from rag.models import Chunk, ChunkMetadata, SourceType
from rag.redis_health import RedisHealth
from rag.vector_store.faiss_store import FAISSStore

DIM = 4


def _chunk(i: int) -> Chunk:
    return Chunk(
        id=f"c-{i}",
        text=f"t{i}",
        token_count=1,
        metadata=ChunkMetadata(source="s", source_type=SourceType.TEXT, chunk_index=i),
    )


class _Replica:
    """Minimal replica: a version string and a reload that adopts the 'file'."""

    def __init__(self, server: fakeredis.FakeServer, file: dict[str, str], **kw: float) -> None:
        self.version = "v0"
        self.file = file
        self.reloads = 0
        self.health = RedisHealth(fakeredis.FakeRedis(server=server), retry_interval_s=0.0)
        self.sync = IndexSync(
            self.health, local_version=lambda: self.version, reload=self._reload, **kw
        )

    def _reload(self) -> None:
        self.reloads += 1
        self.version = self.file["version"]


@pytest.fixture()
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


class TestIndexVersion:
    def test_depends_on_model_and_contents_not_on_instance(self) -> None:
        a, b = FAISSStore(dim=DIM), FAISSStore(dim=DIM)
        vecs = np.eye(2, DIM, dtype=np.float32)
        a.swap_index([_chunk(0), _chunk(1)], vecs, snapshot_id="snap")
        b.swap_index([_chunk(1), _chunk(0)], vecs[::-1], snapshot_id="snap")
        assert index_version(a, "m") == index_version(b, "m")
        assert index_version(a, "m") != index_version(a, "other-model")
        before = index_version(a, "m")
        a.add([_chunk(2)], np.eye(1, DIM, dtype=np.float32))
        assert index_version(a, "m") != before


class TestSync:
    def test_no_remote_version_is_a_no_op(self, server: fakeredis.FakeServer) -> None:
        r = _Replica(server, {"version": "v1"})
        assert r.sync.remote_version() is None
        assert r.sync.sync() is False and r.reloads == 0

    def test_reloads_when_behind_then_is_idempotent(self, server: fakeredis.FakeServer) -> None:
        r = _Replica(server, {"version": "v1"})
        fakeredis.FakeRedis(server=server).set(VERSION_KEY, "v1")
        before = INDEX_RELOADS._value.get()
        assert r.sync.sync() is True
        assert r.sync.sync() is False
        assert r.reloads == 1 and r.version == "v1"
        assert INDEX_RELOADS._value.get() == before + 1

    def test_file_newer_than_announcement_is_retried_later(
        self, server: fakeredis.FakeServer
    ) -> None:
        r = _Replica(server, {"version": "v2"})
        fakeredis.FakeRedis(server=server).set(VERSION_KEY, "v1")
        assert r.sync.sync() is False
        assert r.version == "v2"

    def test_publish_sets_key_and_notifies(self, server: fakeredis.FakeServer) -> None:
        r = _Replica(server, {"version": "v1"})
        sub = fakeredis.FakeRedis(server=server).pubsub(ignore_subscribe_messages=True)
        sub.subscribe(CHANNEL)
        r.sync.publish("v9")
        assert r.sync.remote_version() == "v9"
        msg = None
        deadline = time.monotonic() + 2.0
        while msg is None and time.monotonic() < deadline:
            msg = sub.get_message(timeout=0.1)  # first call consumes the subscribe ack
        assert msg is not None and msg["data"] == b"v9"

    def test_redis_down_paths_do_not_raise(self, server: fakeredis.FakeServer) -> None:
        r = _Replica(server, {"version": "v1"})
        server.connected = False
        with patch("rag.redis_health.logger"):
            assert r.sync.remote_version() is None
            r.health._up = True
            r.sync.publish("v1")
            assert not r.health.up
            assert r.sync.sync() is False
            r.sync.publish("v1")  # while down: silently skipped


class TestWriter:
    def test_writer_catches_up_before_mutating(self, server: fakeredis.FakeServer) -> None:
        r = _Replica(server, {"version": "v5"})
        fakeredis.FakeRedis(server=server).set(VERSION_KEY, "v5")
        with r.sync.writer():
            assert r.version == "v5"
            assert fakeredis.FakeRedis(server=server).get(LOCK_KEY) is not None
        assert fakeredis.FakeRedis(server=server).get(LOCK_KEY) is None

    def test_writer_is_reentrant(self, server: fakeredis.FakeServer) -> None:
        r = _Replica(server, {"version": "v0"})
        with r.sync.writer(), r.sync.writer():
            assert fakeredis.FakeRedis(server=server).get(LOCK_KEY) is not None
        assert fakeredis.FakeRedis(server=server).get(LOCK_KEY) is None

    def test_writers_on_two_replicas_are_exclusive(self, server: fakeredis.FakeServer) -> None:
        a = _Replica(server, {"version": "v0"})
        b = _Replica(server, {"version": "v0"})
        inside: list[str] = []
        overlap = threading.Event()

        def work(name: str, rep: _Replica) -> None:
            with rep.sync.writer():
                inside.append(name)
                if len(inside) > 1:
                    overlap.set()
                time.sleep(0.1)
                inside.remove(name)

        threads = [
            threading.Thread(target=work, args=("a", a)),
            threading.Thread(target=work, args=("b", b)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not overlap.is_set()

    def test_busy_lease_times_out_and_writes_anyway(self, server: fakeredis.FakeServer) -> None:
        r = _Replica(server, {"version": "v0"}, lock_wait_s=0.0)
        fakeredis.FakeRedis(server=server).set(LOCK_KEY, "stuck")
        with patch("rag.index_sync.logger") as log, r.sync.writer():
            pass
        log.warning.assert_called_once()
        assert fakeredis.FakeRedis(server=server).get(LOCK_KEY) == b"stuck"

    def test_writer_without_redis_still_runs(self, server: fakeredis.FakeServer) -> None:
        r = _Replica(server, {"version": "v0"})
        server.connected = False
        ran = False
        with patch("rag.redis_health.logger"):
            r.health._up = True
            with r.sync.writer():
                ran = True
        assert ran and not r.health.up

    def test_release_failure_is_swallowed(self, server: fakeredis.FakeServer) -> None:
        r = _Replica(server, {"version": "v0"})
        with patch("rag.redis_health.logger"), r.sync.writer():
            server.connected = False
        assert not r.health.up


class TestListener:
    def _wait(self, cond: object, timeout: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():  # type: ignore[operator]
                return True
            time.sleep(0.02)
        return False

    def test_pubsub_message_triggers_reload(self, server: fakeredis.FakeServer) -> None:
        file = {"version": "v0"}
        writer = _Replica(server, file)
        follower = _Replica(server, file, poll_interval_s=30.0)
        follower.sync.start()
        try:
            time.sleep(0.2)  # let the subscription register
            file["version"] = "v1"
            writer.sync.publish("v1")
            assert self._wait(lambda: follower.version == "v1")
        finally:
            follower.sync.stop()

    def test_missed_message_is_caught_by_polling(self, server: fakeredis.FakeServer) -> None:
        file = {"version": "v0"}
        follower = _Replica(server, file, poll_interval_s=0.05)
        follower.sync.start()
        follower.sync.start()  # idempotent
        try:
            file["version"] = "v7"
            # Version key set with no PUBLISH: only the poll can notice.
            fakeredis.FakeRedis(server=server).set(VERSION_KEY, "v7")
            assert self._wait(lambda: follower.version == "v7")
        finally:
            follower.sync.stop()

    def test_listener_survives_redis_outage(self, server: fakeredis.FakeServer) -> None:
        file = {"version": "v0"}
        follower = _Replica(server, file, poll_interval_s=0.05)
        with patch("rag.redis_health.logger"):
            server.connected = False
            follower.sync.start()
            try:
                time.sleep(0.2)
                server.connected = True
                file["version"] = "v3"
                fakeredis.FakeRedis(server=server).set(VERSION_KEY, "v3")
                assert self._wait(lambda: follower.version == "v3")
            finally:
                follower.sync.stop()

    def test_listener_recovers_from_pubsub_errors(self, server: fakeredis.FakeServer) -> None:
        r = _Replica(server, {"version": "v0"}, poll_interval_s=0.01)
        broken = MagicMock()
        broken.get_message.side_effect = redis.ConnectionError("dropped")
        broken.close.side_effect = redis.ConnectionError("already gone")
        with (
            patch.object(r.health.client, "pubsub", return_value=broken),
            patch("rag.redis_health.logger"),
        ):
            r.sync.start()
            time.sleep(0.1)
            r.sync.stop()
        assert broken.get_message.called

    def test_unexpected_errors_are_logged_not_fatal(self, server: fakeredis.FakeServer) -> None:
        r = _Replica(server, {"version": "v0"}, poll_interval_s=0.01)
        with (
            patch.object(r.sync, "sync", side_effect=RuntimeError("bug")),
            patch("rag.index_sync.logger") as log,
        ):
            r.sync.start()
            time.sleep(0.1)
            r.sync.stop()
        assert log.exception.called
