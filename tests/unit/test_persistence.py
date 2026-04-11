"""Unit tests for rag.persistence.DriftStore."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from rag.models import DriftResult
from rag.persistence import DriftStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(
    *,
    statistic: float = 0.3,
    pvalue: float = 0.04,
    drifted: bool = True,
    window_size: int = 50,
    snapshot_size: int = 200,
    evaluated_at: datetime | None = None,
) -> DriftResult:
    return DriftResult(
        statistic=statistic,
        pvalue=pvalue,
        drifted=drifted,
        window_size=window_size,
        snapshot_size=snapshot_size,
        evaluated_at=evaluated_at or datetime(2026, 1, 1, 12, 0, 0),
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> Generator[DriftStore, None, None]:
    """DriftStore backed by a temporary SQLite file; closed after the test."""
    s = DriftStore(tmp_path / "test.db")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# save_window
# ---------------------------------------------------------------------------


class TestSaveWindow:
    def test_returns_window_id_string(self, store: DriftStore) -> None:
        wid = store.save_window(_result())
        assert isinstance(wid, str)
        assert len(wid) > 0

    def test_custom_window_id_is_preserved(self, store: DriftStore) -> None:
        wid = store.save_window(_result(), window_id="my-id-123")
        assert wid == "my-id-123"

    def test_duplicate_window_id_raises(self, store: DriftStore) -> None:
        import sqlite3

        store.save_window(_result(), window_id="dup")
        with pytest.raises(sqlite3.IntegrityError):
            store.save_window(_result(), window_id="dup")

    def test_triggered_reindex_stored(self, store: DriftStore) -> None:
        store.save_window(_result(), triggered_reindex=True, window_id="ri")
        row = store._conn.execute(
            "SELECT triggered_reindex FROM drift_history WHERE window_id = 'ri'"
        ).fetchone()
        assert row is not None
        assert row[0] == 1

    def test_not_triggered_reindex_stored(self, store: DriftStore) -> None:
        store.save_window(_result(), triggered_reindex=False, window_id="nri")
        row = store._conn.execute(
            "SELECT triggered_reindex FROM drift_history WHERE window_id = 'nri'"
        ).fetchone()
        assert row is not None
        assert row[0] == 0


# ---------------------------------------------------------------------------
# load_history
# ---------------------------------------------------------------------------


class TestLoadHistory:
    def test_empty_store_returns_empty_list(self, store: DriftStore) -> None:
        assert store.load_history() == []

    def test_returns_drift_result_objects(self, store: DriftStore) -> None:
        store.save_window(_result())
        rows = store.load_history()
        assert len(rows) == 1
        assert isinstance(rows[0], DriftResult)

    def test_fields_round_trip(self, store: DriftStore) -> None:
        r = _result(statistic=0.77, pvalue=0.02, drifted=True, window_size=25, snapshot_size=100)
        store.save_window(r)
        loaded = store.load_history()[0]
        assert loaded.statistic == pytest.approx(0.77)
        assert loaded.pvalue == pytest.approx(0.02)
        assert loaded.drifted is True
        assert loaded.window_size == 25
        assert loaded.snapshot_size == 100

    def test_ordered_oldest_first(self, store: DriftStore) -> None:
        store.save_window(_result(evaluated_at=datetime(2026, 1, 3)), window_id="c")
        store.save_window(_result(evaluated_at=datetime(2026, 1, 1)), window_id="a")
        store.save_window(_result(evaluated_at=datetime(2026, 1, 2)), window_id="b")
        rows = store.load_history()
        ts = [r.evaluated_at for r in rows]
        assert ts == sorted(ts)

    def test_limit_caps_rows(self, store: DriftStore) -> None:
        for i in range(5):
            store.save_window(
                _result(evaluated_at=datetime(2026, 1, i + 1)),
                window_id=f"w{i}",
            )
        rows = store.load_history(limit=3)
        assert len(rows) == 3

    def test_drifted_only_filters(self, store: DriftStore) -> None:
        store.save_window(_result(drifted=True), window_id="d1")
        store.save_window(_result(drifted=False), window_id="nd")
        store.save_window(_result(drifted=True), window_id="d2")
        rows = store.load_history(drifted_only=True)
        assert len(rows) == 2
        assert all(r.drifted for r in rows)

    def test_multiple_saves_accumulate(self, store: DriftStore) -> None:
        for i in range(4):
            store.save_window(
                _result(evaluated_at=datetime(2026, 1, i + 1)),
                window_id=f"w{i}",
            )
        assert len(store.load_history()) == 4


# ---------------------------------------------------------------------------
# export_to_wandb
# ---------------------------------------------------------------------------


class TestExportToWandb:
    def test_calls_log_event_for_each_row(self, store: DriftStore) -> None:
        store.save_window(_result(), window_id="e1")
        store.save_window(_result(), window_id="e2")
        with patch("rag.persistence.log_event") as mock_log:
            store.export_to_wandb()
        assert mock_log.call_count == 2

    def test_event_name_is_drift_history(self, store: DriftStore) -> None:
        store.save_window(_result(), window_id="ev")
        with patch("rag.persistence.log_event") as mock_log:
            store.export_to_wandb()
        assert mock_log.call_args[0][0] == "drift_history"

    def test_payload_contains_score(self, store: DriftStore) -> None:
        store.save_window(_result(statistic=0.55), window_id="sc")
        with patch("rag.persistence.log_event") as mock_log:
            store.export_to_wandb()
        data: dict[str, object] = mock_log.call_args[0][1]
        assert data["score"] == pytest.approx(0.55)

    def test_payload_contains_window_id(self, store: DriftStore) -> None:
        store.save_window(_result(), window_id="check-id")
        with patch("rag.persistence.log_event") as mock_log:
            store.export_to_wandb()
        data = mock_log.call_args[0][1]
        assert data["window_id"] == "check-id"

    def test_export_empty_store_is_noop(self, store: DriftStore) -> None:
        with patch("rag.persistence.log_event") as mock_log:
            store.export_to_wandb()
        mock_log.assert_not_called()

    def test_triggered_reindex_in_payload(self, store: DriftStore) -> None:
        store.save_window(_result(), triggered_reindex=True, window_id="tr")
        with patch("rag.persistence.log_event") as mock_log:
            store.export_to_wandb()
        data = mock_log.call_args[0][1]
        assert data["triggered_reindex"] == 1


# ---------------------------------------------------------------------------
# WAL mode
# ---------------------------------------------------------------------------


class TestWalMode:
    def test_journal_mode_is_wal(self, store: DriftStore) -> None:
        row = store._conn.execute("PRAGMA journal_mode").fetchone()
        assert row is not None
        assert row[0] == "wal"


# ---------------------------------------------------------------------------
# Persistence across reconnections
# ---------------------------------------------------------------------------


class TestReconnect:
    def test_data_survives_reopen(self, tmp_path: Path) -> None:
        db = tmp_path / "reopen.db"
        s1 = DriftStore(db)
        s1.save_window(_result(statistic=0.42), window_id="persist")
        s1.close()

        s2 = DriftStore(db)
        rows = s2.load_history()
        s2.close()

        assert len(rows) == 1
        assert rows[0].statistic == pytest.approx(0.42)
