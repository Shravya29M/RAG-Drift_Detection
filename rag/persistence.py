"""SQLite persistence layer for drift evaluation history."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from rag.models import DriftResult
from rag.tracking import log_event

_DDL = """
CREATE TABLE IF NOT EXISTS drift_history (
    window_id        TEXT    PRIMARY KEY,
    score            REAL    NOT NULL,
    pvalue           REAL    NOT NULL,
    drifted          INTEGER NOT NULL,
    window_size      INTEGER NOT NULL,
    snapshot_size    INTEGER NOT NULL,
    timestamp        TEXT    NOT NULL,
    triggered_reindex INTEGER NOT NULL
);
"""

_PRAGMAS = "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;"


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the SQLite database at *db_path* with WAL mode enabled."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.executescript(_PRAGMAS)
    conn.executescript(_DDL)
    conn.commit()
    return conn


class DriftStore:
    """Persistent SQLite store for drift evaluation windows.

    All writes go to a single ``drift_history`` table in WAL mode for safe
    concurrent access from the background scheduler thread.

    Args:
        db_path: Path to the SQLite file.  Created on first use if absent.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = _connect(db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_window(
        self,
        result: DriftResult,
        *,
        triggered_reindex: bool = False,
        window_id: str | None = None,
    ) -> str:
        """Persist one drift evaluation window to the database.

        Args:
            result: The evaluation result to store.
            triggered_reindex: Whether this window caused a re-index.
            window_id: Override the auto-generated UUID4 identifier.

        Returns:
            The ``window_id`` string used for this record.
        """
        wid = window_id or str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO drift_history
                (window_id, score, pvalue, drifted, window_size,
                 snapshot_size, timestamp, triggered_reindex)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                wid,
                result.statistic,
                result.pvalue,
                int(result.drifted),
                result.window_size,
                result.snapshot_size,
                result.evaluated_at.isoformat(),
                int(triggered_reindex),
            ),
        )
        self._conn.commit()
        return wid

    def load_history(
        self,
        *,
        limit: int | None = None,
        drifted_only: bool = False,
    ) -> list[DriftResult]:
        """Load drift evaluation records from the database, oldest first.

        Args:
            limit: Cap the number of rows returned.  ``None`` returns all.
            drifted_only: When ``True``, only return windows where drift fired.

        Returns:
            List of :class:`~rag.models.DriftResult` objects ordered by
            ascending ``timestamp``.
        """
        query = (
            "SELECT score, pvalue, drifted, window_size, snapshot_size, timestamp"
            " FROM drift_history"
        )
        params: list[object] = []

        if drifted_only:
            query += " WHERE drifted = 1"

        query += " ORDER BY timestamp ASC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        results: list[DriftResult] = []
        for score, pvalue, drifted, window_size, snapshot_size, ts in rows:
            results.append(
                DriftResult(
                    statistic=float(score),
                    pvalue=float(pvalue),
                    drifted=bool(drifted),
                    window_size=int(window_size),
                    snapshot_size=int(snapshot_size),
                    evaluated_at=datetime.fromisoformat(ts),
                )
            )
        return results

    def export_to_wandb(self) -> None:
        """Log all stored drift windows to W&B as individual ``drift_history`` events.

        No-ops silently when W&B is disabled or no active run exists (delegated
        to :func:`~rag.tracking.log_event`).
        """
        rows = self._conn.execute(
            """
            SELECT window_id, score, pvalue, drifted, window_size,
                   snapshot_size, timestamp, triggered_reindex
            FROM drift_history
            ORDER BY timestamp ASC
            """
        ).fetchall()

        for wid, score, pvalue, drifted, window_size, snapshot_size, ts, triggered in rows:
            log_event(
                "drift_history",
                {
                    "window_id": wid,
                    "score": float(score),
                    "pvalue": float(pvalue),
                    "drifted": int(drifted),
                    "window_size": int(window_size),
                    "snapshot_size": int(snapshot_size),
                    "timestamp": ts,
                    "triggered_reindex": int(triggered),
                },
            )

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
