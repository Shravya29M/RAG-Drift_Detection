"""Unit tests for rag.cli — one test per command, HTTP calls mocked."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from rag.cli import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status_code: int, body: object) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = body
    resp.text = json.dumps(body)
    resp.raise_for_status.return_value = None
    return resp


def _http_error(status_code: int, body: str = "error") -> httpx.HTTPStatusError:
    req = MagicMock(spec=httpx.Request)
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = body
    return httpx.HTTPStatusError(body, request=req, response=resp)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


class TestIngestCommand:
    def test_happy_path_prints_job_id(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("hello")
        mock_resp = _mock_response(200, {"job_id": "abc-123", "message": "ingestion queued"})

        with patch("rag.cli.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            rc = main(["ingest", str(doc)])

        assert rc == 0
        out = capsys.readouterr().out
        assert "abc-123" in out

    def test_missing_file_returns_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["ingest", str(tmp_path / "nonexistent.txt")])
        assert rc == 1
        assert "not found" in capsys.readouterr().err

    def test_http_error_returns_1(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        doc = tmp_path / "f.txt"
        doc.write_text("x")
        with patch("rag.cli.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.side_effect = _http_error(500)
            rc = main(["ingest", str(doc)])
        assert rc == 1


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


class TestQueryCommand:
    def test_happy_path_prints_answer(self, capsys: pytest.CaptureFixture[str]) -> None:
        body = {
            "answer": "42 is the answer.",
            "chunks": [
                {
                    "id": "c0",
                    "text": "some chunk",
                    "token_count": 2,
                    "metadata": {"source": "doc.txt", "source_type": "text", "chunk_index": 0},
                }
            ],
            "scores": [0.91],
            "latency_ms": 12.5,
        }
        mock_resp = _mock_response(200, body)

        with patch("rag.cli.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            rc = main(["query", "What is the answer?"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "42 is the answer." in out
        assert "doc.txt" in out

    def test_http_error_returns_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("rag.cli.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.side_effect = _http_error(422)
            rc = main(["query", "hello"])
        assert rc == 1

    def test_k_flag_is_forwarded(self, capsys: pytest.CaptureFixture[str]) -> None:
        body = {"answer": "ok", "chunks": [], "scores": [], "latency_ms": 1.0}
        mock_resp = _mock_response(200, body)

        with patch("rag.cli.httpx.Client") as mock_cls:
            mock_post = mock_cls.return_value.__enter__.return_value.post
            mock_post.return_value = mock_resp
            main(["query", "-k", "10", "search term"])

        call_kwargs = mock_post.call_args
        sent_payload = call_kwargs[1]["json"]
        assert sent_payload["k"] == 10


# ---------------------------------------------------------------------------
# drift-status
# ---------------------------------------------------------------------------


class TestDriftStatusCommand:
    def test_happy_path_prints_state(self, capsys: pytest.CaptureFixture[str]) -> None:
        body = {
            "history": [
                {
                    "statistic": 0.35,
                    "pvalue": 0.03,
                    "drifted": True,
                    "window_size": 50,
                    "snapshot_size": 200,
                    "evaluated_at": "2026-01-01T00:00:00",
                }
            ],
            "consecutive_alerts": 2,
            "reindex_triggered": False,
            "buffer_size": 10,
        }
        mock_resp = _mock_response(200, body)

        with patch("rag.cli.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.get.return_value = mock_resp
            rc = main(["drift-status"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "consecutive_alerts" in out
        assert "2" in out

    def test_http_error_returns_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("rag.cli.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.get.side_effect = _http_error(503)
            rc = main(["drift-status"])
        assert rc == 1


# ---------------------------------------------------------------------------
# reindex
# ---------------------------------------------------------------------------


class TestReindexCommand:
    def test_happy_path_prints_job_id(self, capsys: pytest.CaptureFixture[str]) -> None:
        body = {
            "job_id": "reindex-xyz",
            "status": "pending",
            "created_at": "2026-01-01T00:00:00",
            "completed_at": None,
            "error": None,
        }
        mock_resp = _mock_response(200, body)

        with patch("rag.cli.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            rc = main(["reindex"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "reindex-xyz" in out
        assert "pending" in out

    def test_http_error_returns_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("rag.cli.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.side_effect = _http_error(500)
            rc = main(["reindex"])
        assert rc == 1

    def test_request_error_returns_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("rag.cli.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.side_effect = httpx.ConnectError(
                "refused"
            )
            rc = main(["reindex"])
        assert rc == 1
        assert "could not reach" in capsys.readouterr().err
