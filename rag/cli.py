"""Command-line interface for the RAG drift pipeline.

Commands
--------
ingest <path>   Upload a file to POST /ingest and print the job ID.
query <text>    Send a query to POST /query and print the answer + sources.
drift-status    Fetch GET /drift and display drift state.
reindex         Trigger POST /reindex and print the job ID.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

_DEFAULT_BASE_URL = "http://localhost:8000"


def _client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=30.0)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    """Upload *args.path* via POST /ingest and print the returned job ID."""
    path = Path(args.path)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    with _client(args.base_url) as client:
        with path.open("rb") as fh:
            resp = client.post(
                "/ingest",
                files={"files": (path.name, fh, "application/octet-stream")},
            )
        resp.raise_for_status()

    body = resp.json()
    print(f"job_id: {body['job_id']}")
    print(f"message: {body['message']}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Send *args.text* to POST /query and print the answer with source chunks."""
    payload: dict[str, object] = {"query": args.text, "k": args.k}

    with _client(args.base_url) as client:
        resp = client.post("/query", json=payload)
        resp.raise_for_status()

    body = resp.json()
    print(body["answer"])
    print()
    for i, (chunk, score) in enumerate(zip(body["chunks"], body["scores"], strict=True), start=1):
        source = chunk.get("metadata", {}).get("source", "?")
        print(f"  [{i}] score={score:.3f}  source={source}")
    print(f"\nlatency: {body['latency_ms']:.1f} ms")
    return 0


def cmd_drift_status(args: argparse.Namespace) -> int:
    """Fetch GET /drift and print the current drift state."""
    with _client(args.base_url) as client:
        resp = client.get("/drift")
        resp.raise_for_status()

    body = resp.json()
    print(f"consecutive_alerts : {body['consecutive_alerts']}")
    print(f"reindex_triggered  : {body['reindex_triggered']}")
    print(f"buffer_size        : {body['buffer_size']}")
    print(f"history_entries    : {len(body['history'])}")
    if body["history"]:
        last = body["history"][-1]
        print(f"last_statistic     : {last['statistic']:.4f}")
        print(f"last_drifted       : {last['drifted']}")
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    """Trigger POST /reindex and print the returned job ID."""
    with _client(args.base_url) as client:
        resp = client.post("/reindex")
        resp.raise_for_status()

    body = resp.json()
    print(f"job_id: {body['job_id']}")
    print(f"status: {body['status']}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="rag",
        description="RAG drift pipeline CLI",
    )
    parser.add_argument(
        "--base-url",
        default=_DEFAULT_BASE_URL,
        help="Base URL of the RAG API server (default: %(default)s).",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a document file.")
    p_ingest.add_argument("path", help="Path to the file to ingest.")
    p_ingest.set_defaults(func=cmd_ingest)

    # query
    p_query = sub.add_parser("query", help="Query the RAG pipeline.")
    p_query.add_argument("text", help="Natural-language query string.")
    p_query.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5).",
    )
    p_query.set_defaults(func=cmd_query)

    # drift-status
    p_drift = sub.add_parser("drift-status", help="Show current drift state.")
    p_drift.set_defaults(func=cmd_drift_status)

    # reindex
    p_reindex = sub.add_parser("reindex", help="Trigger a manual re-index.")
    p_reindex.set_defaults(func=cmd_reindex)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns an exit code.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success, non-zero on error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except httpx.HTTPStatusError as exc:
        print(f"error: HTTP {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        return 1
    except httpx.RequestError as exc:
        print(f"error: could not reach {args.base_url}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
