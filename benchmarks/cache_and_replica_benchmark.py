"""Query-cache latency/hit-rate and multi-replica drift-consistency benchmark.

Subcommands (run in this order; see benchmarks/run_cache_replica_benchmark.sh):

  ingest     POST the Python-docs corpus to the load balancer, then wait until
             every replica reports the same index version and chunk count.
  http       Replay the query mix through the load balancer; per-request
             client latency, plus hit/miss counts from each replica's /metrics.
  drift      Inject a topic shift through the load balancer, N trials; compare
             /drift and /remediations across all replicas.
  inprocess  Same query mix against Retriever directly (real encoder, real
             FAISS index from the ingest step, real Redis), cache off vs on.

Every subcommand merges its section into benchmarks/cache_replica_results.json.
No number in that file is estimated: each is computed from the requests made.

Query mix: ``--distinct`` base queries drawn from the on-topic pool, requested
``--requests`` times with Zipf(``--zipf``) popularity; a ``--variant-p``
fraction of requests are case/whitespace variants of their base query. The
achievable hit rate is a property of this mix, reported as ``ideal_hit_rate``
(1 - unique normalised queries / requests) next to the measured one.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")  # faiss + torch share OpenMP badly on macOS
os.environ.setdefault("WANDB_DISABLED", "true")

import httpx  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.cache.query_cache import KEY_PREFIX, normalize_query  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
RESULTS = HERE / "cache_replica_results.json"
REPLICAS = ("http://localhost:8001", "http://localhost:8002", "http://localhost:8003")
LB = "http://localhost:8000"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _save(section: str, data: dict[str, Any]) -> None:
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results[section] = data
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({section: data}, indent=2))


def _queries() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((CORPUS / "queries.json").read_text())
    return data


def _variant(q: str, rng: np.random.Generator) -> str:
    kind = int(rng.integers(0, 3))
    if kind == 0:
        return q.upper()
    if kind == 1:
        return q.title()
    return "  " + q.replace(" ", "   ") + " "


def query_mix(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    """Deterministic Zipf-distributed query sequence with repeats and variants."""
    pool = _queries()["on_topic"]
    rng = np.random.default_rng(args.seed)
    base = list(rng.choice(pool, size=args.distinct, replace=False))
    weights = 1.0 / np.arange(1, args.distinct + 1) ** args.zipf
    idx = rng.choice(args.distinct, size=args.requests, p=weights / weights.sum())
    seq = [_variant(base[i], rng) if rng.random() < args.variant_p else base[i] for i in idx]
    unique = len({normalize_query(q) for q in seq})
    meta = {
        "requests": args.requests,
        "distinct_base_queries": args.distinct,
        "zipf_exponent": args.zipf,
        "variant_fraction": args.variant_p,
        "seed": args.seed,
        "unique_after_normalisation": unique,
        "ideal_hit_rate": round(1 - unique / args.requests, 4),
    }
    return seq, meta


def _summary(ms: list[float]) -> dict[str, float | int]:
    arr = np.asarray(ms)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "mean_ms": round(float(arr.mean()), 3),
    }


_METRIC = re.compile(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-+\deE.naNinf]+)$")


def _metrics(base: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in httpx.get(f"{base}/metrics", timeout=10).text.splitlines():
        m = _METRIC.match(line)
        if m and not line.startswith("#"):
            out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
    return out


def _flush_cache(redis_url: str) -> int:
    import redis

    client = redis.Redis.from_url(redis_url)
    keys = list(client.scan_iter(match=f"{KEY_PREFIX}:*", count=1000))
    for i in range(0, len(keys), 500):
        client.unlink(*keys[i : i + 500])
    return len(keys)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> None:
    files = sorted((CORPUS / "docs").rglob("*.txt"))
    payload = [
        ("files", (str(p.relative_to(CORPUS / "docs")), p.read_bytes(), "text/plain"))
        for p in files
    ]
    t0 = time.monotonic()
    r = httpx.post(f"{LB}/ingest", files=payload, timeout=300)
    r.raise_for_status()
    ingested_by = r.headers.get("x-upstream")
    job_id = r.json()["job_id"]
    print(f"ingest job {job_id} on {ingested_by}; waiting for all replicas…", file=sys.stderr)

    statuses: list[dict[str, Any]] = []
    while time.monotonic() - t0 < args.timeout:
        statuses = [httpx.get(f"{b}/index/status", timeout=10).json() for b in REPLICAS]
        versions = {s["index_version"] for s in statuses}
        counts = {s["chunk_count"] for s in statuses}
        if (
            len(versions) == 1
            and statuses[0]["shared_version"] in versions
            and len(counts) == 1
            and counts.pop() > 0
        ):
            break
        time.sleep(2)
    else:
        raise SystemExit(f"replicas did not converge: {statuses}")
    converged_s = round(time.monotonic() - t0, 1)

    # The job ran on one replica; every replica and the LB must report it.
    jobs: list[dict[str, Any]] = []
    while time.monotonic() - t0 < args.timeout:
        jobs = [httpx.get(f"{b}/jobs/{job_id}", timeout=10).json() for b in REPLICAS]
        if all(j.get("status") in ("done", "error") for j in jobs):
            break
        time.sleep(1)
    lb_codes = [httpx.get(f"{LB}/jobs/{job_id}", timeout=10).status_code for _ in range(9)]
    _save(
        "ingest",
        {
            "job_id": job_id,
            "job_status_per_replica": {
                b: j.get("status") for b, j in zip(REPLICAS, jobs, strict=True)
            },
            "job_record_identical_on_all_replicas": all(j == jobs[0] for j in jobs),
            "job_lookups_via_lb_status_codes": lb_codes,
            **_chunk_token_stats(),
            "files_uploaded": len(files),
            "bytes_uploaded": sum(len(p[1][1]) for p in payload),
            "ingested_by": ingested_by,
            "chunk_count_per_replica": {
                b: s["chunk_count"] for b, s in zip(REPLICAS, statuses, strict=True)
            },
            "index_version_per_replica": {
                b: s["index_version"] for b, s in zip(REPLICAS, statuses, strict=True)
            },
            "seconds_until_all_replicas_converged": converged_s,
            "note": "includes parsing, chunking and embedding on the ingesting replica",
        },
    )


def _chunk_token_stats() -> dict[str, Any]:
    """Re-tokenize every persisted chunk exactly as the model encodes it."""
    import json as _json

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    with np.load(CORPUS / "index" / "faiss.index", allow_pickle=False) as data:
        chunks = _json.loads(str(data["chunks_json"]))
    lengths = [len(tok(c["text"], verbose=False)["input_ids"]) for c in chunks]
    return {
        "chunks_in_persisted_index": len(chunks),
        "chunk_tokens_with_specials_max": max(lengths),
        "chunk_tokens_with_specials_p50": int(np.percentile(lengths, 50)),
        "chunks_over_model_max_seq_length_256": sum(n > 256 for n in lengths),
    }


# ---------------------------------------------------------------------------
# http
# ---------------------------------------------------------------------------


def cmd_http(args: argparse.Namespace) -> None:
    seq, mix = query_mix(args)
    flushed = _flush_cache(args.redis_url) if args.mode == "on" else 0
    latencies: list[float] = []
    upstreams: dict[str, int] = {}
    with httpx.Client(base_url=LB, timeout=60) as client:
        for w in range(args.warmup):
            client.post("/query", json={"query": f"warmup request {w} zzqx"})
        before = {b: _metrics(b) for b in REPLICAS}  # after warmup: counts cover `seq` only
        for q in seq:
            t0 = time.perf_counter()
            r = client.post("/query", json={"query": q})
            latencies.append((time.perf_counter() - t0) * 1000)
            r.raise_for_status()
            up = r.headers.get("x-upstream", "?")
            upstreams[up] = upstreams.get(up, 0) + 1
    after = {b: _metrics(b) for b in REPLICAS}

    def delta(name: str) -> dict[str, float]:
        return {b: after[b].get(name, 0.0) - before[b].get(name, 0.0) for b in REPLICAS}

    hits, misses = delta("rag_cache_hits_total"), delta("rag_cache_misses_total")
    bypassed = delta("rag_cache_bypassed_total")
    total_h, total_m = sum(hits.values()), sum(misses.values())
    lookups = total_h + total_m
    if args.mode == "off" and lookups:
        raise SystemExit("cache lookups recorded with --mode off: stack still has cache enabled")
    _save(
        f"http_cache_{args.mode}",
        {
            "query_mix": mix,
            "warmup_requests_excluded": args.warmup,
            "cache_keys_flushed_before_run": flushed,
            "client_latency": _summary(latencies),
            "requests_per_replica": upstreams,
            "cache_hits_per_replica": hits,
            "cache_misses_per_replica": misses,
            "cache_bypassed_total": sum(bypassed.values()),
            "measured_hit_rate": round(total_h / lookups, 4) if lookups else None,
            "note": "sequential client, keyless (extractive) generation; end-to-end /query",
        },
    )


# ---------------------------------------------------------------------------
# drift
# ---------------------------------------------------------------------------


def _wait_drained(expect_history: int | None, timeout: float) -> list[dict[str, Any]]:
    """Wait until every replica's local queue is empty and, if given, the shared
    history has *expect_history* windows (``None``: any length, but identical)."""
    t0 = time.monotonic()
    drifts: list[dict[str, Any]] = []
    while time.monotonic() - t0 < timeout:
        queues = [_metrics(b).get("rag_scheduler_queue_size", -1) for b in REPLICAS]
        drifts = [httpx.get(f"{b}/drift", timeout=10).json() for b in REPLICAS]
        lengths = {len(d["history"]) for d in drifts}
        if (
            all(q == 0 for q in queues)
            and len(lengths) == 1
            and (expect_history is None or lengths == {expect_history})
        ):
            return drifts
        time.sleep(1)
    raise SystemExit(f"drift state did not settle (expected {expect_history} windows): {drifts}")


def cmd_drift(args: argparse.Namespace) -> None:
    q = _queries()
    job_id = json.loads(RESULTS.read_text())["ingest"]["job_id"]  # from the ingest step
    rng = np.random.default_rng(args.seed + 1)
    window = args.window
    trials = []
    with httpx.Client(base_url=LB, timeout=60) as client:

        def send(texts: list[str]) -> dict[str, int]:
            ups: dict[str, int] = {}
            for text in texts:
                r = client.post("/query", json={"query": str(text)})
                r.raise_for_status()
                up = r.headers.get("x-upstream", "?")
                ups[up] = ups.get(up, 0) + 1
            return ups

        for trial in range(args.trials):
            _wait_drained(None, args.timeout)  # let earlier traffic finish first
            client.post("/drift/reset").raise_for_status()
            evaluated_before = {
                b: _metrics(b).get('rag_drift_windows_evaluated_total{mode="shared"}', 0.0)
                for b in REPLICAS
            }
            t0 = time.monotonic()
            on = list(rng.choice(q["on_topic"], size=window * (1 + args.clean), replace=False))
            off = list(rng.choice(q["off_topic"], size=window * args.shifted, replace=False))
            ups = send(on[:window])  # calibration window
            _wait_drained(0, args.timeout)
            ups2 = send(on[window:])  # clean in-distribution windows
            _wait_drained(args.clean, args.timeout)
            ups3 = send(off)  # injected shift
            drifts = _wait_drained(args.clean + args.shifted, args.timeout)
            remediations = [httpx.get(f"{b}/remediations").json() for b in REPLICAS]
            jobs = [httpx.get(f"{b}/jobs/{job_id}").json() for b in REPLICAS]
            evaluated = {
                b: _metrics(b).get('rag_drift_windows_evaluated_total{mode="shared"}', 0.0)
                - evaluated_before[b]
                for b in REPLICAS
            }
            hist = drifts[0]["history"]
            shift_hist = hist[args.clean :]
            drifted = [h["drifted"] for h in shift_hist]
            first = next((i + 1 for i, d in enumerate(drifted) if d), None)
            requests_per_replica: dict[str, int] = {}
            for part in (ups, ups2, ups3):
                for k, v in part.items():
                    requests_per_replica[k] = requests_per_replica.get(k, 0) + v
            trials.append(
                {
                    "trial": trial + 1,
                    "identical_drift_state_on_all_replicas": all(d == drifts[0] for d in drifts),
                    "identical_remediations_on_all_replicas": all(
                        r == remediations[0] for r in remediations
                    ),
                    "identical_job_record_on_all_replicas": all(j == jobs[0] for j in jobs)
                    and jobs[0].get("status") == "done",
                    "clean_windows_drifted": [h["drifted"] for h in hist[: args.clean]],
                    "shift_windows_drifted": drifted,
                    "shift_windows_quality_degraded": [h["quality_degraded"] for h in shift_hist],
                    "baseline_mean_score": drifts[0]["baseline_mean_score"],
                    "shift_mean_scores": [round(h["mean_top_score"], 4) for h in shift_hist],
                    "first_drifted_window_after_shift": first,
                    "reindex_triggered": drifts[0]["reindex_triggered"],
                    "open_remediations": sum(r["status"] == "open" for r in remediations[0]),
                    "remediation_occurrences": [r["occurrences"] for r in remediations[0]],
                    "windows_evaluated_by_replica": evaluated,
                    "requests_per_replica": requests_per_replica,
                    "seconds": round(time.monotonic() - t0, 1),
                }
            )
            print(json.dumps(trials[-1]), file=sys.stderr)

    _save(
        "drift_replicas",
        {
            "replicas": len(REPLICAS),
            "trials": args.trials,
            "window_size": window,
            "clean_windows": args.clean,
            "shifted_windows": args.shifted,
            "trials_detected": sum(
                t["first_drifted_window_after_shift"] is not None for t in trials
            ),
            "trials_escalated": sum(t["reindex_triggered"] for t in trials),
            "trials_identical_on_all_replicas": sum(
                t["identical_drift_state_on_all_replicas"]
                and t["identical_remediations_on_all_replicas"]
                and t["identical_job_record_on_all_replicas"]
                for t in trials
            ),
            "median_first_drifted_window": statistics.median(
                [t["first_drifted_window_after_shift"] or 0 for t in trials]
            ),
            "per_trial": trials,
        },
    )


# ---------------------------------------------------------------------------
# outage
# ---------------------------------------------------------------------------


def _compose(*args: str) -> None:
    import subprocess

    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.yml",
            "-f",
            "benchmarks/compose.bench.yml",
            *args,
        ],
        check=True,
        cwd=HERE.parent,
        capture_output=True,
    )


def cmd_outage(args: argparse.Namespace) -> None:
    """Stop Redis under live traffic, then bring it back; count failures."""
    pool = _queries()["on_topic"][: args.requests]

    def burst() -> dict[str, Any]:
        codes: dict[int, int] = {}
        lat: list[float] = []
        with httpx.Client(base_url=LB, timeout=60) as client:
            for text in pool:
                t0 = time.perf_counter()
                code = client.post("/query", json={"query": text}).status_code
                lat.append((time.perf_counter() - t0) * 1000)
                codes[code] = codes.get(code, 0) + 1
        return {"status_codes": codes, "latency": _summary(lat)}

    def gauges() -> dict[str, dict[str, float]]:
        out = {}
        for b in REPLICAS:
            m = _metrics(b)
            out[b] = {
                "rag_redis_up": m.get("rag_redis_up", -1),
                "cache_errors": m.get('rag_redis_errors_total{component="cache"}', 0.0),
                "cache_bypassed": m.get("rag_cache_bypassed_total", 0.0),
            }
        return out

    _compose("stop", "redis")
    during = burst()
    during_gauges = gauges()
    during_status = [httpx.get(f"{b}/index/status").json()["redis_up"] for b in REPLICAS]
    _compose("start", "redis")
    time.sleep(args.retry_wait)  # one reconnection-probe interval (redis.retry_interval_s)
    after = burst()
    after_gauges = gauges()
    _save(
        "redis_outage",
        {
            "requests_per_phase": len(pool),
            "during_outage": {**during, "replica_gauges": during_gauges, "redis_up": during_status},
            "after_restart": {**after, "replica_gauges": after_gauges},
        },
    )


# ---------------------------------------------------------------------------
# inprocess
# ---------------------------------------------------------------------------


def cmd_inprocess(args: argparse.Namespace) -> None:
    import redis
    import torch

    from rag.cache.query_cache import QueryCache
    from rag.embedding.encoder import SentenceTransformerEncoder
    from rag.index_sync import index_version
    from rag.redis_health import RedisHealth
    from rag.retrieval.retriever import Retriever
    from rag.vector_store.faiss_store import FAISSStore

    seq, mix = query_mix(args)
    encoder = SentenceTransformerEncoder(MODEL)
    store = FAISSStore(dim=encoder.dim)
    if not store.load(Path(args.index)):
        raise SystemExit(f"no index at {args.index}; run the ingest step first")
    client = redis.Redis.from_url(args.redis_url)
    health = RedisHealth(client)

    runs: list[dict[str, Any]] = []
    for rep in range(args.repeats):
        for mode in ("off", "on"):
            if mode == "on":
                _flush_cache(args.redis_url)
                retriever = Retriever(
                    store,
                    encoder,
                    cache=QueryCache(health, ttl_s=3600),
                    index_version=lambda: index_version(store, MODEL),
                )
            else:
                retriever = Retriever(store, encoder)
            for w in range(args.warmup):
                retriever.retrieve(f"warmup request {w} zzqx", k=5)
            lat: list[float] = []
            hit_lat: list[float] = []
            miss_lat: list[float] = []
            hits = 0
            for text in seq:
                t0 = time.perf_counter()
                result = retriever.retrieve(text, k=5)
                ms = (time.perf_counter() - t0) * 1000
                lat.append(ms)
                (hit_lat if result.cache_hit else miss_lat).append(ms)
                hits += result.cache_hit
            runs.append(
                {
                    "repeat": rep + 1,
                    "cache": mode,
                    "latency": _summary(lat),
                    "hit_latency": _summary(hit_lat) if mode == "on" else None,
                    "miss_latency": _summary(miss_lat) if mode == "on" else None,
                    "measured_hit_rate": round(hits / len(seq), 4) if mode == "on" else None,
                }
            )
            print(json.dumps(runs[-1]), file=sys.stderr)

    info = client.info("server")
    _save(
        "inprocess",
        {
            "query_mix": mix,
            "chunks_in_index": len(store.list_chunks()),
            "k": 5,
            "warmup_requests_excluded": args.warmup,
            "runs": runs,
            "environment": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "torch_threads": torch.get_num_threads(),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "redis_version": info.get("redis_version"),
                "redis_url_host": args.redis_url.split("@")[-1],
            },
            "note": (
                "latency = Retriever.retrieve wall time: encode + FAISS search (+ Redis get/put)"
            ),
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def mix_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--requests", type=int, default=3000)
        p.add_argument("--distinct", type=int, default=500)
        p.add_argument("--zipf", type=float, default=1.0)
        p.add_argument("--variant-p", type=float, default=0.2)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--warmup", type=int, default=20)
        p.add_argument("--redis-url", default="redis://localhost:6379/0")

    p = sub.add_parser("ingest")
    p.add_argument("--timeout", type=float, default=1800)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("http")
    mix_args(p)
    p.add_argument("--mode", choices=("on", "off"), required=True)
    p.set_defaults(func=cmd_http)

    p = sub.add_parser("drift")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--clean", type=int, default=2)
    p.add_argument("--shifted", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--timeout", type=float, default=180)
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("outage")
    p.add_argument("--requests", type=int, default=60)
    p.add_argument("--retry-wait", type=float, default=6.0)
    p.set_defaults(func=cmd_outage)

    p = sub.add_parser("inprocess")
    mix_args(p)
    p.add_argument("--index", default=str(CORPUS / "index" / "faiss.index"))
    p.add_argument("--repeats", type=int, default=2)
    p.set_defaults(func=cmd_inprocess)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
