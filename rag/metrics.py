"""Prometheus metrics shared across the service (default registry, one per process).

Each replica exposes its own values on ``GET /metrics``; scrape replicas
individually rather than through the load balancer.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Latency buckets tuned for sub-millisecond cache hits up to multi-second misses.
_LATENCY_BUCKETS = (
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
)

CACHE_HITS = Counter("rag_cache_hits", "Query cache lookups served from Redis")
CACHE_MISSES = Counter("rag_cache_misses", "Query cache lookups that fell through to retrieval")
CACHE_BYPASSED = Counter(
    "rag_cache_bypassed", "Queries that skipped the cache because Redis was unavailable"
)
CACHE_OPERATION_SECONDS = Histogram(
    "rag_cache_operation_seconds",
    "Latency of query cache operations against Redis",
    ["op"],
    buckets=_LATENCY_BUCKETS,
)
CACHE_INVALIDATED_KEYS = Counter(
    "rag_cache_invalidated_keys", "Cache entries deleted after an index version change"
)
RETRIEVAL_SECONDS = Histogram(
    "rag_retrieval_latency_seconds",
    "End-to-end retrieval latency (encode + search, or cache hit)",
    ["cache"],
    buckets=_LATENCY_BUCKETS,
)

REDIS_UP = Gauge("rag_redis_up", "1 while Redis is reachable, 0 while degraded to local mode")
REDIS_ERRORS = Counter(
    "rag_redis_errors", "Redis failures that forced a local fallback", ["component"]
)

DRIFT_WINDOWS_EVALUATED = Counter(
    "rag_drift_windows_evaluated",
    "Drift windows evaluated by this replica",
    ["mode"],
)
INDEX_RELOADS = Counter(
    "rag_index_reloads", "Index reloads triggered by a newer shared index version"
)
