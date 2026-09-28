#!/usr/bin/env bash
# Reproduces benchmarks/cache_replica_results.json end to end.
# Needs Docker and network access (Python docs, SQuAD dev set, model weights).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
B=benchmarks/cache_and_replica_benchmark.py
compose() { INDEX_DIR=./benchmarks/corpus/index docker compose -f docker-compose.yml -f benchmarks/compose.bench.yml "$@"; }

$PY benchmarks/build_docs_corpus.py
rm -f benchmarks/cache_replica_results.json
rm -rf benchmarks/corpus/index && mkdir -p benchmarks/corpus/index

compose down -v --remove-orphans
QUERY_CACHE_ENABLED=true compose up -d --build --wait
$PY $B ingest
$PY $B http --mode on
$PY $B drift
$PY $B outage

QUERY_CACHE_ENABLED=false compose up -d --wait   # recreates replicas without the cache
$PY $B http --mode off
compose down

# In-process run uses a standalone Redis so replicas don't compete for CPU.
docker run -d --rm --name rag-bench-redis -p 6391:6379 redis:7-alpine >/dev/null
trap 'docker stop rag-bench-redis >/dev/null' EXIT
sleep 1
$PY $B inprocess --redis-url redis://localhost:6391/0
