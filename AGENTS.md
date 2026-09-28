# RAG Drift Detection — working context

A RAG pipeline that keeps monitoring itself after deployment. Documents are ingested into FAISS
and queries answered from retrieved chunks; in the background a drift monitor projects query
embeddings onto a PCA basis fitted on the corpus, calibrates a baseline from the first window of
real traffic, and KS-tests each subsequent window against it.

The design decision worth remembering: **drift alone never means the index is stale.** Escalation
is quality-gated. Sustained drift with healthy retrieval scores is a benign topic shift (webhook,
recalibrate baseline, no re-index). Sustained drift *plus* degraded retrieval scores opens a
deduplicated remediation incident, because re-embedding unchanged chunks cannot add missing
knowledge — the fix is ingesting documents that cover the new demand.

With `REDIS_URL` set, several replicas run as one service: Redis holds the query cache, the
drift windows + baseline + hysteresis state + history, the remediation incidents, and the
current index version. Without it (or while it is down) each process runs standalone.

## Layout

```
rag/
  api.py            FastAPI app: /ingest /query /drift /remediations /reindex /index/status /metrics
  settings.py       config/default.yaml + .env, loaded once at startup
  models.py         every pydantic config and response model
  metrics.py        prometheus_client metrics (cache, Redis health, drift windows, reloads)
  redis_health.py   shared client + up/down tracking; the single place fallback is decided
  cache/            query_cache.py (embedding + top-k, keyed by index version)
  jobs.py           Redis job records so GET /jobs/{id} answers on every replica
  index_sync.py     index version key, pub/sub + poll reload, cross-replica writer lease
  remediation.py    dedupe/cooldown rules + Redis incident store (WATCH/MULTI)
  drift/            snapshot.py (PCA basis) detector.py (KS + hysteresis, pure close_window)
                    redis_state.py (shared windows via Lua, fenced commits, local fallback)
                    alarm.py (SOFT/HARD/AUTO) scheduler.py (APScheduler + queue)
  retrieval/        retriever.py (cache lookup happens here)
  embedding/        encoder.py (sentence-transformers)
  vector_store/     faiss_store.py, base.py
  ingestion/        parsers.py chunker.py (sized in embedding-model tokens) metadata.py
tests/unit/         per-layer isolation tests
tests/integration/  FastAPI TestClient against mock encoder + store; test_api_redis.py runs two
                    in-process replicas on one fakeredis; test_redis_live.py needs REDIS_URL
tests/eval/         reproducible retrieval smoke eval
benchmarks/         drift_detection_benchmark.py + results.json
                    cache_and_replica_benchmark.py + cache_replica_results.json
deploy/nginx.conf   round-robin LB for the 3-replica docker-compose stack
frontend/           Next.js UI (has its own AGENTS.md)
```

## Commands

```bash
pip install -r requirements.txt && pip install ruff mypy pytest pytest-cov "fakeredis[lua]"
pytest                                   # coverage on by default, fails under 95%
REDIS_URL=redis://localhost:6379/0 pytest tests/integration/test_redis_live.py -o addopts=""
                                         # flushes logical DB 15 — use a disposable Redis
docker compose up --build -d --wait      # redis + api1..3 + nginx; LB :8000, replicas :8001-8003
benchmarks/run_cache_replica_benchmark.sh  # corpus download, ingest, http/drift/inprocess runs
ruff check . && ruff format --check .
mypy --strict rag/
OMP_NUM_THREADS=1 .venv/bin/python -m tests.eval.run_sample_corpus
python benchmarks/drift_detection_benchmark.py
```

## Verified metrics

Measured 2026-09-21 unless noted. Re-derive with the command shown; do not estimate.

| Metric | Value | How to re-derive |
|---|---|---|
| Test count | 532 (5 need `REDIS_URL`, skipped without it) — measured 2026-09-27 | `pytest --collect-only -q -o addopts=""` |
| Branch + line coverage | 98.22% with live Redis (2104 statements, 28 missed; 424 branches, 15 partial) — 2026-09-27 | `REDIS_URL=… pytest` |
| Coverage floor enforced in CI | 95% | `pyproject.toml`, `[tool.pytest.ini_options]` |
| Type checking | `mypy --strict rag/` clean, 36 source files | `mypy --strict rag/` |
| Lint / format | `ruff check` and `ruff format --check` both clean | as above |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | `config/default.yaml:9`, `rag/models.py:344` |
| Embedding dimension | 384 | not hardcoded; derived from the persisted index (below) or `encoder.dim` at runtime |
| Chunks in the persisted sample index | **8**, from 2 source documents (unchanged by token chunking: every section ≤108 tokens) | `index/faiss.index` is a **zip bundle**, not a raw FAISS file — open with `zipfile`, read `vectors.npy` → shape `(8, 384)` |

> The 8-chunk index is the bundled two-document demo corpus
> (`samples/embedding_drift.md`, `samples/retrieval_augmented_generation.md`). It is not a
> production corpus size. Do not cite it as one.

### Retrieval smoke eval (run 2026-07-12)

16 manually labelled queries over the two bundled documents, source-level labels.

| Metric | Result |
|---|---|
| Hit Rate@1 | 1.00 (16/16) |
| Hit Rate@5 | 1.00 (16/16) |
| MRR@5 | 1.00 |
| Baseline mean top-k score | 0.248 |
| Covered-topic mean top-k score | 0.345 → benign recalibration |
| Off-topic mean top-k score | 0.046 → AUTO path |

Smoke test only: two documents, author-written labels, no LLM faithfulness eval.
Re-run 2026-09-27 after the switch to token-sized chunks: identical results, as expected —
all 8 sections of the two documents are ≤108 tokens, so both chunkers give one chunk each.

### Drift detector benchmark (run 2026-07-15)

20 trials over SQuAD articles across 6 topics, 10 windows (500 queries) of in-distribution
traffic then an abrupt shift to 6 unrelated topics. Default config throughout: window 50,
KS at alpha=0.05 Bonferroni-corrected, 32 PCA dims, 3-window hysteresis.

| Metric | Result |
|---|---|
| Topic-shift detection rate | 20/20 trials |
| Median time to first detection | 1 window (50 queries) |
| Hysteresis alarm within 6-window budget | 20/20 trials (median 3 windows) |
| Per-window false positives on clean traffic | 9/200 windows (4.5%) |
| False **escalations** after hysteresis | **0/200 windows** |

That last pair is the before/after for the hysteresis logic: the KS test fires at its configured
alpha (4.5% ≈ 0.05, so it is calibrated), and hysteresis absorbs every one of those single-window
blips. Data: `benchmarks/drift_detection_results.json`.

Not yet benchmarked: benign-shift traffic and gradual-decay traffic (covered in unit tests only).

### Query cache + 3 replicas (run 2026-09-27, token-sized chunks)

Corpus: Python 3.14 docs (library/howto/tutorial/reference), 380 files, ingested through the load
balancer into **13,698 chunks** of ≤200 model tokens (max 202 incl. [CLS]/[SEP], 0 over 256;
same version on all 3 replicas, 195.5 s from upload to converged, embedding included). The
ingest job read `done` with an identical record on all 3 replicas and 9/9 via the LB.
Query mix: 3,000 requests over 500 heading-derived queries, Zipf 1.0, 20% case/whitespace
variants → 412 unique after normalisation, so the mix's ceiling is 0.8627. Keyless generation;
macOS arm64, Docker Desktop, Redis 7.4.11. Data: `benchmarks/cache_replica_results.json`;
rerun: `benchmarks/run_cache_replica_benchmark.sh`.

| Metric | Cache off | Cache on |
|---|---|---|
| Hit rate (HTTP, summed replica counters) | — | **0.8627** (2,588 / 3,000) |
| `/query` via nginx, p50 / p95 | 10.92 / 34.70 ms | **1.26 / 30.06 ms** |
| `Retriever.retrieve` in-process, p50 / p95 (2 repeats) | 6.77 / 11.77 ms; 6.21 / 11.11 ms | **0.39 / 13.99 ms; 0.35 / 13.03 ms** |
| In-process hits only, p50 / p95 | — | 0.37 / 0.68 ms |
| In-process misses only, p50 / p95 | — | 13.31 / 17.27 ms |

p50 improves ~9–18×; **p95 does not improve much** (HTTP −4.6 ms; in-process ~2 ms *worse*):
13.7% of requests miss, so p95 sits in the misses — the long tail of first-seen queries plus a
Redis GET + SET. An earlier run on 512-word chunks (2,470 chunks) gave the same hit rate and a
similar shape; search is still a small share of latency next to query encoding.

Drift across replicas: 5 trials, each 1 calibration + 2 clean + 4 off-topic (SQuAD) windows of
50 sent round-robin through nginx. **5/5** trials: identical `/drift`, `/remediations` and job
record on all 3 replicas, drift on the first shifted window, AUTO escalation, one incident
deduplicated across trials. Clean windows: 1/10 drifted (single window, absorbed by hysteresis).
Every replica evaluated windows in every trial.
Redis stopped under traffic: 60/60 `/query` returned 200, `rag_redis_up` 0 on all replicas;
back to 1 after restart.

## Known UNKNOWNs

- **Query latency with a real LLM: not measured.** The cache benchmark above runs keyless
  (extractive) generation; LLM time is excluded from every latency number in this file.
  Latency under concurrent load is also unmeasured (the benchmark client is sequential).

## Deliberately uncovered

`api.py:101-138`, the FastAPI lifespan startup, constructs a real `SentenceTransformerEncoder`;
covering it means downloading model weights in CI. Everything else in `api.py` is exercised
through `TestClient` against the mock encoder/store harness in `tests/integration/test_api.py`
(import `_make_state`, `_make_chunk`, `DIM` from there rather than rebuilding it).

## Pinned dependency: `anthropic < 1`

`requirements.txt` pins `anthropic>=0.50,<1`. The 1.x SDK removed `temperature` from the
`messages.create` overloads — Anthropic dropped sampling parameters (`temperature`, `top_p`,
`top_k`) on current models, where they now return a 400. `AnthropicRouter.complete`
(`rag/generation/llm.py:94`) passes `temperature=self._config.temperature`, so it fails
`mypy --strict` against 1.x.

The pin holds behavior exactly as-is. Migrating to 1.x is a real decision, not a mechanical
fix, because it changes generation behavior:

- The configured default model is `claude-opus-4-6` (`config/default.yaml:28`), which still
  *accepts* `temperature` at the API level — but the 1.x SDK no longer types it, so you would
  need `extra_body` (ugly) or to drop the parameter.
- `GenerationConfig.temperature` is also consumed by the OpenAI and Groq routers, where it
  remains valid. Dropping it only from the Anthropic path leaves a confusing half-state.
- 1.x is also a broader migration: it moves from `httpx` to `httpx2`.

Renovate will raise the 1.x bump as a standalone `major-update` PR with a 7-day soak. Decide
the `temperature` question there rather than letting the bump land silently.

## Known debt

- **Two drift-history stores.** `DriftStore` (`rag/persistence.py`, SQLite) is still not wired
  into the API, and shared drift history now lives in Redis (`rag:drift:<version>:history`,
  capped at 1000 windows). Pick one before adding history features; do not wire SQLite in as a
  second writer.

## Known limitation: replicas diverge while Redis is down (no merge-back, by design)

Redis is the only thing that makes the replicas one service. While it is unreachable each
replica degrades independently and keeps serving (`rag_redis_up 0`, one warning per transition):

| State | During the outage | When Redis returns |
|---|---|---|
| Query cache | bypassed (`rag_cache_bypassed_total`) | resumes; nothing to reconcile |
| Drift windows, baseline, hysteresis, history | each replica's own local detector | shared state resumes where it stopped; the local partial window is **discarded**; local windows evaluated during the outage are **not** merged into shared history |
| Remediation incidents | opened in that replica's memory | stay local-only and invisible to other replicas; **not** copied to Redis |
| Jobs | recorded locally only | `GET /jobs/{id}` for an outage-era job 404s on other replicas |
| Index version sync | no pub/sub or polling; ingest writes the file without the lease | polling catches every replica up to the shared version |

Consequences: `/drift` and `/remediations` can differ between replicas for the length of the
outage, and an AUTO escalation during it is only visible on the replica that raised it. Merge-back
is deliberately not built — reconciling KS windows or incident dedupe after the fact has no
single right answer. If that matters, alert on `rag_redis_up == 0` and treat the outage window
as unmonitored.

## Gotchas

- `_start_drift_monitor` needs a snapshot of **at least 2 embeddings**. A store with one chunk
  silently leaves the monitor unstarted and `/drift/simulate` returns 409.
- `/drift/simulate` clamps `windows` to `[1, 10]`.
- `POST /remediations/{id}/resolve` requires a JSON body with a non-empty `resolution`; an empty
  string is a 422, an already-resolved incident is a 409.
- `RemediationIncident` requires `opened_at` and `updated_at`; there is no `reason` field.
- Env overrides beat YAML: `QDRANT_URL` → `vector_store.qdrant_url`,
  `DRIFT_WEBHOOK_URL` → `alarm.webhook_url`, `REDIS_URL` → `redis.url`,
  `QUERY_CACHE_ENABLED` → `cache.enabled`. An exported-but-empty variable does **not** blank a
  configured value.
- **Index version = sha256(model name, snapshot ID, sorted chunk IDs)** — not the vectors. The
  FAISS store mints a new snapshot ID on every add/delete/swap and persists it in the bundle, so
  every mutation invalidates the cache and replicas loading the same file agree on the version.
  Bundles written before this change load with snapshot ID `legacy`.
- **Every index mutation must go through `_index_writer` + `_commit_index_change`** (api.py):
  the lease stops two replicas writing the shared file at once, the commit publishes the version,
  purges old cache keys and restarts drift calibration. Encoding happens outside the lease.
- Drift state is namespaced by index version, so an ingest on any replica starts a fresh
  calibration everywhere; restarting one replica rejoins the existing shared state.
- The query cache stores the query embedding too: cache hits still feed the drift monitor.
- `/query` no longer encodes twice — the drift monitor reuses `RetrievalResult.query_embedding`.
- The repo `.env` sets `GROQ_API_KEY`; `docker compose` loads it, so traffic through the stack
  calls Groq. `benchmarks/compose.bench.yml` blanks every LLM key for that reason.
- **Chunks are sized in embedding-model tokens** (default 200, overlap 30), packed from whole
  words using the encoder's own tokenizer, and `chunk_text` rejects a `chunk_size` above the
  model's limit (`max_seq_length` − [CLS]/[SEP] = 254). Use `encoder.max_input_tokens`, never
  `tokenizer.model_max_length` — for all-MiniLM-L6-v2 that says 512, the model truncates at 256.
  Before 2026-09-27 chunks were 512 *whitespace words*; one such docs chunk was 1,580 word
  pieces, ~84% never embedded. `tests/unit/test_chunker_tokens.py` fails if any chunk exceeds
  the model's max length (downloads the tokenizer + `sentence_bert_config.json`, not weights).
- WordPiece turns any word over 100 characters into a single `[UNK]` token.
- Encoders without a `tokenizer` (test doubles) fall back to whitespace-word chunking.
- An **empty** `SEED_SAMPLE_DATA` means the default (seed); only a non-`true` value disables it.

## Dependencies

Renovate (`renovate.json`): weekly Monday. The ML/vector stack (torch, faiss, sentence-transformers,
numpy, scipy, scikit-learn) is excluded from automerge, gets its own PR each, and is labelled
`needs-eval-rerun` — those bumps can move retrieval quality without failing a test, so rerun the
eval and the drift benchmark before merging. Requires the Renovate GitHub App on the repo.
