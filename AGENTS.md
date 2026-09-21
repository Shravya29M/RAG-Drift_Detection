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

## Layout

```
rag/
  api.py            FastAPI app: /ingest /query /drift /remediations /reindex /metrics
  settings.py       config/default.yaml + .env, loaded once at startup
  models.py         every pydantic config and response model
  drift/            snapshot.py (PCA basis) detector.py (KS + hysteresis)
                    alarm.py (SOFT/HARD/AUTO) scheduler.py (APScheduler + queue)
  retrieval/        retriever.py
  embedding/        encoder.py (sentence-transformers)
  vector_store/     faiss_store.py, base.py
  ingestion/        parsers.py chunker.py metadata.py
tests/unit/         per-layer isolation tests
tests/integration/  FastAPI TestClient against mock encoder + store
tests/eval/         reproducible retrieval smoke eval
benchmarks/         drift_detection_benchmark.py + results.json
frontend/           Next.js UI (has its own AGENTS.md)
```

## Commands

```bash
pip install -r requirements.txt && pip install ruff mypy pytest pytest-cov
pytest                                   # coverage on by default, fails under 95%
ruff check . && ruff format --check .
mypy --strict rag/
OMP_NUM_THREADS=1 .venv/bin/python -m tests.eval.run_sample_corpus
python benchmarks/drift_detection_benchmark.py
```

## Verified metrics

Measured 2026-09-21 unless noted. Re-derive with the command shown; do not estimate.

| Metric | Value | How to re-derive |
|---|---|---|
| Test count | 395 | `pytest --collect-only -q -o addopts=""` |
| Branch + line coverage | 97.17% (1250 statements, 25 missed; 232 branches, 13 partial) | `pytest` |
| Coverage floor enforced in CI | 95% | `pyproject.toml`, `[tool.pytest.ini_options]` |
| Type checking | `mypy --strict rag/` clean, 28 source files | `mypy --strict rag/` |
| Lint / format | `ruff check` and `ruff format --check` both clean | as above |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | `config/default.yaml:9`, `rag/models.py:344` |
| Embedding dimension | 384 | not hardcoded; derived from the persisted index (below) or `encoder.dim` at runtime |
| Chunks in the persisted sample index | **8**, from 2 source documents | `index/faiss.index` is a **zip bundle**, not a raw FAISS file — open with `zipfile`, read `vectors.npy` → shape `(8, 384)` |

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

## Known UNKNOWNs

- **Query latency (mean or p95): not measured anywhere.** Per-query latency *is* computed at
  `rag/retrieval/retriever.py:118` and returned on `/query` (`rag/api.py:442`), but nothing
  aggregates it — no committed benchmark, eval output, or log has a mean or percentile. To get
  it: extend `tests/eval/run_sample_corpus.py` to collect `RetrievalResult.latency_ms` per query
  and emit percentiles. Do not quote a latency number until that exists.

## Deliberately uncovered

`api.py:101-138`, the FastAPI lifespan startup, constructs a real `SentenceTransformerEncoder`;
covering it means downloading model weights in CI. Everything else in `api.py` is exercised
through `TestClient` against the mock encoder/store harness in `tests/integration/test_api.py`
(import `_make_state`, `_make_chunk`, `DIM` from there rather than rebuilding it).

## Gotchas

- `_start_drift_monitor` needs a snapshot of **at least 2 embeddings**. A store with one chunk
  silently leaves the monitor unstarted and `/drift/simulate` returns 409.
- `/drift/simulate` clamps `windows` to `[1, 10]`.
- `POST /remediations/{id}/resolve` requires a JSON body with a non-empty `resolution`; an empty
  string is a 422, an already-resolved incident is a 409.
- `RemediationIncident` requires `opened_at` and `updated_at`; there is no `reason` field.
- Env overrides beat YAML: `QDRANT_URL` → `vector_store.qdrant_url`,
  `DRIFT_WEBHOOK_URL` → `alarm.webhook_url`. An exported-but-empty variable does **not** blank a
  configured value.

## Dependencies

Renovate (`renovate.json`): weekly Monday. The ML/vector stack (torch, faiss, sentence-transformers,
numpy, scipy, scikit-learn) is excluded from automerge, gets its own PR each, and is labelled
`needs-eval-rerun` — those bumps can move retrieval quality without failing a test, so rerun the
eval and the drift benchmark before merging. Requires the Renovate GitHub App on the repo.
