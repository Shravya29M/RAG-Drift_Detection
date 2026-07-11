# RAG Drift Detection

![CI](https://github.com/Shravya29M/RAG-Drift_Detection/actions/workflows/ci.yml/badge.svg)

 Most RAG portfolio projects stop right after deployment. This one focuses on what happens after it goes live.
 
It starts like a typical pipeline. Documents are ingested, a FAISS vector index is built, and user queries are answered by retrieving the most relevant chunks and passing them to an LLM.
The difference is that it does not assume things will keep working well over time. In the background, a drift monitor keeps track of how incoming queries evolve. It projects query embeddings onto a PCA basis fitted on the indexed corpus, calibrates a baseline from the first window of real query traffic, and compares each subsequent 50-query window against that baseline with a per-dimension two-sample KS test (Bonferroni-corrected p-values).

If the system detects consistent drift across multiple windows, with hysteresis to avoid false alarms, it triggers alerts in stages. It starts with logs and can escalate to webhooks or callbacks if needed. It can also automatically trigger re-indexing, allowing the system to adapt on its own and maintain retrieval quality without constant manual intervention.

---

## Architecture

```
  Documents (PDF / MD / TXT / URL)
          │
          ▼
  ┌───────────────┐       ┌─────────────────────────────────────┐
  │   Ingestion   │       │           Drift Monitor             │
  │  parse →      │       │                                     │
  │  chunk →      │  vecs │  query vecs                         │
  │  embed        │──────▶│  ──────────▶ DriftScheduler         │
  └──────┬────────┘       │                  │ queue            │
         │ chunks+vecs    │                  ▼                  │
         ▼                │           DriftDetector             │
  ┌─────────────┐         │        (rolling window,             │
  │  FAISSStore │         │         PCA + KS test,              │
  │  (A/B swap, │         │         hysteresis)                 │
  │  RLock)     │◀────────│                  │ DriftResult      │
  └──────┬──────┘ reindex │                  ▼                  │
         │                │            DriftAlarm               │
         │ top-k search   │    SOFT ──▶ W&B log                 │
         ▼                │    HARD ──▶ webhook POST            │
  ┌─────────────┐         │    AUTO ──▶ re-index callback       │
  │  Retriever  │         └─────────────────────────────────────┘
  │  (encode +  │
  │   filter)   │              ┌─────────────────┐
  └──────┬──────┘              │  Persistence     │
         │ chunks              │  (SQLite WAL,    │
         ▼                     │  drift_history)  │
  ┌─────────────┐              └─────────────────┘
  │  LLMRouter  │
  │  (OpenAI /  │         ┌──────────────────────────┐
  │  Anthropic) │         │  CLI  (rag ingest/query/  │
  └──────┬──────┘         │       drift-status/       │
         │ answer         │       reindex)             │
         ▼                └──────────────────────────┘
    FastAPI  :8000
```

---

## Quick start

```bash
git clone https://github.com/shravyamunugala/RAG-Drift_Detection.git
cd RAG-Drift_Detection

# Optional: add an LLM API key for generated answers.
# With no key the API runs in keyless mode (returns retrieved context verbatim).
cp .env.example .env

# Bring up the API
docker-compose up --build -d

# Confirm the API is healthy
curl http://localhost:8000/healthz
# {"status":"ok"}

# A fresh instance seeds itself with the sample docs in samples/ so the
# index and drift monitor are live immediately (disable with SEED_SAMPLE_DATA=false).

# Ingest a document
curl -X POST http://localhost:8000/ingest \
  -F "files=@my_doc.pdf"

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is drift detection?", "k": 5}'
```

Without Docker, using the CLI directly:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn rag.api:app --reload &

python -m rag.cli ingest my_doc.pdf
python -m rag.cli query "What is drift detection?"
python -m rag.cli drift-status
```

---

## API reference

| Method | Path | Body / Params | Description |
|--------|------|---------------|-------------|
| `GET` | `/healthz` | — | Liveness probe. Returns `{"status":"ok"}`. |
| `POST` | `/ingest` | `files` (multipart), `urls` (JSON array form field) | Parse, chunk, embed, and index documents. Returns `job_id`. |
| `POST` | `/query` | `{"query": str, "k": int, "filters": obj\|null}` | Retrieve top-k chunks, generate answer. Returns answer + sources + latency. |
| `GET` | `/jobs/{job_id}` | — | Poll status of an async ingest or re-index job. |
| `GET` | `/drift` | — | Current drift state: history, consecutive alert count, buffer size. |
| `POST` | `/drift/simulate` | `windows` (query param, default 3) | Demo: push off-topic query traffic through the monitor to walk the hysteresis counter up and trigger the tiered alarms. |
| `POST` | `/drift/reset` | — | Clear drift history and hysteresis counter. |
| `POST` | `/reindex` | — | Trigger manual re-index; re-embeds all stored chunks and swaps the FAISS index. |
| `GET` | `/metrics` | — | Prometheus-compatible gauge/counter text for queue depth, drift state, job counts. |

---

## Eval results

Evaluated on 50 ground-truth (query, relevant document) pairs from the test corpus. Drift scenarios use a synthetic shifted embedding set (mean-shifted by 2σ) injected after query 100.

| Metric | Baseline | Post-drift | Post-reindex |
|--------|----------|------------|--------------|
| Hit Rate@1 | — | — | — |
| Hit Rate@5 | — | — | — |
| MRR | — | — | — |
| Faithfulness (LLM-as-judge) | — | — | — |
| Drift detection latency (queries) | — | n/a | n/a |
| False positive rate (control set) | — | n/a | n/a |


---

## Watching it work

1. Start the stack (`docker-compose up` or `uvicorn rag.api:app`) — sample docs are ingested and the drift monitor starts automatically.
2. Ask a few questions via `/query` or the web UI; query embeddings stream into the monitor's rolling window.
3. Hit `POST /drift/simulate?windows=3` (or the **simulate drift** button in the UI). Off-topic traffic drifts three consecutive windows, the alarm escalates soft → webhook → auto, and the system re-indexes and recalibrates its baseline. `GET /drift` and `GET /metrics` show every step.

The FAISS index is persisted to `index/` after every write and restored on startup, so restarts lose nothing.

## Deployment

- **API** — any Docker host. The image is torch-sized (~2 GB), so free tiers with <1 GB RAM won't fit; Hugging Face Spaces (Docker space, `app_port: 8000`), Railway, Fly.io, or a small VPS all work. Set `DRIFT_WEBHOOK_URL` to receive escalated alerts; LLM keys are optional (keyless mode degrades gracefully).
- **Frontend** — `frontend/` is a Next.js app; deploy to Vercel and set `NEXT_PUBLIC_API_URL` to the API's public URL.

## What I'd do differently

The main thing I'd change is the embedding model loading. Right now `SentenceTransformerEncoder` downloads and initialises the model on construction, which means the first API request after a cold start has multi-second latency and the Docker image has no pre-baked weights — they're fetched at runtime. In a production setting I'd bake the weights into the image layer during the builder stage (`RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('...')"`) and expose a startup readiness probe that only passes once the model is warm. I'd also replace the in-process `queue.Queue` + APScheduler pattern with a proper task queue (Celery + Redis or ARQ) so ingest and drift-check jobs survive process restarts and can be scaled horizontally without losing work. Finally, the SQLite drift history is fine for a single-node demo but would need to be replaced with Postgres if multiple API replicas are running, since WAL mode doesn't help across separate processes on separate hosts.
