# RAG Drift Detection

Most RAG portfolios stop at initial deployment. This one monitors what happens afterward. The system ingests documents, builds a FAISS vector index, and answers natural-language queries by retrieving relevant chunks and passing them to an LLM. A background drift monitor continuously compares the distribution of live query embeddings against the indexed knowledge using a PCA-projected two-sample KS test. When consecutive windows breach the threshold — hysteresis prevents noise-triggered alerts — the system fires a tiered alarm (log → webhook → callback) and optionally re-indexes automatically, restoring retrieval quality without human intervention.

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

# Copy and fill in API keys
cp .env.example .env   # set OPENAI_API_KEY or ANTHROPIC_API_KEY

# Bring up API + Qdrant
docker-compose up --build -d

# Confirm the API is healthy
curl http://localhost:8000/healthz
# {"status":"ok"}

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

*Placeholders — run `notebooks/walkthrough.ipynb` to populate.*

---

## What I'd do differently

The main thing I'd change is the embedding model loading. Right now `SentenceTransformerEncoder` downloads and initialises the model on construction, which means the first API request after a cold start has multi-second latency and the Docker image has no pre-baked weights — they're fetched at runtime. In a production setting I'd bake the weights into the image layer during the builder stage (`RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('...')"`) and expose a startup readiness probe that only passes once the model is warm. I'd also replace the in-process `queue.Queue` + APScheduler pattern with a proper task queue (Celery + Redis or ARQ) so ingest and drift-check jobs survive process restarts and can be scaled horizontally without losing work. Finally, the SQLite drift history is fine for a single-node demo but would need to be replaced with Postgres if multiple API replicas are running, since WAL mode doesn't help across separate processes on separate hosts.
