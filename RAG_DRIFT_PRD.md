# PRD: Real-Time RAG Pipeline with Adaptive Drift Detection

## Project metadata

| Field | Value |
|---|---|
| Version | 1.0 — April 2026 |
| Timeline | ~6 weeks, 6 phases |
| Stack | Python 3.11, FastAPI, FAISS/Qdrant, sentence-transformers, W&B |
| Goal | End-to-end portfolio project demonstrating ML systems engineering |

---

## 1. Overview

Build a production-grade Retrieval-Augmented Generation (RAG) system with an operational drift detection layer. The system ingests documents, indexes them as embeddings in a vector store, answers user queries by retrieving relevant chunks and passing them to an LLM, and continuously monitors whether the distribution of incoming queries is diverging from the indexed knowledge — triggering re-indexing when retrieval quality is at risk of degrading.

**The differentiator:** most RAG portfolios stop at initial deployment. This one monitors what happens to the system over time and responds automatically.

---

## 2. Architecture

Five layers. Each independently testable with a clear interface boundary.

```
Layer           Components                                  Stack
─────────────── ─────────────────────────────────────────── ──────────────────────────────────
Ingestion       File parsers, chunker, metadata extractor   PyMuPDF, LangChain splitters, Pydantic
Embedding       Batch encoder, cache, model registry        sentence-transformers, HuggingFace
Vector Store    Index builder, similarity search, filters   FAISS (local), Qdrant (prod)
Retrieval+Gen   Query encoder, re-ranker, prompt builder    OpenAI/Anthropic API, cross-encoder
Drift Monitor   Distribution tracker, alarm, re-index job  SciPy, W&B, APScheduler
```

### Data flow

```
Offline  │ Documents → Chunker → Embedder → Vector Store → Distribution Snapshot
Online   │ User Query → Query Embedder → Retriever → Re-ranker → LLM → Response
Monitor  │ Query Log → Drift Detector → Alert / Re-index Trigger
```

---

## 3. Scope

### In scope
- Document ingestion: PDF, Markdown, plain text, URL
- Chunking with sliding window (default 512 tokens, 64 overlap), metadata preserved
- Embedding with pluggable model support (sentence-transformers default, OpenAI Ada optional)
- Vector store: FAISS locally, Qdrant for production path
- Retrieval: top-k cosine similarity + optional cross-encoder re-ranking
- Metadata filtering (e.g. "only docs from last 30 days")
- Generation: prompt templating, context injection, streaming via SSE
- Drift detection module (see Section 5)
- Alerting and auto re-index trigger
- Experiment tracking: Weights & Biases for all runs
- REST API: FastAPI backend
- Frontend demo: Streamlit (fast) or Next.js (polished)
- Evaluation harness: retrieval quality + generation quality metrics

### Out of scope (v1)
- Multi-tenant / auth layer
- Fine-tuning embedding or generation models
- Production-scale infra (Kubernetes, managed vector DB)

---

## 4. Functional requirements

### Ingestion
- Accept PDF, Markdown, plain text, URL inputs
- Chunk with sliding window, configurable overlap
- Preserve source metadata: filename, page number, timestamp, section header
- Support incremental ingestion without full re-index

### Retrieval
- Return top-k chunks (default k=5) ranked by cosine similarity
- Cross-encoder re-ranking as optional second pass
- Metadata filtering support
- Retrieval latency P95 < 200ms for index < 100k chunks (local FAISS)

### Generation
- Build prompt from retrieved context + query using configurable template
- Stream responses via SSE
- Log all prompt/response pairs with token counts to W&B
- If no relevant chunk found (score below threshold), return "no information found" — do not hallucinate

### Drift monitor
- Compute drift score after every N queries (configurable, default 50)
- Store drift history in SQLite locally, exportable to W&B
- Expose drift metrics at /metrics (Prometheus-compatible)
- Trigger re-index if drift_score > threshold for 3 consecutive windows (hysteresis to avoid flapping)

---

## 5. Drift detection module

### Detection strategy
1. At index build time, snapshot the distribution of chunk embeddings (mean + covariance, or kernel density estimate).
2. For each incoming query batch (window size N=50), compute aggregate query embedding distribution.
3. Apply Maximum Mean Discrepancy (MMD) or two-sample KS test on PCA-reduced embeddings (32 dims).
4. If test statistic exceeds threshold α, raise alert and optionally trigger re-indexing.

### Alarm levels

| Level | Action |
|---|---|
| Soft alert | Log drift score to W&B, surface in dashboard |
| Hard alert | POST to webhook (Slack/email), pause query serving |
| Auto re-index | Pull fresh documents, rebuild index, swap atomically |
| Fallback | If re-index fails, serve stale index with alert badge |

### Metrics tracked
- `drift_score` — MMD statistic per window
- `retrieval_hit_rate@k` — fraction of queries where ground truth doc is in top-k
- `mean_reciprocal_rank` (MRR)
- `generation_faithfulness` — LLM-graded score via evaluator prompt
- `index_freshness_hours` — time since last re-index

---

## 6. Non-functional requirements

| Requirement | Detail |
|---|---|
| Modularity | Each layer swappable via config YAML. No hard-coded model names in core logic. |
| Reproducibility | All experiments tracked in W&B with config hash. Any run reproducible from one config file. |
| Testability | Unit tests for chunker, embedder interface, drift calculator. Integration test for full query flow. |
| Documentation | README with architecture diagram, setup in <5 commands, annotated Jupyter notebook walkthrough. |
| Observability | Structured JSON logs. W&B dashboard with retrieval, generation, and drift metrics. |
| Deployability | Dockerfile + docker-compose. One-command local startup. HuggingFace Spaces or Fly.io for live demo. |

---

## 7. Technology stack

| Category | Choice |
|---|---|
| Language | Python 3.11. Type annotations throughout. |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (default) + OpenAI Ada (optional) |
| Vector store | FAISS (local dev), Qdrant (prod path) |
| LLM | OpenAI GPT-4o or Anthropic Claude via API. Router pattern, swappable. |
| Drift detection | SciPy (KS test). Optional: `alibi-detect` for MMD. |
| Experiment tracking | Weights & Biases (W&B). Log everything. |
| API | FastAPI + uvicorn. Auto-generated OpenAPI docs at /docs. |
| Frontend | Streamlit (fast demo). Optional: Next.js. |
| Scheduling | APScheduler for background drift checks and re-index jobs. |
| Containerization | Docker + docker-compose. |
| Testing | pytest + httpx for API integration tests. |

---

## 8. Repository structure

```
rag-drift/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── config/
│   └── default.yaml              # all tunables in one place
├── rag/
│   ├── ingestion/
│   │   ├── parsers.py            # PDF, MD, TXT, URL → raw text
│   │   ├── chunker.py            # sliding window chunker
│   │   └── metadata.py           # metadata extraction + schema
│   ├── embedding/
│   │   ├── encoder.py            # pluggable encoder interface
│   │   ├── cache.py              # embedding cache layer
│   │   └── registry.py          # model registry (name → encoder)
│   ├── vector_store/
│   │   ├── base.py               # abstract VectorStore interface
│   │   ├── faiss_store.py        # FAISS implementation
│   │   └── qdrant_store.py       # Qdrant implementation
│   ├── retrieval/
│   │   ├── retriever.py          # top-k cosine similarity search
│   │   └── reranker.py           # cross-encoder re-ranking
│   ├── generation/
│   │   ├── prompt.py             # template builder
│   │   ├── llm.py                # LLM router (OpenAI / Anthropic)
│   │   └── streaming.py          # SSE streaming response
│   ├── drift/
│   │   ├── detector.py           # MMD / KS test logic
│   │   ├── snapshot.py           # index distribution snapshot
│   │   ├── alarm.py              # alert + re-index trigger
│   │   └── scheduler.py          # APScheduler background job
│   ├── api.py                    # FastAPI app + all routes
│   ├── logging.py                # JSONFormatter, get_logger()
│   └── models.py                 # Pydantic models for all layers
├── tests/
│   ├── unit/
│   ├── integration/
│   └── eval/                     # retrieval + generation eval harness
├── notebooks/
│   └── walkthrough.ipynb         # annotated end-to-end demo
└── README.md
```

---

## 9. API endpoints

| Method | Path | Description |
|---|---|---|
| POST | /ingest | Ingest documents (file upload or URL list) |
| POST | /query | Query the RAG system. Returns answer + source chunks. |
| GET | /jobs/{id} | Status of async ingest or re-index job |
| GET | /drift | Current drift score, window history, last alert timestamp |
| POST | /drift/reset | Reset drift history and reference snapshot |
| POST | /reindex | Trigger manual re-index |
| GET | /metrics | Prometheus-compatible metrics |
| GET | /healthz | Liveness probe |

---

## 10. Build phases

| Phase | Days | Deliverable | Demo checkpoint |
|---|---|---|---|
| 1 | 1–4 | Ingestion + FAISS index build. Chunker unit tests passing. | `ingest()` + raw similarity search works in REPL |
| 2 | 5–9 | Full RAG pipeline: retrieval + generation. FastAPI /query live. Streamlit demo. | End-to-end question answering via UI |
| 3 | 10–11 | W&B integration. Log retrieval hits, token counts, latency per request. | W&B dashboard populated after 10 queries |
| 4 | 12–16 | Drift detection module. Simulate drift with shifted query set. Alert fires. | Drift score rises on synthetic drift dataset |
| 5 | 17–19 | Auto re-index trigger. Atomic index swap. Integration tests end-to-end. | Full drift → alert → re-index → recovery cycle |
| 6 | 20–23 | Dockerfile, docker-compose, Fly.io/HF Spaces deploy. README + diagram. | `docker-compose up` → live demo URL |

---

## 11. Evaluation plan

### Retrieval evaluation
- Construct 50 ground-truth (query, relevant_doc) pairs from your corpus
- Measure Hit Rate@1, Hit Rate@5, MRR at baseline
- Re-measure after simulating drift and again after re-index to confirm recovery

### Drift detection evaluation
Simulate three scenarios:
1. **Gradual drift** — slowly shift query topic over 200 queries
2. **Sudden drift** — inject completely off-topic queries after query 100
3. **Control** — same query distribution throughout, drift score should stay flat

Confirm: drift score rises in scenarios 1 & 2, stays flat in 3. Tune threshold α to minimize false positive rate on the control.

### Generation quality
- Use an LLM-as-judge prompt to score faithfulness (does answer follow only from retrieved context?) on 20 held-out queries
- Target: faithfulness > 0.85 on non-drifted index


---

## 14. Stretch goals

- Fine-grained attribution: highlight which chunk contributed most to each answer
- User feedback loop: thumbs up/down feeds back into relevance scoring
- Multi-index routing: detect query topic and route to different specialized indexes
- Async ingestion pipeline using asyncio for concurrent document fetching
