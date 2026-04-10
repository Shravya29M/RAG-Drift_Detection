# CLAUDE.md — rag-drift

## Project overview

This is a portfolio RAG pipeline with adaptive drift detection. Full spec is in `PRD.md` at the repo root — read it before writing any code. The architecture has five layers (ingestion, embedding, vector store, retrieval+generation, drift monitor) each in its own subpackage under `rag/`.

## Python conventions

- Python 3.11+. Use `from __future__ import annotations` in every file.
- Type annotations are mandatory everywhere — functions, class attributes, return types. No `Any` unless genuinely unavoidable and commented.
- Use Pydantic v2 for all data models. Models live in `rag/models.py` unless they're layer-specific, in which case co-locate with the layer.
- Docstrings on every public function and class. One-line summary, then args/returns if non-obvious.
- No `print()` anywhere in library code. Use `rag.logging.get_logger(__name__)` which returns a structured JSON logger.

## Code style

- Formatter and linter: `ruff`. Config is in `pyproject.toml`. Run `ruff check . --fix && ruff format .` before finishing any task.
- Type checker: `mypy --strict`. All files must pass. Fix type errors, don't silence them with `# type: ignore` unless there's no other option (add a comment explaining why).
- Max line length: 100 (set in pyproject.toml).
- Imports: stdlib → third-party → internal, separated by blank lines. No wildcard imports.

## Architecture rules

- **Layer interfaces are abstract base classes.** `VectorStore`, `Encoder`, and `LLMRouter` in `rag/vector_store/base.py`, `rag/embedding/encoder.py`, and `rag/generation/llm.py` respectively. Concrete implementations inherit from these. Never import a concrete class directly in consuming code — always depend on the abstract interface.
- **Config is centralized.** All tunables (chunk size, overlap, top-k, drift window, threshold α, model names) live in `config/default.yaml` and are loaded once at startup into a `Settings` Pydantic model. No magic numbers in code — reference `settings.chunk_size`, not `512`.
- **No HTTP in the core layers.** The FastAPI app in `rag/api.py` is the only place that handles requests and responses. Ingestion, retrieval, generation, and drift modules are plain Python classes with no web framework dependency.
- **Drift module is stateful but not a singleton.** `DriftDetector` takes a snapshot and a config at init. The APScheduler job holds the instance. Don't use module-level globals.

## Testing

- Test runner: `pytest`. Run with `pytest -x -q` during development.
- Every new module needs a corresponding test file in `tests/unit/test_<module>.py`.
- Use `pytest.fixture` for shared setup (broker, encoder, temp FAISS index). No setup in test bodies.
- Mock external calls (LLM API, W&B) using `unittest.mock.patch`. Tests must pass with no internet access and no API keys set.
- Concurrency tests go in `tests/integration/` not `tests/unit/`.
- Target: `pytest` passes clean before any task is considered done.

## W&B logging

- Import pattern: `import wandb` only in `rag/tracking.py`. All other modules call `from rag.tracking import log_event` — never import wandb directly in core logic.
- If `WANDB_API_KEY` is not set, `log_event` should no-op silently (dev mode). Never crash because W&B is unavailable.
- Log on: every query (retrieval hits, latency, token count), every drift window (score, window size, threshold), every re-index (trigger reason, duration, hit rate before/after).

## File structure

Refer to PRD.md Section 8 for the full tree. Key rules:
- One class per file for core abstractions (encoder, retriever, detector). Helpers and utils can be grouped.
- `rag/models.py` is the single source of truth for shared Pydantic models. Don't redefine schemas in multiple places.
- `rag/api.py` contains only route definitions and request/response wiring. Business logic belongs in the layer modules.

## Environment and secrets

- Secrets are in `.env` (gitignored). Load with `python-dotenv` at startup.
- Required env vars: `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (at least one), `WANDB_API_KEY` (optional), `QDRANT_URL` (optional, falls back to FAISS).
- Never hardcode API keys or URLs. Never log them.

## Docker

- `Dockerfile` uses a multi-stage build: builder stage installs deps, runtime stage is slim.
- `docker-compose.yml` brings up: `api` (FastAPI on port 8000), `qdrant` (vector store on port 6333).
- The API container mounts `./data` for document uploads and `./index` for the FAISS index file.
- Health check: `GET /healthz` must return 200 before the container is considered ready.

## When implementing a new feature

1. Check PRD.md to confirm it's in scope and understand the spec.
2. Write the Pydantic model or update `rag/models.py` first.
3. Implement the core logic with full type annotations.
4. Write unit tests. Run `pytest -x` and confirm passing.
5. If it's user-facing, add or update the FastAPI route in `rag/api.py`.
6. Run `ruff check . --fix && ruff format . && mypy --strict rag/` and fix all issues.
7. Update the relevant section of `README.md` if behavior changed.

## Common pitfalls to avoid

- Don't use `queue.Queue` for the FAISS index — it's not thread-safe for concurrent reads during a swap. Use `threading.RLock` around index reads/writes and maintain two index slots (A/B) for atomic swaps.
- Don't call `wandb.init()` in tests. The `rag.tracking` module checks for `WANDB_DISABLED=true` env var and no-ops.
- Don't store raw embeddings in the job registry — they're large. Store the job_id and look up the embedding from the vector store when needed.
- FAISS `IndexFlatIP` expects normalized vectors for cosine similarity. Always L2-normalize embeddings before adding to the index.
- APScheduler jobs run in a thread pool by default. Ensure anything they call is thread-safe.



## Response constraints

- Be concise. Prefer the shortest correct answer.
- Avoid explanations unless explicitly requested.
- Do not restate the problem or requirements.
- Output only what is necessary (no filler text).
- Prefer bullet points over paragraphs when possible.
- Omit obvious context and boilerplate.
- Do not include comments in code unless essential.

## Execution rules

- Assume permission is granted for all file edits.
- Do not ask for confirmation before modifying files.
- Proceed with changes directly unless explicitly told otherwise.
- If unsure, make a reasonable assumption and continue.
- Do not pause for approval.
- Do not ever write Claude or created by Claude anywhere in the code or comments. I dont want to mention claude or AI or AI model names anywhere, it should look like I wrote it. 