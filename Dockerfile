# syntax=docker/dockerfile:1
# Multi-stage build: builder installs deps, runtime is a slim image.

FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml .
RUN pip install --upgrade pip && pip install build && python -m build --wheel

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl
COPY . .
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/healthz || exit 1
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
