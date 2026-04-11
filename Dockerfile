# syntax=docker/dockerfile:1
# Stage 1 — builder: install all Python dependencies into an isolated venv.
# Stage 2 — runtime: copy the venv + source onto a minimal slim image.

# ---- builder ----------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools and libgomp (required by faiss-cpu at install time).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated venv so we can copy it cleanly to the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first (layer is cached as long as requirements.txt unchanged).
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---- runtime ----------------------------------------------------------------
FROM python:3.11-slim AS runtime

# libgomp1 is needed at runtime by faiss-cpu.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre-built venv from the builder stage.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Ensure Python uses the venv and doesn't write .pyc files into the image layer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application source.
COPY rag/       ./rag/
COPY config/    ./config/
COPY pyproject.toml .

# Directories mounted at runtime via docker-compose volumes;
# pre-create them so the container starts cleanly even without a mount.
RUN mkdir -p /app/data /app/index

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=5 \
    CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
