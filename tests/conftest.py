"""Shared test configuration: environment hardening for hermetic runs."""

from __future__ import annotations

import os

# faiss and torch each bundle their own OpenMP runtime; letting both spin up
# thread pools crashes the interpreter on macOS. Single-threaded OpenMP is
# plenty for test-sized workloads and must be set before either library loads.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Never talk to W&B from tests, regardless of the developer's shell env.
os.environ.setdefault("WANDB_DISABLED", "true")
