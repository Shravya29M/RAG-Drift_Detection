"""W&B integration wrapper; exposes log_event() and isolates all wandb imports here."""

from __future__ import annotations

import os

import wandb


def log_event(event: str, data: dict[str, object]) -> None:
    """Log a named event to Weights & Biases.

    No-ops silently when:
    - ``WANDB_DISABLED=true`` is set in the environment (test / dev mode), or
    - ``wandb.run`` is ``None`` (no active run has been initialised).

    All other modules must call this function instead of importing ``wandb``
    directly.

    Args:
        event: Short human-readable event name used as a W&B metric prefix
            (e.g. ``"drift_window"``, ``"query"``).
        data: Flat mapping of metric names to scalar values.  Keys will be
            prefixed with ``"<event>/"``.
    """
    if os.environ.get("WANDB_DISABLED", "").lower() == "true":
        return
    if wandb.run is None:
        return
    wandb.log({f"{event}/{k}": v for k, v in data.items()})
