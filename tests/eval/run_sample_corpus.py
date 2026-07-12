"""Reproducible retrieval and quality-gate evaluation on the bundled samples.

Run from the repository root:

    python -m tests.eval.run_sample_corpus

The retrieval metrics use the production SentenceTransformer encoder, FAISS store,
and Retriever.  The gate scenarios deliberately script the *drift verdict* so this
small two-document corpus does not pretend to validate a statistical detector; they
exercise the decision policy with scores obtained from real retrieval.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

# The model is downloaded by the project's normal setup flow.  Keep evaluation
# reproducible and prevent an otherwise-cached run from making a network HEAD request.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from rag.drift.detector import DriftDetector
from rag.embedding.encoder import Encoder, SentenceTransformerEncoder
from rag.ingestion.chunker import chunk_text
from rag.ingestion.parsers import parse_markdown
from rag.models import DriftConfig, DriftResult, IngestConfig, SourceType
from rag.retrieval.retriever import Retriever
from rag.vector_store.faiss_store import FAISSStore

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "samples"
LABELS = Path(__file__).with_name("sample_corpus.json")


@dataclass(frozen=True)
class Label:
    query: str
    source: str


class _ScriptedDriftSnapshot:
    """Return a fixed drift verdict while preserving the detector's gate logic."""

    def __init__(self, drifted: bool) -> None:
        self._drifted = drifted

    def compare(self, window: np.ndarray, *, reference: np.ndarray) -> DriftResult:
        return DriftResult(
            statistic=1.0 if self._drifted else 0.0,
            pvalue=0.001 if self._drifted else 1.0,
            drifted=self._drifted,
            window_size=len(window),
            snapshot_size=len(reference),
        )


def _load_labels() -> list[Label]:
    raw = json.loads(LABELS.read_text(encoding="utf-8"))
    return [Label(**item) for item in raw]


def _build_retriever() -> tuple[Retriever, Encoder]:
    chunks = []
    for path in sorted(SAMPLES.glob("*.md")):
        for section in parse_markdown(path):
            chunks.extend(
                chunk_text(
                    section,
                    path.name,
                    SourceType.MARKDOWN,
                    IngestConfig(chunk_size=512, chunk_overlap=64),
                    file_path=path,
                )
            )
    encoder = SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2")
    store = FAISSStore(encoder.dim)
    store.add(chunks, encoder.encode([chunk.text for chunk in chunks]))
    return Retriever(store, encoder), encoder


def _retrieval_metrics(retriever: Retriever, labels: list[Label]) -> dict[str, float]:
    reciprocal_ranks: list[float] = []
    hit1 = 0
    hit5 = 0
    for label in labels:
        result = retriever.retrieve(label.query, k=5)
        ranks = [
            i + 1 for i, chunk in enumerate(result.chunks) if chunk.metadata.source == label.source
        ]
        if ranks:
            reciprocal_ranks.append(1.0 / ranks[0])
            hit1 += int(ranks[0] == 1)
            hit5 += 1
        else:
            reciprocal_ranks.append(0.0)
    total = len(labels)
    return {
        "hit_rate_at_1": hit1 / total,
        "hit_rate_at_5": hit5 / total,
        "mrr_at_5": sum(reciprocal_ranks) / total,
    }


def _mean_scores(
    retriever: Retriever, encoder: Encoder, queries: list[str]
) -> tuple[np.ndarray, list[float]]:
    embeddings = encoder.encode(queries)
    scores = []
    for query in queries:
        result = retriever.retrieve(query, k=5)
        scores.append(float(np.mean(result.scores)) if result.scores else 0.0)
    return embeddings, scores


def _gate_decision(
    retriever: Retriever,
    encoder: Encoder,
    baseline_queries: list[str],
    candidate_queries: list[str],
    *,
    mode: Literal["benign", "auto", "fallback"],
) -> tuple[str, float, float]:
    baseline_embeddings, baseline_scores = _mean_scores(retriever, encoder, baseline_queries)
    candidate_embeddings, candidate_scores = _mean_scores(retriever, encoder, candidate_queries)
    window_size = len(baseline_queries)
    config = DriftConfig(window_size=window_size, pca_components=1, hysteresis_windows=3)
    detector = DriftDetector(_ScriptedDriftSnapshot(drifted=True), config)  # type: ignore[arg-type]
    for embedding, score in zip(baseline_embeddings, baseline_scores, strict=True):
        detector.add_query_embedding(embedding, top_score=score)
    for _ in range(3):
        for embedding, score in zip(candidate_embeddings, candidate_scores, strict=True):
            detector.add_query_embedding(embedding, top_score=None if mode == "fallback" else score)
    last = detector.history[-1]
    decision = "benign_recalibration" if last.recalibrated else "auto_reindex"
    return decision, float(np.mean(baseline_scores)), float(np.mean(candidate_scores))


def main() -> None:
    retriever, encoder = _build_retriever()
    labels = _load_labels()
    metrics = _retrieval_metrics(retriever, labels)

    rag_queries = [
        label.query for label in labels if label.source == "retrieval_augmented_generation.md"
    ]
    drift_queries = [label.query for label in labels if label.source == "embedding_drift.md"]
    off_topic = [
        "best slow cooker recipes for beef stew",
        "how to improve marathon training pace",
        "top beach resorts in the Maldives",
        "acoustic guitar chords for beginners",
        "when to plant tomatoes in a home garden",
        "latest football transfer rumours",
        "easy sourdough bread starter instructions",
        "mystery novel reviews this year",
    ]

    benign, baseline_score, covered_score = _gate_decision(
        retriever, encoder, rag_queries, drift_queries, mode="benign"
    )
    stale, _, off_topic_score = _gate_decision(
        retriever, encoder, rag_queries, off_topic, mode="auto"
    )
    fallback, _, _ = _gate_decision(retriever, encoder, rag_queries, off_topic, mode="fallback")
    output = {
        "corpus": "2 bundled markdown documents; 16 manually labeled source-level queries",
        "retrieval": metrics,
        "quality_gate": {
            "baseline_mean_top_score": baseline_score,
            "covered_topic_mean_top_score": covered_score,
            "off_topic_mean_top_score": off_topic_score,
            "covered_topic_decision": benign,
            "off_topic_decision": stale,
            "no_score_decision": fallback,
        },
        "limitations": (
            "The gate scenarios script a sustained drift verdict; they validate the quality "
            "decision, not detector precision on a production traffic distribution."
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
