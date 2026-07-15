"""Drift-detection benchmark for RAG-Drift_Detection on real human queries.

Scenario per trial: a corpus is built from SQuAD articles on topic set A;
query traffic starts on topic A (baseline + clean windows), then shifts to
unrelated topic set B. The detector should stay quiet on clean traffic and
fire after the shift.

Measured over N trials with disjoint random topic splits:
  - false-alarm rate: drifted windows among clean (in-distribution) windows
  - detection rate: trials where the shift is flagged within the budget
  - windows-to-detect: windows after the shift until first drifted=True
  - windows-to-alarm: windows until hysteresis alarm (3 consecutive)

Uses the repo's own DriftDetector/DistributionSnapshot with default config
(window=50, alpha=0.05, pca=32, hysteresis=3) and its default encoder
(all-MiniLM-L6-v2). No LLM calls; embeddings only.
"""
import json
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")  # run from repo root
from rag.drift.detector import DriftDetector
from rag.drift.snapshot import DistributionSnapshot
from rag.models import DriftConfig

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

SEED = 3407
N_TRIALS = 20
CLEAN_WINDOWS = 10          # in-distribution windows fed after baseline
MAX_SHIFT_WINDOWS = 6       # detection budget after the shift
TITLES_PER_SIDE = 6         # articles per topic set
OUT = sys.argv[1] if len(sys.argv) > 1 else "drift_benchmark_results.json"

print("Loading SQuAD validation split...", flush=True)
ds = load_dataset("rajpurkar/squad", split="validation")
by_title = defaultdict(lambda: {"questions": [], "contexts": set()})
for row in ds:
    by_title[row["title"]]["questions"].append(row["question"])
    by_title[row["title"]]["contexts"].add(row["context"])

# titles with enough questions to fill baseline + clean/shift windows
cfg = DriftConfig()
need_q = cfg.window_size * (1 + CLEAN_WINDOWS)  # per A-side
titles = sorted(t for t, v in by_title.items() if len(v["questions"]) >= 80)
print(f"{len(titles)} titles with >=80 questions; window={cfg.window_size}, "
      f"alpha={cfg.threshold_alpha}, pca={cfg.pca_components}, "
      f"hysteresis={cfg.hysteresis_windows}", flush=True)

print("Loading encoder all-MiniLM-L6-v2...", flush=True)
enc = SentenceTransformer("all-MiniLM-L6-v2")

# Pre-embed everything once per title (questions and contexts)
emb_cache_q, emb_cache_c = {}, {}
for t in titles:
    emb_cache_q[t] = enc.encode(by_title[t]["questions"], show_progress_bar=False)
    emb_cache_c[t] = enc.encode(sorted(by_title[t]["contexts"]), show_progress_bar=False)
print("embedding cache built", flush=True)

rng = random.Random(SEED)
trials = []
for trial in range(N_TRIALS):
    picks = rng.sample(titles, TITLES_PER_SIDE * 2)
    side_a, side_b = picks[:TITLES_PER_SIDE], picks[TITLES_PER_SIDE:]

    corpus_vecs = np.vstack([emb_cache_c[t] for t in side_a])
    a_queries = np.vstack([emb_cache_q[t] for t in side_a])
    b_queries = np.vstack([emb_cache_q[t] for t in side_b])
    a_idx = rng.sample(range(len(a_queries)), min(need_q, len(a_queries)))
    b_idx = rng.sample(range(len(b_queries)),
                       min(cfg.window_size * MAX_SHIFT_WINDOWS, len(b_queries)))

    snapshot = DistributionSnapshot(corpus_vecs, cfg)
    det = DriftDetector(snapshot, cfg)

    # baseline calibration window
    for i in a_idx[:cfg.window_size]:
        det.add_query_embedding(a_queries[i])
    assert det.baseline_ready

    # clean in-distribution windows
    false_alarms = 0
    clean_fed = 0
    for i in a_idx[cfg.window_size:]:
        r = det.add_query_embedding(a_queries[i])
        clean_fed += 1
        if r is not None and r.drifted:
            false_alarms += 1
        if clean_fed >= cfg.window_size * CLEAN_WINDOWS:
            break
    clean_windows_run = clean_fed // cfg.window_size

    # distribution shift: topic B traffic
    windows_to_detect = None
    windows_to_alarm = None
    shift_windows = 0
    for n, i in enumerate(b_idx):
        r = det.add_query_embedding(b_queries[i])
        if r is not None:
            shift_windows += 1
            if r.drifted and windows_to_detect is None:
                windows_to_detect = shift_windows
            if det.consecutive_alerts >= cfg.hysteresis_windows and windows_to_alarm is None:
                windows_to_alarm = shift_windows
        if shift_windows >= MAX_SHIFT_WINDOWS:
            break

    trials.append({
        "trial": trial, "titles_a": side_a, "titles_b": side_b,
        "clean_windows": clean_windows_run, "false_alarms": false_alarms,
        "windows_to_detect": windows_to_detect, "windows_to_alarm": windows_to_alarm,
    })
    print(f"trial {trial+1}/{N_TRIALS}: clean_windows={clean_windows_run} "
          f"false_alarms={false_alarms} detect_in={windows_to_detect} "
          f"alarm_in={windows_to_alarm}", flush=True)

total_clean = sum(t["clean_windows"] for t in trials)
total_fa = sum(t["false_alarms"] for t in trials)
detected = [t for t in trials if t["windows_to_detect"] is not None]
alarmed = [t for t in trials if t["windows_to_alarm"] is not None]
ttd = sorted(t["windows_to_detect"] for t in detected)
summary = {
    "config": {"window_size": cfg.window_size, "alpha": cfg.threshold_alpha,
               "pca_components": cfg.pca_components,
               "hysteresis_windows": cfg.hysteresis_windows,
               "encoder": "all-MiniLM-L6-v2", "dataset": "rajpurkar/squad validation",
               "n_trials": N_TRIALS, "seed": SEED},
    "clean_windows_total": total_clean,
    "false_alarm_windows": total_fa,
    "false_alarm_rate": total_fa / total_clean if total_clean else None,
    "detection_rate": len(detected) / N_TRIALS,
    "alarm_rate_within_budget": len(alarmed) / N_TRIALS,
    "median_windows_to_detect": ttd[len(ttd)//2] if ttd else None,
    "median_queries_to_detect": ttd[len(ttd)//2] * cfg.window_size if ttd else None,
    "trials": trials,
}
with open(OUT, "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "trials"}, indent=2))
