"""Download a public documentation corpus and derive benchmark query sets.

Corpus: the official Python 3.14 documentation, plain-text build (PSF licence),
sections library/, howto/, tutorial/, reference/. Chunking is NOT done here —
the files are ingested through the running service's POST /ingest so the
benchmark exercises the real ingestion path and multi-replica index sync.

Query sets written to benchmarks/corpus/queries.json:
  * on_topic  — built from the docs' own section headings, qualified by the
    page's module/topic name ("json basic usage", "how to use asyncio
    streams"). Synthetic: no public query log exists for these docs.
  * off_topic — real human questions from SQuAD v2.0 dev, restricted to
    articles unrelated to programming. Used only as the injected drift.

Usage:  python benchmarks/build_docs_corpus.py
"""

from __future__ import annotations

import io
import json
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

DOCS_URL = "https://docs.python.org/3/archives/python-3.14-docs-text.zip"
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
SECTIONS = ("library", "howto", "tutorial", "reference")
# SQuAD articles that overlap a programming corpus are excluded from "off-topic".
SQUAD_EXCLUDE = {"Computational_complexity_theory", "Packet_switching", "Prime_number"}
SEED = 20260927

OUT = Path(__file__).resolve().parent / "corpus"
_UNDERLINE = re.compile(r"^([=*\-~^\"#+])\1{2,}$")
_MODULE_TITLE = re.compile(r'^"([^"]+)"\s+---\s+(.+)$')
_TEMPLATES = ("{t}", "how to use {t}", "{t} example", "what is {t}", "python {t}")


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 — fixed https URLs
        data: bytes = resp.read()
    return data


def _headings(text: str) -> list[str]:
    lines = text.splitlines()
    found = []
    for prev, line in zip(lines, lines[1:], strict=False):
        title = prev.strip()
        if title and _UNDERLINE.match(line.strip()) and len(line.strip()) == len(prev.rstrip()):
            found.append(title)
    return found


def _on_topic_queries(doc_dir: Path, rng: random.Random) -> list[str]:
    queries: set[str] = set()
    for path in sorted(doc_dir.rglob("*.txt")):
        heads = _headings(path.read_text(encoding="utf-8"))
        if not heads:
            continue
        top = heads[0]
        m = _MODULE_TITLE.match(top)
        topic = m.group(1) if m else path.stem.replace("_", " ")
        if m:
            queries.add(m.group(2).strip().lower())
        for head in heads[1:]:
            head = re.sub(r"[\"'`]", "", head)
            head = re.sub(r"^\d+(\.\d+)*\.?\s+", "", head).strip()  # "9.4. Random Remarks"
            if not (3 <= len(head) <= 60) or head.lower() in {"examples", "footnotes", "notes"}:
                continue
            base = f"{topic} {head.lower()}"
            queries.add(rng.choice(_TEMPLATES).format(t=base))
    return sorted(queries)


def _off_topic_queries() -> list[str]:
    data = json.loads(_fetch(SQUAD_URL))
    out = []
    for article in data["data"]:
        if article["title"] in SQUAD_EXCLUDE:
            continue
        for para in article["paragraphs"]:
            out.extend(qa["question"].strip() for qa in para["qas"])
    return sorted(set(out))


def main() -> None:
    rng = random.Random(SEED)
    doc_dir = OUT / "docs"
    if not doc_dir.exists():
        print(f"downloading {DOCS_URL}", file=sys.stderr)
        with zipfile.ZipFile(io.BytesIO(_fetch(DOCS_URL))) as zf:
            for name in zf.namelist():
                parts = Path(name).parts
                if len(parts) >= 3 and parts[1] in SECTIONS and name.endswith(".txt"):
                    target = doc_dir / Path(*parts[1:])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(name))

    files = sorted(doc_dir.rglob("*.txt"))
    words = sum(len(p.read_text(encoding="utf-8").split()) for p in files)
    on_topic = _on_topic_queries(doc_dir, rng)
    off_topic = _off_topic_queries()
    meta = {
        "docs_source": DOCS_URL,
        "sections": list(SECTIONS),
        "files": len(files),
        "whitespace_words": words,
        "on_topic_source": "section headings of the corpus, templated (synthetic)",
        "off_topic_source": f"{SQUAD_URL} minus {sorted(SQUAD_EXCLUDE)}",
        "seed": SEED,
    }
    (OUT / "queries.json").write_text(
        json.dumps({"meta": meta, "on_topic": on_topic, "off_topic": off_topic}, indent=1)
    )
    print(json.dumps({**meta, "on_topic": len(on_topic), "off_topic": len(off_topic)}, indent=1))


if __name__ == "__main__":
    main()
