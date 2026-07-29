"""Lexical (BM25) half of hybrid retrieval, over a corpus artifact built at ingestion time."""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from config.settings import settings

# Split on anything that is not alphanumeric, keeping intra-word hyphens out of the way:
# "post-processing" becomes two terms, which is what a reader searching either half wants.
_TOKEN = re.compile(r"[a-z0-9]+")

_index: dict | None = None


def _tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric terms."""
    return _TOKEN.findall(text.lower())


def _load() -> dict | None:
    """Load and build the BM25 index once, or return None if the artifact is absent.

    Absence is not an error. Retrieval degrades to dense-only, which is what the system did
    before hybrid existed — a missing lexical half should cost recall, not availability.
    """
    global _index
    if _index is not None:
        return _index or None

    path = Path(settings.bm25_corpus_path)
    if not path.exists():
        _index = {}
        return None

    chunks = json.loads(path.read_text())["chunks"]
    _index = {
        "chunks": chunks,
        "by_id": {c["chunk_id"]: c for c in chunks},
        "bm25": BM25Okapi([_tokenize(c["text"]) for c in chunks]),
    }
    return _index


def chunk_by_id(chunk_id: str) -> dict | None:
    """Return the full chunk record for an id, or None if the corpus is unavailable."""
    index = _load()
    return index["by_id"].get(chunk_id) if index else None


def is_available() -> bool:
    """Whether the lexical half can run at all."""
    return _load() is not None


def search(query: str, top_n: int, condition: str | None = None) -> list[str]:
    """Return up to top_n chunk ids ranked by BM25, optionally restricted to one condition.

    Scoring runs over the whole corpus before the condition filter is applied, so the term
    statistics stay global. Scoring only within a condition would make a term that is common
    everywhere look rare inside the subset, and inflate its weight for no good reason.
    """
    index = _load()
    if index is None:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = index["bm25"].get_scores(tokens)
    ranked = sorted(
        (
            (score, chunk)
            for score, chunk in zip(scores, index["chunks"])
            if score > 0 and (condition is None or chunk["condition"] == condition)
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return [chunk["chunk_id"] for _, chunk in ranked[:top_n]]


def build_corpus_artifact(chunks: list[dict], path: Path | None = None) -> int:
    """Write the chunk corpus that BM25 reads at query time; called from the ingestion run."""
    destination = Path(path or settings.bm25_corpus_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "embedding_model": settings.rag_embedding_model,
        "n_chunks": len(chunks),
        "chunks": [
            {k: c[k] for k in ("chunk_id", "condition", "section", "source", "text")}
            for c in chunks
        ],
    }, indent=2))
    return len(chunks)