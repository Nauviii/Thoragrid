"""Show dense-only and hybrid retrieval side by side, for the same queries.

This is not a benchmark. There are no gold labels here and nothing is scored, so it cannot
say which retriever is better — only whether the lexical half changes anything at all. That
is still worth seeing: if hybrid returns exactly what dense returned for every query, BM25 is
wired in but contributing nothing, and no amount of architecture diagram makes that untrue.

The queries below are chosen to stress the two halves differently. Some lean on paraphrase,
which is where embeddings are strong; others hinge on a token that has to match exactly —
an abbreviation, a drug name, a procedure — which is what BM25 was added to recover.

Usage:
    python scripts/compare_retrieval.py
    python scripts/compare_retrieval.py --query "tension pneumothorax needle decompression"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinecone import Pinecone

from config.settings import settings
from core.rag import bm25_index
from core.rag.retriever import _dense_search, _hybrid_search

# (query, condition filter or None). The filtered ones mirror the image path, where the LLM
# writes a short keyword query and retrieval is already narrowed to one condition.
_PROBES: list[tuple[str, str | None]] = [
    ("what happens when air leaks into the space around the lung", None),
    ("tension pneumothorax needle decompression", None),
    ("PEEP mechanical ventilation obstructive disease", None),
    ("honeycombing usual interstitial pneumonia", None),
    ("pleural fluid blunting costophrenic angle", "Effusion"),
    ("enlarged cardiac silhouette cardiothoracic ratio", "Cardiomegaly"),
]

_TOP_K = 4


def _label(chunk: dict) -> str:
    """Short, stable identifier for a chunk in the comparison table."""
    return f"{chunk['condition'][:14]}/{chunk['section'][:12]}  {chunk['chunk_id'][:8]}"


def _compare(query: str, condition: str | None, index) -> tuple[int, int]:
    """Print one query's two result lists; return (changed_positions, total_positions)."""
    candidates = max(_TOP_K * settings.rag_candidate_multiplier, _TOP_K)
    dense = [chunk for _, chunk in
             _dense_search(query, index, candidates, settings.pinecone_namespace, condition)]
    dense_top = dense[:_TOP_K]
    hybrid = _hybrid_search(query, index, _TOP_K, settings.pinecone_namespace, condition)

    scope = f"  [condition={condition}]" if condition else ""
    print(f'\n"{query}"{scope}')
    print(f'  {"":<4}{"dense only":<44}{"hybrid":<44}')
    print("  " + "-" * 90)

    changed = 0
    for rank in range(max(len(dense_top), len(hybrid))):
        left = _label(dense_top[rank]) if rank < len(dense_top) else ""
        right = _label(hybrid[rank]) if rank < len(hybrid) else ""
        mark = "" if left == right else "  <- moved"
        if left != right:
            changed += 1
        print(f"  {rank + 1:<4}{left:<44}{right:<44}{mark}")

    dense_ids = {c["chunk_id"] for c in dense_top}
    hybrid_ids = {c["chunk_id"] for c in hybrid}
    only_hybrid = hybrid_ids - dense_ids
    if only_hybrid:
        print(f"  {len(only_hybrid)} chunk(s) surfaced only by hybrid:")
        for chunk in hybrid:
            if chunk["chunk_id"] in only_hybrid:
                print(f'      {_label(chunk)}  "{chunk["text"][:64].strip()}…"')
    else:
        print("  same set of chunks, only ordering may differ")

    return changed, max(len(dense_top), len(hybrid))


def main() -> None:
    """Run every probe and summarise how often the lexical half changed the answer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", help="run this query instead of the defaults")
    parser.add_argument("--condition", default=None, help="restrict a custom query to one condition")
    args = parser.parse_args()

    if not bm25_index.is_available():
        raise SystemExit(
            "models/weights/bm25_corpus.json is missing — hybrid would fall back to dense and "
            "this comparison would print two identical columns. Re-run ingestion first."
        )

    index = Pinecone(api_key=settings.pinecone_api_key).Index(settings.pinecone_index_name)
    probes = [(q, args.condition) for q in args.query] if args.query else _PROBES

    print(f"Encoder : {settings.rag_embedding_model}")
    print(f"Index   : {settings.pinecone_index_name}  (top_k={_TOP_K}, "
          f"candidates={_TOP_K * settings.rag_candidate_multiplier}, rrf_k={settings.rag_rrf_k})")

    total_changed = total_positions = queries_affected = 0
    for query, condition in probes:
        changed, positions = _compare(query, condition, index)
        total_changed += changed
        total_positions += positions
        queries_affected += changed > 0

    print()
    print("=" * 92)
    print(f"Hybrid changed the result for {queries_affected} of {len(probes)} queries, "
          f"{total_changed} of {total_positions} ranked positions.")
    if total_changed == 0:
        print("BM25 is wired in but is not moving anything. Worth investigating before")
        print("describing this system as hybrid.")


if __name__ == "__main__":
    main()