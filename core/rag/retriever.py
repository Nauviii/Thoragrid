"""Hybrid retrieval: dense embeddings and BM25, fused by reciprocal rank."""

from collections import defaultdict

from pinecone import Index
from sentence_transformers import SentenceTransformer

from config.settings import settings
from core.rag import bm25_index

_MODEL = SentenceTransformer(settings.rag_embedding_model)


def _adaptive_top_k(n_conditions: int) -> int:
    """Return per-condition top_k scaled to number of conditions above threshold."""
    if n_conditions == 1:
        return 4
    if n_conditions <= 3:
        return 3
    return 2


def _dense_search(
    query: str, index: Index, top_n: int, namespace: str, condition: str | None,
) -> list[tuple[str, dict]]:
    """Return (chunk_id, chunk) pairs from Pinecone, ranked by cosine similarity."""
    vector = _MODEL.encode(query).tolist()
    response = index.query(
        vector=vector,
        top_k=top_n,
        namespace=namespace,
        filter={"condition": {"$eq": condition}} if condition else None,
        include_metadata=True,
    )
    return [
        (match.id, {
            "chunk_id":  match.id,
            "condition": match.metadata.get("condition", ""),
            "section":   match.metadata.get("section", ""),
            "source":    match.metadata.get("source", ""),
            "text":      match.metadata.get("text", ""),
            "score":     round(match.score, 4),
        })
        for match in response.matches
    ]


def _reciprocal_rank_fusion(rankings: list[list[str]], k: int) -> list[str]:
    """Fuse ranked id lists into one ordering by summed reciprocal rank.

    k damps the influence of the top positions. At the conventional 60 a first place is worth
    1/61 and a tenth 1/70, so agreement across both retrievers outranks a single strong hit —
    which is the property the fusion exists for.
    """
    totals: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            totals[chunk_id] += 1.0 / (k + rank)
    return sorted(totals, key=totals.get, reverse=True)


def _hybrid_search(
    query: str, index: Index, top_k: int, namespace: str, condition: str | None,
) -> list[dict]:
    """Run both retrievers, fuse by reciprocal rank, and return the top_k chunks."""
    candidates = max(top_k * settings.rag_candidate_multiplier, top_k)

    dense = _dense_search(query, index, candidates, namespace, condition)
    dense_by_id = dict(dense)
    dense_ids = [chunk_id for chunk_id, _ in dense]

    if not bm25_index.is_available():
        return [dense_by_id[i] for i in dense_ids[:top_k]]

    lexical_ids = bm25_index.search(query, candidates, condition)
    fused = _reciprocal_rank_fusion([dense_ids, lexical_ids], settings.rag_rrf_k)

    results = []
    for chunk_id in fused[:top_k]:
        chunk = dense_by_id.get(chunk_id) or bm25_index.chunk_by_id(chunk_id)

        if chunk:
            chunk = dict(chunk)
            chunk.setdefault("score", None)
            results.append(chunk)
    return results


def retrieve_for_image_path(
    rag_queries: list[dict],
    index: Index,
    namespace: str,
) -> list[dict]:
    """Retrieve and deduplicate chunks for all above-threshold conditions (image path).

    Args:
        rag_queries: [{"condition": str, "query": str}, ...] from LLM Call 1 output.
    """
    top_k = _adaptive_top_k(len(rag_queries))
    seen, chunks = set(), []

    for item in rag_queries:
        for chunk in _hybrid_search(
            item["query"], index, top_k, namespace, item["condition"]
        ):
            if chunk["chunk_id"] not in seen:
                seen.add(chunk["chunk_id"])
                chunks.append(chunk)

    return chunks


def retrieve_for_text_path(
    query: str,
    index: Index,
    namespace: str,
    top_k: int | None = None,
) -> list[dict]:
    """Retrieve chunks without a condition filter, for the general text Q&A path."""
    return _hybrid_search(query, index, top_k or settings.rag_top_k, namespace, None)