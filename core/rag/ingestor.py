"""Embed text chunks and upsert vectors to Pinecone.

The BM25 corpus artifact is written by the ingestion script rather than here: this runs
once per condition, and building the artifact inside it would leave only the last one.
"""

from sentence_transformers import SentenceTransformer
from pinecone import Index

from config.settings import settings

_MODEL = SentenceTransformer(settings.rag_embedding_model)


def embed_and_upsert(
    chunks: list[dict],
    index: Index,
    namespace: str,
    batch_size: int = 100,
) -> int:
    """Embed chunks and upsert to Pinecone; return number of vectors upserted."""
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    embeddings = _MODEL.encode(texts, batch_size=32, show_progress_bar=False)

    vectors = [
        {
            "id": c["chunk_id"],
            "values": emb.tolist(),
            "metadata": {
                "condition": c["condition"],
                "section":   c["section"],
                "source":    c["source"],
                "text":      c["text"],
            },
        }
        for c, emb in zip(chunks, embeddings)
    ]

    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size], namespace=namespace)

    return len(vectors)