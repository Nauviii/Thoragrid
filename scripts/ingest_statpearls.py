"""Orchestrate knowledge base ingestion: chunk all condition files and upsert to Pinecone."""
import sys
from pathlib import Path
import json
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from pinecone import Pinecone
from config.settings import settings
from config.condition_mapping import CONDITION_MAPPING
from core.rag.chunking import chunk_condition_file
from core.rag.ingestor import embed_and_upsert
from core.rag.bm25_index import build_corpus_artifact

RAW_DIR = Path("data/knowledge_base/raw/statpearls")
CHUNKS_DIR = Path("data/knowledge_base/processed/chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Run full ingestion pipeline for all available condition files."""
    index = Pinecone(api_key=settings.pinecone_api_key).Index(settings.pinecone_index_name)
    total = 0
    all_chunks: list[dict] = []

    for condition, meta in CONDITION_MAPPING.items():
        filepath = RAW_DIR / f"{condition}.txt"

        if not filepath.exists():
            print(f"skip    {condition} (file not found)")
            continue

        source = "statpearls" if meta["fetch"] == "api" else "manual"
        chunks = chunk_condition_file(filepath, condition, source=source)

        # Persist chunks as JSONL for auditability and re-upsert without re-chunking
        (CHUNKS_DIR / f"{condition}.jsonl").write_text(
            "\n".join(json.dumps(c) for c in chunks), encoding="utf-8"
        )

        n = embed_and_upsert(chunks, index, namespace=settings.pinecone_namespace)
        print(f"upserted {condition:<20} {n:>3} vectors  [{source}]")
        total += n
        all_chunks.extend(chunks)

    # Written once, from the same chunks that were embedded, so the lexical half can never
    # describe a corpus the dense half does not contain.
    written = build_corpus_artifact(all_chunks)
    print(f"lexical corpus  {written:>3} chunks -> {settings.bm25_corpus_path}")

    print(f"done. total {total} vectors upserted.")


if __name__ == "__main__":
    main()