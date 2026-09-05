from __future__ import annotations

from pathlib import Path

from tradingagents.dataflows.symbol_utils import normalize_a_share_symbol
from tradingagents.rag.chunking import chunk_document
from tradingagents.rag.embeddings import build_embedder
from tradingagents.rag.loaders import load_documents
from tradingagents.rag.store import QdrantKnowledgeStore


def ingest_path(
    path: str | Path,
    *,
    ticker: str,
    publish_date: str,
    config: dict,
    doc_type: str = "user_document",
    chunk_chars: int = 900,
    overlap_chars: int = 120,
    source_name: str | None = None,
) -> dict:
    """Parse, chunk and upsert a user/company document into the configured RAG store."""

    canonical = normalize_a_share_symbol(ticker)
    docs = load_documents(
        path,
        ticker=canonical,
        publish_date=publish_date,
        doc_type=doc_type,
        source_name=source_name,
    )
    chunks = []
    for doc in docs:
        chunks.extend(
            chunk_document(
                doc,
                target_chars=chunk_chars,
                overlap_chars=overlap_chars,
            )
        )
    embedder = build_embedder(config)
    store = QdrantKnowledgeStore(
        url=str(config.get("qdrant_url", "http://localhost:6333")),
        collection=str(config.get("qdrant_collection", "a_share_knowledge")),
        embedder=embedder,
        api_key=config.get("qdrant_api_key") or None,
    )
    count = store.upsert_chunks(chunks)
    hashes = sorted(
        {
            str(doc.metadata.get("file_hash", ""))
            for doc in docs
            if doc.metadata.get("file_hash")
        }
    )
    return {
        "ticker": canonical,
        "documents": len(docs),
        "chunks": count,
        "file_hashes": hashes,
        "collection": store.collection,
    }
