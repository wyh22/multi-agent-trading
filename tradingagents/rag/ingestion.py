from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tradingagents.rag.scope import normalize_scope
from tradingagents.rag.chunking import chunk_document
from tradingagents.rag.embeddings import build_embedder
from tradingagents.rag.loaders import load_documents
from tradingagents.rag.store import QdrantKnowledgeStore


def ingest_path(
    path: str | Path,
    *,
    ticker: str | None,
    publish_date: str,
    config: dict,
    scope_type: str = "company",
    scope_key: str | None = None,
    industry: str | None = None,
    doc_type: str = "user_document",
    chunk_chars: int = 900,
    overlap_chars: int = 120,
    source_name: str | None = None,
    publish_date_source: str = "USER",
    publish_date_confidence: float = 0.5,
    publish_date_verified: bool = False,
    source_authority: str | None = None,
    source_url: str | None = None,
) -> dict:
    """Parse, chunk and upsert a user/company document into the configured RAG store."""

    scope = normalize_scope(
        scope_type=scope_type,
        scope_key=scope_key,
        ticker=ticker,
        industry=industry,
    )
    docs = load_documents(
        path,
        ticker=scope.ticker,
        publish_date=publish_date,
        doc_type=doc_type,
        source_name=source_name,
        publish_date_source=publish_date_source,
        publish_date_confidence=publish_date_confidence,
        publish_date_verified=publish_date_verified,
        source_authority=source_authority,
        source_url=source_url,
    )
    docs = [
        replace(
            doc,
            scope_type=scope.scope_type,
            scope_key=scope.scope_key,
            industry=scope.industry,
        )
        for doc in docs
    ]
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
        "ticker": scope.ticker,
        "scope_type": scope.scope_type,
        "scope_key": scope.scope_key,
        "scope_id": scope.scope_id,
        "industry": scope.industry,
        "documents": len(docs),
        "chunks": count,
        "file_hashes": hashes,
        "collection": store.collection,
        "publish_date_provenance": {
            "source": publish_date_source,
            "confidence": float(publish_date_confidence),
            "verified": bool(publish_date_verified),
            "authority": source_authority or "",
            "url": source_url or "",
        },
    }
