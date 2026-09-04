"""Ingest PIT-aware company documents into Qdrant."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.symbol_utils import normalize_a_share_symbol
from tradingagents.rag.chunking import chunk_document
from tradingagents.rag.embeddings import build_embedder
from tradingagents.rag.models import KnowledgeDocument
from tradingagents.rag.store import QdrantKnowledgeStore


def load_jsonl(path: Path):
    docs = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        docs.append(KnowledgeDocument(
            doc_id=str(row.get("doc_id") or f"{path.stem}-{line_no}"),
            ticker=normalize_a_share_symbol(str(row["ticker"])),
            title=str(row.get("title", "")), text=str(row["text"]),
            publish_date=str(row["publish_date"]), source=str(row.get("source", "local")),
            url=str(row.get("url", "")), doc_type=str(row.get("doc_type", "document")),
            metadata=dict(row.get("metadata", {}) or {}),
        ))
    return docs


def load_directory(path: Path, ticker: str, publish_date: str, doc_type: str):
    docs = []
    for p in sorted(path.rglob("*")):
        if p.suffix.lower() not in {".txt", ".md"} or not p.is_file():
            continue
        docs.append(KnowledgeDocument(
            doc_id=f"{ticker}:{p.relative_to(path)}", ticker=normalize_a_share_symbol(ticker),
            title=p.stem, text=p.read_text(encoding="utf-8", errors="ignore"),
            publish_date=publish_date, source="local-file", url=str(p), doc_type=doc_type,
        ))
    return docs


def main():
    parser = argparse.ArgumentParser(description="向 Qdrant 写入 A 股公告/财报知识片段")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl", type=Path)
    src.add_argument("--directory", type=Path)
    parser.add_argument("--ticker")
    parser.add_argument("--publish-date")
    parser.add_argument("--doc-type", default="document")
    parser.add_argument("--chunk-chars", type=int, default=900)
    parser.add_argument("--overlap-chars", type=int, default=120)
    args = parser.parse_args()

    if args.jsonl:
        docs = load_jsonl(args.jsonl)
    else:
        if not args.ticker or not args.publish_date:
            parser.error("--directory 模式必须提供 --ticker 和 --publish-date")
        docs = load_directory(args.directory, args.ticker, args.publish_date, args.doc_type)

    config = get_config()
    embedder = build_embedder(config)
    store = QdrantKnowledgeStore(
        url=str(config.get("qdrant_url", "http://localhost:6333")),
        collection=str(config.get("qdrant_collection", "a_share_knowledge")),
        embedder=embedder, api_key=config.get("qdrant_api_key") or None,
    )
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc, target_chars=args.chunk_chars, overlap_chars=args.overlap_chars))
    count = store.upsert_chunks(chunks)
    print(f"documents={len(docs)} chunks={count} collection={store.collection}")


if __name__ == "__main__":
    main()
