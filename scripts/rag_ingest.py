"""Ingest PIT-aware company documents into Qdrant."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.dataflows.config import get_config
from tradingagents.rag.scope import normalize_knowledge_scope
from tradingagents.rag.chunking import chunk_document
from tradingagents.rag.embeddings import build_embedder
from tradingagents.rag.ingestion import ingest_path
from tradingagents.rag.models import KnowledgeDocument
from tradingagents.rag.store import QdrantKnowledgeStore


def _provenance_metadata(row: dict) -> dict:
    metadata = dict(row.get("metadata", {}) or {})
    source = str(
        row.get("publish_date_source")
        or metadata.get("publish_date_source")
        or "USER"
    ).upper()
    confidence = float(
        row.get(
            "publish_date_confidence",
            metadata.get("publish_date_confidence", 0.5),
        )
    )
    verified = bool(
        row.get(
            "publish_date_verified",
            metadata.get("publish_date_verified", False),
        )
    )
    metadata.update(
        {
            "publish_date_source": source,
            "publish_date_confidence": max(0.0, min(1.0, confidence)),
            "publish_date_verified": verified,
            "source_authority": str(
                row.get("source_authority")
                or metadata.get("source_authority")
                or row.get("source")
                or ""
            ),
            "source_url": str(
                row.get("source_url")
                or metadata.get("source_url")
                or row.get("url")
                or ""
            ),
        }
    )
    return metadata


def load_jsonl(path: Path):
    docs = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        docs.append(
            KnowledgeDocument(
                doc_id=str(row.get("doc_id") or f"{path.stem}-{line_no}"),
                ticker=normalize_knowledge_scope(str(row["ticker"])),
                title=str(row.get("title", "")),
                text=str(row["text"]),
                publish_date=str(row["publish_date"]),
                source=str(row.get("source", "local")),
                url=str(row.get("url", "")),
                doc_type=str(row.get("doc_type", "document")),
                metadata=_provenance_metadata(row),
            )
        )
    return docs


def load_directory(
    path: Path,
    ticker: str,
    publish_date: str,
    doc_type: str,
):
    docs = []
    for item in sorted(path.rglob("*")):
        if item.suffix.lower() not in {".txt", ".md"} or not item.is_file():
            continue
        docs.append(
            KnowledgeDocument(
                doc_id=f"{ticker}:{item.relative_to(path)}",
                ticker=normalize_knowledge_scope(ticker),
                title=item.stem,
                text=item.read_text(encoding="utf-8", errors="ignore"),
                publish_date=publish_date,
                source="local-file",
                url=str(item),
                doc_type=doc_type,
            )
        )
    return docs


def _legacy_ingest(docs, args, config):
    embedder = build_embedder(config)
    store = QdrantKnowledgeStore(
        url=str(config.get("qdrant_url", "http://localhost:6333")),
        collection=str(config.get("qdrant_collection", "a_share_knowledge")),
        embedder=embedder,
        api_key=config.get("qdrant_api_key") or None,
    )
    chunks = []
    for doc in docs:
        chunks.extend(
            chunk_document(
                doc,
                target_chars=args.chunk_chars,
                overlap_chars=args.overlap_chars,
            )
        )
    count = store.upsert_chunks(chunks)
    print(
        f"documents={len(docs)} chunks={count} collection={store.collection}"
    )




def _download_url(url: str, *, max_mb: int = 50) -> Path:
    import requests

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("--url 仅支持 http/https")
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".md", ".markdown"}:
        suffix = ".pdf"

    response = requests.get(
        url,
        stream=True,
        timeout=(10, 60),
        headers={"User-Agent": "TradingAgents-RAG/1.0"},
    )
    response.raise_for_status()
    max_bytes = max(1, int(max_mb)) * 1024 * 1024
    total = 0
    handle = tempfile.NamedTemporaryFile(
        suffix=suffix,
        prefix="tradingagents_rag_url_",
        delete=False,
    )
    try:
        with handle:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if not block:
                    continue
                total += len(block)
                if total > max_bytes:
                    raise ValueError(
                        f"URL document exceeds {max_mb} MB safety limit"
                    )
                handle.write(block)
        return Path(handle.name)
    except Exception:
        Path(handle.name).unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="向 Qdrant 写入 A 股公告/财报/用户文档"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl", type=Path)
    src.add_argument("--directory", type=Path)
    src.add_argument(
        "--file",
        type=Path,
        help="单个 PDF/DOCX/TXT/MD 文档",
    )
    src.add_argument(
        "--url",
        help="直接下载并入库一个官方 HTTP/HTTPS PDF/DOCX/TXT/MD URL",
    )
    parser.add_argument("--ticker")
    parser.add_argument("--publish-date")
    parser.add_argument("--doc-type", default="document")
    parser.add_argument(
        "--publish-date-source",
        default="USER",
        help="披露日来源，例如 USER/CNINFO/SSE/SZSE/COMPANY_IR",
    )
    parser.add_argument(
        "--publish-date-confidence",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--publish-date-verified",
        action="store_true",
        help="仅在你已核验官方披露日时使用；历史 PIT 检索依赖该标记",
    )
    parser.add_argument(
        "--source-authority",
        default="",
        help="原始权威来源，例如 CNINFO/SSE/SZSE/COMPANY_IR",
    )
    parser.add_argument(
        "--source-url",
        default="",
        help="原始公告/财报 URL；不要填写临时下载地址",
    )
    parser.add_argument("--chunk-chars", type=int, default=900)
    parser.add_argument("--overlap-chars", type=int, default=120)
    args = parser.parse_args()

    config = get_config()

    if args.url:
        if not args.ticker or not args.publish_date:
            parser.error("--url 模式必须提供 --ticker 和 --publish-date")
        temp_path = _download_url(args.url)
        try:
            result = ingest_path(
                temp_path,
                ticker=args.ticker,
                publish_date=args.publish_date,
                config=config,
                doc_type=args.doc_type,
                chunk_chars=args.chunk_chars,
                overlap_chars=args.overlap_chars,
                source_name=Path(urlparse(args.url).path).name or "remote-document",
                publish_date_source=args.publish_date_source,
                publish_date_confidence=args.publish_date_confidence,
                publish_date_verified=args.publish_date_verified,
                source_authority=args.source_authority or None,
                source_url=args.source_url or args.url,
            )
        finally:
            temp_path.unlink(missing_ok=True)
        print(
            f"documents={result['documents']} chunks={result['chunks']} "
            f"collection={result['collection']} source={args.url}"
        )
        return

    if args.file:
        if not args.ticker or not args.publish_date:
            parser.error("--file 模式必须提供 --ticker 和 --publish-date")
        result = ingest_path(
            args.file,
            ticker=args.ticker,
            publish_date=args.publish_date,
            config=config,
            doc_type=args.doc_type,
            chunk_chars=args.chunk_chars,
            overlap_chars=args.overlap_chars,
            publish_date_source=args.publish_date_source,
            publish_date_confidence=args.publish_date_confidence,
            publish_date_verified=args.publish_date_verified,
            source_authority=args.source_authority or None,
            source_url=args.source_url or None,
        )
        print(
            f"documents={result['documents']} chunks={result['chunks']} "
            f"collection={result['collection']}"
        )
        return

    if args.jsonl:
        docs = load_jsonl(args.jsonl)
    else:
        if not args.ticker or not args.publish_date:
            parser.error("--directory 模式必须提供 --ticker 和 --publish-date")
        docs = load_directory(
            args.directory,
            args.ticker,
            args.publish_date,
            args.doc_type,
        )
    _legacy_ingest(docs, args, config)


if __name__ == "__main__":
    main()
