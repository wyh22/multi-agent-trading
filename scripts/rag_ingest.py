"""Build and update the project-wide PIT-aware RAG corpus."""

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
from tradingagents.rag.ingestion import (
    ingest_documents,
    prepare_path_documents,
)
from tradingagents.rag.models import KnowledgeDocument
from tradingagents.rag.scope import normalize_scope


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown"}


def _download_url(url: str, *, max_mb: int = 50) -> Path:
    import requests

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL ingestion only supports http/https")

    suffix = Path(parsed.path).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
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
                        f"remote document exceeds {max_mb} MB safety limit"
                    )
                handle.write(block)
        return Path(handle.name)
    except Exception:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _row_scope(row: dict):
    return normalize_scope(
        scope_type=str(row.get("scope_type") or "company"),
        scope_key=(
            str(row.get("scope_key")).strip()
            if row.get("scope_key") not in (None, "")
            else None
        ),
        ticker=(
            str(row.get("ticker")).strip()
            if row.get("ticker") not in (None, "")
            else None
        ),
        industry=(
            str(row.get("industry")).strip()
            if row.get("industry") not in (None, "")
            else None
        ),
    )


def _row_provenance(row: dict) -> dict:
    metadata = dict(row.get("metadata", {}) or {})
    publish_date_source = str(
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
    verified = _as_bool(
        row.get(
            "publish_date_verified",
            metadata.get("publish_date_verified", False),
        )
    )
    source_authority = str(
        row.get("source_authority")
        or metadata.get("source_authority")
        or row.get("source")
        or ""
    )
    source_url = str(
        row.get("source_url")
        or metadata.get("source_url")
        or row.get("url")
        or ""
    )
    metadata.update(
        {
            "publish_date_source": publish_date_source,
            "publish_date_confidence": max(
                0.0,
                min(1.0, confidence),
            ),
            "publish_date_verified": verified,
            "source_authority": source_authority,
            "source_url": source_url,
        }
    )
    return {
        "metadata": metadata,
        "publish_date_source": publish_date_source,
        "publish_date_confidence": confidence,
        "publish_date_verified": verified,
        "source_authority": source_authority,
        "source_url": source_url,
    }


def _prepare_text_document(
    row: dict,
    *,
    fallback_doc_id: str,
) -> KnowledgeDocument:
    scope = _row_scope(row)
    provenance = _row_provenance(row)
    text = str(row.get("text") or "").strip()
    if not text:
        raise ValueError("manifest text row is empty")
    publish_date = str(row.get("publish_date") or "").strip()
    if not publish_date:
        raise ValueError("manifest row missing publish_date")

    return KnowledgeDocument(
        doc_id=str(row.get("doc_id") or fallback_doc_id),
        ticker=scope.ticker,
        title=str(row.get("title") or fallback_doc_id),
        text=text,
        publish_date=publish_date,
        source=str(
            provenance["source_authority"]
            or row.get("source")
            or "manifest"
        ),
        url=str(provenance["source_url"]),
        doc_type=str(row.get("doc_type") or "document"),
        scope_type=scope.scope_type,
        scope_key=scope.scope_key,
        industry=scope.industry,
        metadata=provenance["metadata"],
    )


def _prepare_path_row(
    row: dict,
    *,
    path: Path,
    source_name: str | None = None,
) -> list[KnowledgeDocument]:
    scope = _row_scope(row)
    provenance = _row_provenance(row)
    publish_date = str(row.get("publish_date") or "").strip()
    if not publish_date:
        raise ValueError("manifest row missing publish_date")

    return prepare_path_documents(
        path,
        ticker=scope.ticker if scope.scope_type == "company" else None,
        publish_date=publish_date,
        scope_type=scope.scope_type,
        scope_key=scope.scope_key,
        industry=scope.industry,
        doc_type=str(row.get("doc_type") or "document"),
        source_name=source_name,
        publish_date_source=provenance["publish_date_source"],
        publish_date_confidence=provenance[
            "publish_date_confidence"
        ],
        publish_date_verified=provenance[
            "publish_date_verified"
        ],
        source_authority=provenance["source_authority"] or None,
        source_url=provenance["source_url"] or None,
    )


def load_manifest(
    path: Path,
    *,
    max_download_mb: int = 50,
) -> list[KnowledgeDocument]:
    """Load arbitrary company/industry/market/macro/regulation documents."""

    docs: list[KnowledgeDocument] = []
    root = path.resolve().parent
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        sources = [
            key
            for key in ("text", "file", "url")
            if str(row.get(key) or "").strip()
        ]
        if len(sources) != 1:
            raise ValueError(
                f"{path}:{line_no} requires exactly one of text/file/url"
            )

        fallback_doc_id = f"{path.stem}-{line_no}"
        if sources[0] == "text":
            docs.append(
                _prepare_text_document(
                    row,
                    fallback_doc_id=fallback_doc_id,
                )
            )
            continue

        if sources[0] == "file":
            local_path = Path(str(row["file"])).expanduser()
            if not local_path.is_absolute():
                local_path = root / local_path
            docs.extend(
                _prepare_path_row(
                    row,
                    path=local_path,
                    source_name=str(
                        row.get("title") or local_path.name
                    ),
                )
            )
            continue

        url = str(row["url"]).strip()
        temp_path = _download_url(
            url,
            max_mb=max_download_mb,
        )
        try:
            row = dict(row)
            row.setdefault("source_url", url)
            docs.extend(
                _prepare_path_row(
                    row,
                    path=temp_path,
                    source_name=(
                        str(row.get("title")).strip()
                        if row.get("title")
                        else Path(urlparse(url).path).name
                        or fallback_doc_id
                    ),
                )
            )
        finally:
            temp_path.unlink(missing_ok=True)
    return docs


def _single_row_from_args(args) -> dict:
    row = {
        "ticker": args.ticker,
        "scope_type": args.scope_type,
        "scope_key": args.scope_key,
        "industry": args.industry,
        "publish_date": args.publish_date,
        "doc_type": args.doc_type,
        "publish_date_source": args.publish_date_source,
        "publish_date_confidence": args.publish_date_confidence,
        "publish_date_verified": args.publish_date_verified,
        "source_authority": args.source_authority,
        "source_url": args.source_url,
    }
    # Validate before any remote download.
    _row_scope(row)
    if not args.publish_date:
        raise ValueError("--publish-date is required")
    return row


def _directory_documents(args) -> list[KnowledgeDocument]:
    row = _single_row_from_args(args)
    docs: list[KnowledgeDocument] = []
    for item in sorted(args.directory.rglob("*")):
        if not item.is_file() or item.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        docs.extend(
            _prepare_path_row(
                row,
                path=item,
                source_name=item.name,
            )
        )
    if not docs:
        raise ValueError("directory contains no supported documents")
    return docs


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a project-wide PIT-aware RAG corpus for A-share research"
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--manifest",
        type=Path,
        help=(
            "JSONL manifest; each row can use text/file/url and may target "
            "company/industry/market/macro/regulation scope"
        ),
    )
    src.add_argument(
        "--jsonl",
        type=Path,
        help="Backward-compatible alias for a text-based manifest",
    )
    src.add_argument("--directory", type=Path)
    src.add_argument("--file", type=Path)
    src.add_argument("--url")

    parser.add_argument("--ticker")
    parser.add_argument(
        "--scope-type",
        default="company",
        choices=[
            "company",
            "industry",
            "market",
            "macro",
            "regulation",
        ],
    )
    parser.add_argument("--scope-key")
    parser.add_argument("--industry")
    parser.add_argument("--publish-date")
    parser.add_argument("--doc-type", default="document")
    parser.add_argument("--publish-date-source", default="USER")
    parser.add_argument(
        "--publish-date-confidence",
        type=float,
        default=0.5,
    )
    parser.add_argument("--publish-date-verified", action="store_true")
    parser.add_argument("--source-authority", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--chunk-chars", type=int, default=900)
    parser.add_argument("--overlap-chars", type=int, default=120)
    parser.add_argument("--max-download-mb", type=int, default=50)
    args = parser.parse_args()

    config = get_config()

    if args.manifest or args.jsonl:
        docs = load_manifest(
            args.manifest or args.jsonl,
            max_download_mb=args.max_download_mb,
        )
    elif args.directory:
        docs = _directory_documents(args)
    elif args.file:
        row = _single_row_from_args(args)
        docs = _prepare_path_row(
            row,
            path=args.file,
            source_name=args.file.name,
        )
    else:
        row = _single_row_from_args(args)
        url = str(args.url)
        row["source_url"] = args.source_url or url
        temp_path = _download_url(
            url,
            max_mb=args.max_download_mb,
        )
        try:
            docs = _prepare_path_row(
                row,
                path=temp_path,
                source_name=(
                    Path(urlparse(url).path).name
                    or "remote-document"
                ),
            )
        finally:
            temp_path.unlink(missing_ok=True)

    result = ingest_documents(
        docs,
        config=config,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
    )
    print(
        f"documents={result['documents']} chunks={result['chunks']} "
        f"scopes={len(result['scope_ids'])} "
        f"collection={result['collection']}"
    )
    for scope_id in result["scope_ids"]:
        print(f"  - {scope_id}")


if __name__ == "__main__":
    main()
