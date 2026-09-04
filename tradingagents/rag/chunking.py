from __future__ import annotations

import re

from .models import KnowledgeChunk, KnowledgeDocument


def _paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in re.split(r"\n\s*\n+", normalized) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in normalized.split("\n") if p.strip()]
    return parts or [normalized.strip()]


def chunk_document(
    document: KnowledgeDocument,
    *,
    target_chars: int = 900,
    overlap_chars: int = 120,
) -> list[KnowledgeChunk]:
    """Paragraph-aware chunking with a small trailing overlap."""
    if target_chars < 200:
        raise ValueError("target_chars 至少为 200")
    if overlap_chars < 0 or overlap_chars >= target_chars:
        raise ValueError("overlap_chars 必须 >=0 且小于 target_chars")

    paragraphs = _paragraphs(document.text)
    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        para_parts = [para[i:i + target_chars] for i in range(0, len(para), target_chars)] or [para]
        for part in para_parts:
            candidate = (current + "\n" + part).strip() if current else part
            if current and len(candidate) > target_chars:
                pieces.append(current.strip())
                prefix = current[-overlap_chars:] if overlap_chars else ""
                current = (prefix + "\n" + part).strip()
            else:
                current = candidate
    if current.strip():
        pieces.append(current.strip())

    chunks: list[KnowledgeChunk] = []
    for idx, text in enumerate(pieces):
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"{document.doc_id}::chunk-{idx}",
                doc_id=document.doc_id,
                ticker=document.ticker,
                title=document.title,
                text=text,
                publish_date=document.publish_date,
                source=document.source,
                url=document.url,
                doc_type=document.doc_type,
                chunk_index=idx,
                metadata=document.metadata,
            )
        )
    return chunks
