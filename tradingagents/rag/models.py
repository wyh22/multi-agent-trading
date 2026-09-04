from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


def normalize_publish_date(value: str) -> str:
    """Normalize supported date/datetime strings to YYYY-MM-DD."""
    raw = str(value).strip()
    if not raw:
        raise ValueError("publish_date 不能为空")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.date().isoformat()
    except ValueError:
        try:
            return date.fromisoformat(raw[:10]).isoformat()
        except ValueError as exc:
            raise ValueError(f"无法解析 publish_date: {value!r}") from exc


@dataclass(frozen=True)
class KnowledgeDocument:
    doc_id: str
    ticker: str
    title: str
    text: str
    publish_date: str
    source: str = "local"
    url: str = ""
    doc_type: str = "document"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "publish_date", normalize_publish_date(self.publish_date))
        if not self.doc_id.strip():
            raise ValueError("doc_id 不能为空")
        if not self.ticker.strip():
            raise ValueError("ticker 不能为空")
        if not self.text.strip():
            raise ValueError("text 不能为空")


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    doc_id: str
    ticker: str
    title: str
    text: str
    publish_date: str
    source: str = "local"
    url: str = ""
    doc_type: str = "document"
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "publish_date", normalize_publish_date(self.publish_date))

    def to_payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "ticker": self.ticker,
            "title": self.title,
            "text": self.text,
            "publish_date": self.publish_date + "T00:00:00Z",
            "source": self.source,
            "url": self.url,
            "doc_type": self.doc_type,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "KnowledgeChunk":
        return cls(
            chunk_id=str(payload.get("chunk_id", "")),
            doc_id=str(payload.get("doc_id", "")),
            ticker=str(payload.get("ticker", "")),
            title=str(payload.get("title", "")),
            text=str(payload.get("text", "")),
            publish_date=str(payload.get("publish_date", ""))[:10],
            source=str(payload.get("source", "local")),
            url=str(payload.get("url", "")),
            doc_type=str(payload.get("doc_type", "document")),
            chunk_index=int(payload.get("chunk_index", 0) or 0),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class RetrievalHit:
    chunk: KnowledgeChunk
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
