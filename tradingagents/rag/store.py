from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable

from .models import KnowledgeChunk

logger = logging.getLogger(__name__)


class QdrantKnowledgeStore:
    """Qdrant dense-vector store with PIT metadata filters."""

    def __init__(
        self,
        *,
        url: str,
        collection: str,
        embedder=None,
        api_key: str | None = None,
    ):
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("qdrant-client 未安装，请执行 `pip install -e '.[agent]'`") from exc
        self.url = url
        self.collection = collection
        self.embedder = embedder
        self._client = QdrantClient(url=url, api_key=api_key or None, timeout=15)

    def ensure_collection(self) -> None:
        from qdrant_client import models

        try:
            exists = self._client.collection_exists(self.collection)
        except AttributeError:  # pragma: no cover - old client compatibility
            try:
                self._client.get_collection(self.collection)
                exists = True
            except Exception:
                exists = False
        if not exists:
            if self.embedder is None:
                raise RuntimeError(
                    "collection does not exist and no embedder was supplied"
                )
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=self.embedder.dimension, distance=models.Distance.COSINE),
            )
        for field, schema in [
            ("ticker", models.PayloadSchemaType.KEYWORD),
            ("publish_date", models.PayloadSchemaType.DATETIME),
            ("doc_type", models.PayloadSchemaType.KEYWORD),
        ]:
            try:
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                pass

    def upsert_chunks(self, chunks: Iterable[KnowledgeChunk], batch_size: int = 64) -> int:
        from qdrant_client import models

        if self.embedder is None:
            raise RuntimeError("upsert requires an embedding backend")
        self.ensure_collection()
        batch: list[KnowledgeChunk] = []
        total = 0

        def flush(rows: list[KnowledgeChunk]):
            nonlocal total
            if not rows:
                return
            vectors = self.embedder.embed([r.text for r in rows])
            points = []
            for row, vector in zip(rows, vectors, strict=True):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, row.chunk_id))
                points.append(models.PointStruct(id=point_id, vector=vector, payload=row.to_payload()))
            self._client.upsert(collection_name=self.collection, points=points, wait=True)
            total += len(rows)

        for chunk in chunks:
            batch.append(chunk)
            if len(batch) >= batch_size:
                flush(batch)
                batch = []
        flush(batch)
        return total

    @staticmethod
    def _filter(ticker: str, as_of_date: str, doc_type: str | None = None):
        from qdrant_client import models

        must = [
            models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker)),
            models.FieldCondition(
                key="publish_date",
                range=models.DatetimeRange(lte=as_of_date[:10] + "T23:59:59Z"),
            ),
        ]
        if doc_type:
            must.append(models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type)))
        return models.Filter(must=must)

    def query_dense(
        self, query: str, *, ticker: str, as_of_date: str, limit: int = 20, doc_type: str | None = None
    ) -> list[tuple[KnowledgeChunk, float]]:
        if self.embedder is None:
            raise RuntimeError("dense query requires an embedding backend")
        vector = self.embedder.embed([query])[0]
        result = self._client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=self._filter(ticker, as_of_date, doc_type),
            with_payload=True,
            limit=max(1, int(limit)),
        ).points
        rows = []
        for hit in result:
            if not hit.payload:
                continue
            rows.append((KnowledgeChunk.from_payload(dict(hit.payload)), float(hit.score)))
        return rows

    def scroll_chunks(
        self, *, ticker: str, as_of_date: str, limit: int = 1000, doc_type: str | None = None
    ) -> list[KnowledgeChunk]:
        rows: list[KnowledgeChunk] = []
        offset = None
        remaining = max(1, int(limit))
        while remaining > 0:
            points, offset = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=self._filter(ticker, as_of_date, doc_type),
                with_payload=True,
                with_vectors=False,
                limit=min(256, remaining),
                offset=offset,
            )
            for point in points:
                if point.payload:
                    rows.append(KnowledgeChunk.from_payload(dict(point.payload)))
            remaining = limit - len(rows)
            if offset is None or not points:
                break
        return rows[:limit]


    def collection_status(self) -> dict:
        """Return lightweight Qdrant readiness information without loading embeddings."""

        try:
            exists = bool(self._client.collection_exists(self.collection))
        except AttributeError:  # pragma: no cover - old client compatibility
            try:
                self._client.get_collection(self.collection)
                exists = True
            except Exception:
                exists = False
        if not exists:
            return {
                "collection": self.collection,
                "exists": False,
                "points_count": 0,
            }

        info = self._client.get_collection(self.collection)
        return {
            "collection": self.collection,
            "exists": True,
            "points_count": int(getattr(info, "points_count", 0) or 0),
            "indexed_vectors_count": int(
                getattr(info, "indexed_vectors_count", 0) or 0
            ),
        }

    def list_documents(
        self,
        *,
        ticker: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """List parent documents and provenance metadata from stored chunks."""

        from qdrant_client import models

        query_filter = None
        if ticker:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="ticker",
                        match=models.MatchValue(value=ticker),
                    )
                ]
            )

        documents: dict[str, dict] = {}
        offset = None
        scanned = 0
        max_scan = max(512, min(20000, int(limit) * 200))
        while len(documents) < max(1, int(limit)) and scanned < max_scan:
            points, offset = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                limit=min(256, max_scan - scanned),
                offset=offset,
            )
            scanned += len(points)
            for point in points:
                if not point.payload:
                    continue
                chunk = KnowledgeChunk.from_payload(dict(point.payload))
                meta = chunk.metadata or {}
                parent_key = str(meta.get("file_hash") or chunk.doc_id)
                row = documents.setdefault(
                    parent_key,
                    {
                        "document_key": parent_key,
                        "ticker": chunk.ticker,
                        "title": str(meta.get("file_name") or chunk.title),
                        "publish_date": chunk.publish_date,
                        "doc_type": chunk.doc_type,
                        "source": str(
                            meta.get("source_authority") or chunk.source
                        ),
                        "url": str(meta.get("source_url") or chunk.url),
                        "publish_date_source": str(
                            meta.get("publish_date_source") or ""
                        ),
                        "publish_date_confidence": meta.get(
                            "publish_date_confidence"
                        ),
                        "publish_date_verified": meta.get(
                            "publish_date_verified"
                        ),
                        "chunk_count": 0,
                    },
                )
                row["chunk_count"] += 1
                if chunk.publish_date > row["publish_date"]:
                    row["publish_date"] = chunk.publish_date
            if offset is None or not points:
                break

        rows = list(documents.values())
        rows.sort(
            key=lambda row: (
                str(row.get("publish_date") or ""),
                str(row.get("title") or ""),
            ),
            reverse=True,
        )
        return rows[: max(1, int(limit))]
