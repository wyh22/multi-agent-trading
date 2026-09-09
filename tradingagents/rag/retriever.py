from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import date
from typing import Iterable

from .embeddings import HashEmbedding, build_embedder, build_reranker
from .evidence_pack import source_document_key
from .models import KnowledgeChunk, RetrievalHit
from .scope import (
    company_scope_id,
    default_shared_scope_ids,
    industry_scope_id,
)
from .store import QdrantKnowledgeStore


def tokenize_for_bm25(text: str) -> list[str]:
    text = text.lower()
    words = re.findall(r"[a-z0-9_\.-]+|[\u4e00-\u9fff]", text)
    chinese = [w for w in words if len(w) == 1 and "\u4e00" <= w <= "\u9fff"]
    bigrams = ["".join(chinese[i:i + 2]) for i in range(max(0, len(chinese) - 1))]
    return words + bigrams


def bm25_scores(query: str, chunks: list[KnowledgeChunk], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not chunks:
        return []
    docs = [tokenize_for_bm25(c.title + " " + c.text) for c in chunks]
    q = tokenize_for_bm25(query)
    n = len(docs)
    avgdl = sum(map(len, docs)) / max(1, n)
    df = Counter()
    for doc in docs:
        df.update(set(doc))
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in set(q)}
    scores = []
    for doc in docs:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in q:
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + k1 * (1 - b + b * dl / max(avgdl, 1e-9))
            score += idf.get(term, 0.0) * (f * (k1 + 1)) / denom
        scores.append(score)
    return scores


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _publication_date_is_pit_safe(chunk: KnowledgeChunk, cutoff: date) -> bool:
    """Fail closed for explicitly-unverified dates in historical research.

    Legacy indexed documents without the provenance field are kept compatible;
    newly uploaded user documents explicitly store publish_date_verified=False.
    """
    if cutoff >= date.today():
        return True
    verified = (chunk.metadata or {}).get("publish_date_verified")
    return verified is not False


class InMemoryKnowledgeStore:
    """Dependency-free store used by tests and small local demos."""

    def __init__(self, chunks: Iterable[KnowledgeChunk], embedder=None):
        self.chunks = list(chunks)
        self.embedder = embedder or HashEmbedding()
        self._vectors = self.embedder.embed([c.text for c in self.chunks]) if self.chunks else []

    def _eligible(
        self,
        scope_ids: list[str],
        as_of_date: str,
        doc_type: str | None = None,
        legacy_ticker: str | None = None,
    ):
        cutoff = date.fromisoformat(as_of_date[:10])
        scopes = {str(item) for item in scope_ids if str(item)}
        out = []
        for i, c in enumerate(self.chunks):
            scope_id = f"{c.scope_type}:{c.scope_key}"
            if scope_id not in scopes and not (
                legacy_ticker and c.ticker == legacy_ticker
            ):
                continue
            if date.fromisoformat(c.publish_date) > cutoff:
                continue
            if not _publication_date_is_pit_safe(c, cutoff):
                continue
            if doc_type and c.doc_type != doc_type:
                continue
            out.append((i, c))
        return out

    def query_dense(
        self,
        query: str,
        *,
        scope_ids: list[str],
        as_of_date: str,
        limit: int = 20,
        doc_type=None,
        legacy_ticker: str | None = None,
    ):
        qv = self.embedder.embed([query])[0]
        scored = [
            (c, _cosine(qv, self._vectors[i]))
            for i, c in self._eligible(
                scope_ids,
                as_of_date,
                doc_type,
                legacy_ticker=legacy_ticker,
            )
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def scroll_chunks(
        self,
        *,
        scope_ids: list[str],
        as_of_date: str,
        limit: int = 1000,
        doc_type=None,
        legacy_ticker: str | None = None,
    ):
        return [
            c
            for _, c in self._eligible(
                scope_ids,
                as_of_date,
                doc_type,
                legacy_ticker=legacy_ticker,
            )
        ][:limit]

    def resolve_industries(self, ticker: str, limit: int = 8) -> list[str]:
        seen: list[str] = []
        for chunk in self.chunks:
            if chunk.ticker != ticker:
                continue
            industry = str(chunk.industry or "").strip()
            if industry and industry not in seen:
                seen.append(industry)
            if len(seen) >= max(1, int(limit)):
                break
        return seen


class HybridKnowledgeRetriever:
    """Dense + BM25 + RRF + optional cross-encoder reranking with PIT filtering."""

    def __init__(self, store, *, reranker=None, rrf_k: int = 60):
        self.store = store
        self.reranker = reranker
        self.rrf_k = rrf_k

    @classmethod
    def from_config(cls, config: dict):
        embedder = build_embedder(config)
        store = QdrantKnowledgeStore(
            url=str(config.get("qdrant_url", "http://localhost:6333")),
            collection=str(config.get("qdrant_collection", "a_share_knowledge")),
            embedder=embedder,
            api_key=config.get("qdrant_api_key") or None,
        )
        return cls(store, reranker=build_reranker(config))

    def search(
        self,
        query: str,
        *,
        ticker: str,
        as_of_date: str,
        top_k: int = 6,
        candidate_k: int = 30,
        corpus_limit: int = 1000,
        doc_type: str | None = None,
        max_chunks_per_doc: int = 2,
        industry: str | None = None,
        include_shared_scopes: bool = True,
        strategy: str = "hybrid",
        use_reranker: bool | None = None,
    ) -> list[RetrievalHit]:
        """Retrieve PIT-safe evidence with selectable ablation strategies.

        strategy:
            dense  -> Qdrant dense retrieval only
            bm25   -> local BM25 over the PIT/scope-filtered corpus
            hybrid -> Dense + BM25 fused by reciprocal-rank fusion

        use_reranker=None preserves the production default: use the configured
        reranker when available. Retrieval benchmarks pass it explicitly so
        hybrid and hybrid+reranker can be compared fairly.
        """

        strategy = str(strategy or "hybrid").strip().lower()
        if strategy not in {"dense", "bm25", "hybrid"}:
            raise ValueError(
                "strategy must be one of: dense, bm25, hybrid"
            )

        # Qdrant filter is the first PIT gate; final date checks below are a
        # defense-in-depth gate for every retrieval strategy.
        cutoff = date.fromisoformat(as_of_date[:10])
        industries: list[str] = []
        if industry and str(industry).strip():
            industries = [str(industry).strip()]
        elif hasattr(self.store, "resolve_industries"):
            try:
                industries = list(self.store.resolve_industries(ticker))
            except Exception:
                industries = []

        scope_ids = [company_scope_id(ticker)]
        scope_ids.extend(industry_scope_id(item) for item in industries)
        if include_shared_scopes:
            scope_ids.extend(default_shared_scope_ids())
        scope_ids = list(dict.fromkeys(scope_ids))

        dense: list[tuple[KnowledgeChunk, float]] = []
        if strategy in {"dense", "hybrid"}:
            dense = [
                item
                for item in self.store.query_dense(
                    query,
                    scope_ids=scope_ids,
                    as_of_date=as_of_date,
                    legacy_ticker=ticker,
                    limit=candidate_k,
                    doc_type=doc_type,
                )
                if _publication_date_is_pit_safe(item[0], cutoff)
            ]

        sparse: list[tuple[KnowledgeChunk, float]] = []
        if strategy in {"bm25", "hybrid"}:
            corpus = [
                chunk
                for chunk in self.store.scroll_chunks(
                    scope_ids=scope_ids,
                    as_of_date=as_of_date,
                    legacy_ticker=ticker,
                    limit=corpus_limit,
                    doc_type=doc_type,
                )
                if _publication_date_is_pit_safe(chunk, cutoff)
            ]
            sparse_scores = bm25_scores(query, corpus)
            sparse = sorted(
                zip(corpus, sparse_scores, strict=True),
                key=lambda x: x[1],
                reverse=True,
            )[:candidate_k]

        rrf = defaultdict(float)
        dense_score: dict[str, float] = {}
        sparse_score: dict[str, float] = {}
        chunks: dict[str, KnowledgeChunk] = {}

        for rank, (chunk, score) in enumerate(dense, start=1):
            chunks[chunk.chunk_id] = chunk
            dense_score[chunk.chunk_id] = score
            if strategy == "hybrid":
                rrf[chunk.chunk_id] += 1.0 / (self.rrf_k + rank)

        for rank, (chunk, score) in enumerate(sparse, start=1):
            chunks[chunk.chunk_id] = chunk
            sparse_score[chunk.chunk_id] = score
            if strategy == "hybrid":
                rrf[chunk.chunk_id] += 1.0 / (self.rrf_k + rank)

        if strategy == "dense":
            ordered = [
                chunk.chunk_id
                for chunk, _ in dense
                if date.fromisoformat(chunk.publish_date) <= cutoff
            ]
            base_score = dense_score
        elif strategy == "bm25":
            ordered = [
                chunk.chunk_id
                for chunk, _ in sparse
                if date.fromisoformat(chunk.publish_date) <= cutoff
            ]
            base_score = sparse_score
        else:
            ordered = [
                cid
                for cid, _ in sorted(
                    rrf.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                if date.fromisoformat(chunks[cid].publish_date) <= cutoff
            ]
            base_score = rrf

        rerank_scores: dict[str, float] = {}
        rerank_enabled = (
            self.reranker is not None and strategy == "hybrid"
            if use_reranker is None
            else bool(use_reranker and self.reranker is not None)
        )
        if rerank_enabled and ordered:
            rerank_pool = ordered[: max(top_k * 3, top_k)]
            docs = [
                chunks[cid].title + "\n" + chunks[cid].text
                for cid in rerank_pool
            ]
            for cid, score in zip(
                rerank_pool,
                self.reranker.rerank(query, docs),
                strict=True,
            ):
                rerank_scores[cid] = float(score)
            ordered = sorted(
                ordered,
                key=lambda cid: (
                    rerank_scores.get(cid, float("-inf")),
                    base_score.get(cid, float("-inf")),
                ),
                reverse=True,
            )

        hits = []
        per_document: Counter[str] = Counter()
        cap = max(1, int(max_chunks_per_doc))
        for cid in ordered:
            chunk = chunks[cid]
            document_key = source_document_key(chunk)
            if per_document[document_key] >= cap:
                continue
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=rerank_scores.get(
                        cid,
                        float(base_score.get(cid, 0.0)),
                    ),
                    dense_score=dense_score.get(cid),
                    bm25_score=sparse_score.get(cid),
                    rerank_score=rerank_scores.get(cid),
                )
            )
            per_document[document_key] += 1
            if len(hits) >= top_k:
                break
        return hits
