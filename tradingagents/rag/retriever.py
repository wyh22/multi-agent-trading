from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import date
from typing import Iterable

from .embeddings import HashEmbedding, build_embedder, build_reranker
from .models import KnowledgeChunk, RetrievalHit
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


class InMemoryKnowledgeStore:
    """Dependency-free store used by tests and small local demos."""

    def __init__(self, chunks: Iterable[KnowledgeChunk], embedder=None):
        self.chunks = list(chunks)
        self.embedder = embedder or HashEmbedding()
        self._vectors = self.embedder.embed([c.text for c in self.chunks]) if self.chunks else []

    def _eligible(self, ticker: str, as_of_date: str, doc_type: str | None = None):
        cutoff = date.fromisoformat(as_of_date[:10])
        out = []
        for i, c in enumerate(self.chunks):
            if c.ticker != ticker:
                continue
            if date.fromisoformat(c.publish_date) > cutoff:
                continue
            if doc_type and c.doc_type != doc_type:
                continue
            out.append((i, c))
        return out

    def query_dense(self, query: str, *, ticker: str, as_of_date: str, limit: int = 20, doc_type=None):
        qv = self.embedder.embed([query])[0]
        scored = [(c, _cosine(qv, self._vectors[i])) for i, c in self._eligible(ticker, as_of_date, doc_type)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def scroll_chunks(self, *, ticker: str, as_of_date: str, limit: int = 1000, doc_type=None):
        return [c for _, c in self._eligible(ticker, as_of_date, doc_type)][:limit]


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
    ) -> list[RetrievalHit]:
        # Qdrant filter is the first PIT gate; final date check below is a defense-in-depth gate.
        dense = self.store.query_dense(
            query, ticker=ticker, as_of_date=as_of_date, limit=candidate_k, doc_type=doc_type
        )
        corpus = self.store.scroll_chunks(
            ticker=ticker, as_of_date=as_of_date, limit=corpus_limit, doc_type=doc_type
        )
        sparse_scores = bm25_scores(query, corpus)
        sparse = sorted(zip(corpus, sparse_scores, strict=True), key=lambda x: x[1], reverse=True)[:candidate_k]

        rrf = defaultdict(float)
        dense_score = {}
        sparse_score = {}
        chunks: dict[str, KnowledgeChunk] = {}
        for rank, (chunk, score) in enumerate(dense, start=1):
            chunks[chunk.chunk_id] = chunk
            dense_score[chunk.chunk_id] = score
            rrf[chunk.chunk_id] += 1.0 / (self.rrf_k + rank)
        for rank, (chunk, score) in enumerate(sparse, start=1):
            chunks[chunk.chunk_id] = chunk
            sparse_score[chunk.chunk_id] = score
            rrf[chunk.chunk_id] += 1.0 / (self.rrf_k + rank)

        cutoff = date.fromisoformat(as_of_date[:10])
        ordered = [
            cid for cid, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)
            if date.fromisoformat(chunks[cid].publish_date) <= cutoff
        ]

        rerank_scores: dict[str, float] = {}
        if self.reranker and ordered:
            rerank_pool = ordered[: max(top_k * 3, top_k)]
            docs = [chunks[cid].title + "\n" + chunks[cid].text for cid in rerank_pool]
            for cid, score in zip(rerank_pool, self.reranker.rerank(query, docs), strict=True):
                rerank_scores[cid] = float(score)
            ordered = sorted(
                ordered,
                key=lambda cid: (rerank_scores.get(cid, float("-inf")), rrf[cid]),
                reverse=True,
            )

        hits = []
        for cid in ordered[:top_k]:
            hits.append(
                RetrievalHit(
                    chunk=chunks[cid],
                    score=rerank_scores.get(cid, rrf[cid]),
                    dense_score=dense_score.get(cid),
                    bm25_score=sparse_score.get(cid),
                    rerank_score=rerank_scores.get(cid),
                )
            )
        return hits
