from __future__ import annotations

import functools
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.symbol_utils import normalize_a_share_symbol


@functools.lru_cache(maxsize=1)
def _retriever():
    from tradingagents.rag.retriever import HybridKnowledgeRetriever

    return HybridKnowledgeRetriever.from_config(get_config())


@tool
def search_company_knowledge(
    query: Annotated[str, "要检索的公司公告/财报问题或关键词"],
    ticker: Annotated[str, "股票代码，例如 600519.SH"],
    as_of_date: Annotated[str, "研究截止日期，格式 YYYY-MM-DD；检索结果发布日期不得晚于此日"],
    top_k: Annotated[int, "返回证据片段数量，建议 3~8"] = 6,
    doc_type: Annotated[str | None, "可选文档类型过滤，如 annual_report / announcement"] = None,
) -> str:
    """PIT-aware hybrid RAG：Dense+BM25+RRF，并可选 Cross-Encoder Rerank。"""

    config = get_config()
    if not config.get("rag_enabled", False):
        return "RAG_DISABLED: 未启用公司知识库检索。"
    try:
        canonical = normalize_a_share_symbol(ticker)
        hits = _retriever().search(
            query,
            ticker=canonical,
            as_of_date=as_of_date,
            top_k=max(1, min(int(top_k), 10)),
            candidate_k=int(config.get("rag_candidate_k", 30)),
            corpus_limit=int(config.get("rag_bm25_corpus_limit", 1000)),
            doc_type=doc_type,
            max_chunks_per_doc=int(config.get("rag_max_chunks_per_doc", 2)),
        )
    except Exception as exc:  # noqa: BLE001
        return f"RAG_UNAVAILABLE: {type(exc).__name__}: {exc}"
    if not hits:
        return f"NO_RAG_EVIDENCE: {canonical} 在 {as_of_date} 之前没有匹配知识片段。"

    lines = [f"## {canonical} PIT-safe 知识库证据（截止 {as_of_date}）"]
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        excerpt = c.text.replace("\n", " ").strip()
        max_chars = int(config.get("rag_excerpt_chars", 650))
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars] + "…"
        provenance = c.metadata or {}
        authority = str(provenance.get("source_authority") or c.source)
        verified = provenance.get("publish_date_verified")
        verified_label = (
            "verified" if verified is True else "unverified" if verified is False else "legacy"
        )
        evidence_id = f"RAG:{c.doc_id}#chunk-{c.chunk_index}"
        meta = f"[{c.publish_date}] [{authority}] [{verified_label}] {c.title}"
        lines.append(
            f"{i}. {meta}\n"
            f"   evidence_id={evidence_id}\n"
            f"   {excerpt}\n"
            f"   source={provenance.get('source_url') or c.url or c.doc_id}"
        )
    lines.append("\n注意：以上片段均经过 publish_date<=as_of_date 的 PIT 过滤；历史研究还会排除显式未核验披露日的上传文档。引用结论时请保留 evidence_id/原始来源。")
    return "\n".join(lines)
