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
        meta = f"[{c.publish_date}] [{c.source}] {c.title}"
        lines.append(f"{i}. {meta}\n   {excerpt}\n   source={c.url or c.doc_id}")
    lines.append("\n注意：以上片段均经过 publish_date<=as_of_date 的 PIT 过滤；结论仍需结合原始来源核验。")
    return "\n".join(lines)
