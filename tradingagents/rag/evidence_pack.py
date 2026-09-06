from __future__ import annotations

from collections import OrderedDict


_REPAIR_QUERY_GROUPS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        (
            "主营业务",
            "业务与经营",
            "装机容量",
            "发电量",
            "利用小时",
            "运营规模",
            "business operations",
            "competitive position",
        ),
        "主营业务构成 装机容量 发电量 利用小时 运营规模 项目布局 竞争优势 行业地位",
    ),
    (
        (
            "政策",
            "规划",
            "policy",
        ),
        "风电 新能源 行业政策 产业规划 消纳政策 电价机制",
    ),
    (
        (
            "弃风",
            "限电",
            "消纳",
            "curtailment",
        ),
        "弃风率 限电 新能源消纳 利用小时 并网约束",
    ),
    (
        (
            "补贴",
            "国补",
            "可再生能源补贴",
            "subsidy",
        ),
        "可再生能源补贴 国补 补贴应收 回款 补贴退坡",
    ),
    (
        (
            "估值",
            "市盈率",
            "市净率",
            "valuation",
        ),
        "估值 市盈率 市净率 估值方法 同行业可比公司",
    ),
    (
        (
            "风险",
            "risk",
        ),
        "主要风险 经营风险 政策风险 项目建设风险 电价风险 消纳风险",
    ),
)


def build_repair_queries(text: str, *, max_queries: int = 4) -> list[str]:
    """Build a small deterministic query set for document-evidence repair.

    The function intentionally does not use an LLM: missing-item text is already
    available from the Task Contract / Completion Gate, so deterministic query
    expansion is cheaper, reproducible and easier to benchmark.
    """

    raw = str(text or "").strip()
    if not raw:
        return []

    normalized = raw.lower()
    queries: "OrderedDict[str, None]" = OrderedDict()
    for keywords, query in _REPAIR_QUERY_GROUPS:
        if any(keyword.lower() in normalized for keyword in keywords):
            queries.setdefault(query, None)
        if len(queries) >= max(1, int(max_queries)):
            break

    if not queries:
        queries[raw[:500]] = None

    return list(queries)[: max(1, int(max_queries))]


def source_document_key(chunk) -> str:
    """Return a stable parent-document key for retrieval diversity.

    PDF pages have page-specific doc_ids, while all pages share file_hash.
    Grouping by file_hash prevents one long PDF from occupying every top-k slot.
    """

    metadata = getattr(chunk, "metadata", {}) or {}
    return str(metadata.get("file_hash") or getattr(chunk, "doc_id", ""))
