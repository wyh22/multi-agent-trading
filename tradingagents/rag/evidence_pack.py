from __future__ import annotations

from collections import OrderedDict

# Generic equity-research dimensions. These are intentionally sector-agnostic.
_DIMENSION_QUERIES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        (
            "主营业务",
            "业务模式",
            "经营情况",
            "业务与经营",
            "产品",
            "服务",
            "产能",
            "产量",
            "销量",
            "订单",
            "客户",
            "供应商",
            "business",
            "operations",
        ),
        "主营业务 业务模式 产品服务 产能产量 销量订单 客户供应商 区域布局 经营数据",
    ),
    (
        (
            "竞争",
            "行业地位",
            "市场份额",
            "客户集中度",
            "供应商集中度",
            "护城河",
            "competitive",
            "competition",
            "market share",
        ),
        "行业竞争格局 市场份额 行业地位 核心竞争力 护城河 品牌渠道 技术优势 客户集中度",
    ),
    (
        (
            "财务",
            "现金流",
            "盈利",
            "毛利率",
            "净利率",
            "负债",
            "应收",
            "存货",
            "financial",
            "cash flow",
            "profitability",
        ),
        "营业收入 净利润 毛利率 现金流 资产负债 应收账款 存货 盈利质量 财务风险",
    ),
    (
        (
            "估值",
            "市盈率",
            "市净率",
            "可比公司",
            "valuation",
            "pe",
            "pb",
        ),
        "估值 市盈率 市净率 EV EBITDA DCF 可比公司 估值区间",
    ),
    (
        (
            "增长",
            "扩产",
            "资本开支",
            "在建工程",
            "指引",
            "guidance",
            "capex",
            "growth",
        ),
        "增长驱动 资本开支 在建工程 扩产计划 产能规划 业绩指引 新产品 新市场",
    ),
    (
        (
            "行业",
            "政策",
            "监管",
            "产业链",
            "供需",
            "industry",
            "policy",
            "regulation",
        ),
        "行业周期 供需格局 产业链 政策监管 行业标准 价格机制 行业风险",
    ),
    (
        (
            "并购",
            "重组",
            "回购",
            "增持",
            "减持",
            "股权激励",
            "治理",
            "诉讼",
            "处罚",
            "重大合同",
            "corporate action",
            "governance",
        ),
        "并购重组 回购 增减持 股权激励 公司治理 管理层 诉讼处罚 重大合同",
    ),
    (
        (
            "风险",
            "不确定性",
            "risk",
        ),
        "经营风险 行业风险 政策风险 财务风险 原材料风险 汇率利率风险 信用风险 安全环保风险",
    ),
)


def build_retrieval_queries(text: str, *, max_queries: int = 4) -> list[str]:
    """Expand missing research dimensions into generic equity-research queries.

    This planner is deterministic and sector-agnostic. The original missing-item
    text remains the source of intent; generic query templates only improve
    recall for common equity-research dimensions.
    """

    raw = str(text or "").strip()
    if not raw:
        return []

    normalized = raw.lower()
    queries: "OrderedDict[str, None]" = OrderedDict()
    for keywords, query in _DIMENSION_QUERIES:
        if any(keyword.lower() in normalized for keyword in keywords):
            queries.setdefault(query, None)
        if len(queries) >= max(1, int(max_queries)):
            break

    # Always preserve the user's / Completion Gate's exact wording as one query
    # when capacity remains. This prevents the taxonomy from erasing niche needs.
    if len(queries) < max(1, int(max_queries)):
        queries.setdefault(raw[:500], None)

    return list(queries)[: max(1, int(max_queries))]


# Backward-compatible name used by the repair loop.
build_repair_queries = build_retrieval_queries


def source_document_key(chunk) -> str:
    """Stable parent-document key for retrieval diversity."""

    metadata = getattr(chunk, "metadata", {}) or {}
    return str(metadata.get("file_hash") or getattr(chunk, "doc_id", ""))
