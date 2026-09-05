from __future__ import annotations

import logging

from tradingagents.agents.utils.core_stock_tools import get_stock_data
from tradingagents.agents.utils.fundamental_data_tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
)
from tradingagents.agents.utils.macro_data_tools import get_macro_indicators
from tradingagents.agents.utils.market_data_validation_tools import get_verified_market_snapshot
from tradingagents.agents.utils.news_data_tools import (
    get_global_news,
    get_insider_transactions,
    get_news,
)
from tradingagents.agents.utils.rag_tools import search_company_knowledge
from tradingagents.agents.utils.technical_indicators_tools import get_indicators

logger = logging.getLogger(__name__)

GROUP_NAMES = {
    "market": ["get_stock_data", "get_indicators", "get_verified_market_snapshot"],
    "news": ["get_news", "get_global_news", "get_insider_transactions", "get_macro_indicators"],
    "fundamentals": [
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
    ],
}


def _append_unique(group: list, tool) -> None:
    if all(getattr(item, "name", None) != getattr(tool, "name", None) for item in group):
        group.append(tool)


def build_local_tool_groups(config: dict):
    groups = {
        "market": [get_stock_data, get_indicators, get_verified_market_snapshot],
        "news": [get_news, get_global_news, get_insider_transactions, get_macro_indicators],
        "fundamentals": [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ],
        "knowledge": [],
    }
    if config.get("rag_enabled", False):
        # RAG is a shared read-only evidence capability. News and Fundamentals
        # may both use it, while the explicit knowledge group lets the
        # Conversation Supervisor call it directly.
        groups["knowledge"].append(search_company_knowledge)
        _append_unique(groups["news"], search_company_knowledge)
        _append_unique(groups["fundamentals"], search_company_knowledge)
    return groups


def build_tool_groups(config: dict):
    local = build_local_tool_groups(config)
    if not config.get("mcp_enabled", False):
        return local

    from tradingagents.mcp.client import load_mcp_tools_sync

    try:
        remote = load_mcp_tools_sync(
            str(config.get("mcp_url", "http://localhost:8001/mcp"))
        )
        by_name = {tool.name: tool for tool in remote}
        groups = {}
        for group, names in GROUP_NAMES.items():
            missing = [name for name in names if name not in by_name]
            if missing:
                raise RuntimeError(f"MCP Server 缺少工具: {missing}")
            groups[group] = [by_name[name] for name in names]

        groups["knowledge"] = []
        if config.get("rag_enabled", False):
            rag_tool = by_name.get("search_company_knowledge")
            if rag_tool is None:
                # Keep RAG local if the finance MCP server intentionally does
                # not expose the knowledge store. This preserves the deployment
                # boundary without making RAG availability depend on MCP.
                rag_tool = search_company_knowledge
            groups["knowledge"].append(rag_tool)
            _append_unique(groups["news"], rag_tool)
            _append_unique(groups["fundamentals"], rag_tool)

        logger.info(
            "已从 MCP Server 加载 %d 个远程工具",
            sum(len(value) for key, value in groups.items() if key != "knowledge"),
        )
        return groups
    except Exception as exc:  # noqa: BLE001
        if not config.get("mcp_fallback_to_local", True):
            raise
        logger.warning("MCP 工具加载失败，回退本地工具: %s", exc)
        return local
