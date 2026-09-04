from __future__ import annotations

import functools
import os

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("MCP SDK 未安装，请执行 `pip install -e '.[agent]'`") from exc

from starlette.responses import JSONResponse

from tradingagents.agents.utils.agent_utils import (
    get_balance_sheet as lc_get_balance_sheet,
    get_cashflow as lc_get_cashflow,
    get_fundamentals as lc_get_fundamentals,
    get_global_news as lc_get_global_news,
    get_income_statement as lc_get_income_statement,
    get_indicators as lc_get_indicators,
    get_insider_transactions as lc_get_insider_transactions,
    get_macro_indicators as lc_get_macro_indicators,
    get_news as lc_get_news,
    get_stock_data as lc_get_stock_data,
    get_verified_market_snapshot as lc_get_verified_market_snapshot,
)
from tradingagents.agents.utils.rag_tools import search_company_knowledge as lc_search_company_knowledge
from tradingagents.dataflows.config import get_config
from tradingagents.mcp.ifind import IFinDHTTPClient, compact_ifind_json

mcp = FastMCP(
    "TradingAgents A-share Finance Tools",
    instructions=(
        "A-share deterministic finance tools and PIT-aware knowledge retrieval. "
        "All historical queries must pass an explicit research cutoff date."
    ),
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "8001")),
    stateless_http=True,
    json_response=True,
)


def _invoke(tool, **kwargs):
    return tool.invoke(kwargs)


@functools.lru_cache(maxsize=1)
def _ifind_client() -> IFinDHTTPClient:
    return IFinDHTTPClient.from_config(get_config())


def _ifind_disabled() -> str | None:
    config = get_config()
    if not config.get("ifind_enabled", False):
        return "IFIND_DISABLED: 设置TRADINGAGENTS_IFIND_ENABLED=true并配置TRADINGAGENTS_IFIND_REFRESH_TOKEN后启用。"
    if not config.get("ifind_refresh_token"):
        return "IFIND_UNCONFIGURED: 缺少TRADINGAGENTS_IFIND_REFRESH_TOKEN。"
    return None


@mcp.tool()
def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Get PIT-safe A-share OHLCV data."""
    return _invoke(lc_get_stock_data, symbol=symbol, start_date=start_date, end_date=end_date)


@mcp.tool()
def get_indicators(symbol: str, indicator: str, curr_date: str, look_back_days: int = 30) -> str:
    """Get a technical indicator as of the research cutoff date."""
    return _invoke(
        lc_get_indicators, symbol=symbol, indicator=indicator, curr_date=curr_date, look_back_days=look_back_days
    )


@mcp.tool()
def get_verified_market_snapshot(symbol: str, curr_date: str, look_back_days: int = 30) -> str:
    """Get a final verified market snapshot for numeric claims."""
    return _invoke(lc_get_verified_market_snapshot, symbol=symbol, curr_date=curr_date, look_back_days=look_back_days)


@mcp.tool()
def get_fundamentals(ticker: str, curr_date: str) -> str:
    """Get PIT-safe valuation/fundamental summary."""
    return _invoke(lc_get_fundamentals, ticker=ticker, curr_date=curr_date)


@mcp.tool()
def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    """Get disclosed balance sheet available by the cutoff date."""
    return _invoke(lc_get_balance_sheet, ticker=ticker, freq=freq, curr_date=curr_date)


@mcp.tool()
def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    """Get disclosed cash-flow statement available by the cutoff date."""
    return _invoke(lc_get_cashflow, ticker=ticker, freq=freq, curr_date=curr_date)


@mcp.tool()
def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    """Get disclosed income statement available by the cutoff date."""
    return _invoke(lc_get_income_statement, ticker=ticker, freq=freq, curr_date=curr_date)


@mcp.tool()
def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Get company announcements/news available no later than end_date."""
    return _invoke(lc_get_news, ticker=ticker, start_date=start_date, end_date=end_date)


@mcp.tool()
def get_global_news(curr_date: str, look_back_days: int | None = None, limit: int | None = None) -> str:
    """Get macro/global news available by curr_date."""
    return _invoke(lc_get_global_news, curr_date=curr_date, look_back_days=look_back_days, limit=limit)


@mcp.tool()
def get_insider_transactions(ticker: str, curr_date: str) -> str:
    """Get disclosed important transactions available by curr_date."""
    return _invoke(lc_get_insider_transactions, ticker=ticker, curr_date=curr_date)


@mcp.tool()
def get_macro_indicators(indicator: str, curr_date: str, look_back_days: int | None = None) -> str:
    """Get a Chinese macro indicator with explicit research cutoff."""
    return _invoke(lc_get_macro_indicators, indicator=indicator, curr_date=curr_date, look_back_days=look_back_days)


@mcp.tool()
def search_company_knowledge(query: str, ticker: str, as_of_date: str, top_k: int = 6) -> str:
    """Hybrid RAG over company documents with publish-date PIT filtering."""
    return _invoke(lc_search_company_knowledge, query=query, ticker=ticker, as_of_date=as_of_date, top_k=top_k)


@mcp.tool()
def ifind_status() -> str:
    """Check whether the optional Tonghuashun iFinD HTTP adapter is configured."""
    config = get_config()
    return compact_ifind_json({
        "enabled": bool(config.get("ifind_enabled", False)),
        "configured": bool(config.get("ifind_refresh_token")),
        "base_url": config.get("ifind_base_url"),
    })


@mcp.tool()
def ifind_snapshot(codes: str, indicators: str = "latest", start_time: str = "", end_time: str = "") -> str:
    """Query iFinD snapshot/current market data. This is not a historical PIT tool."""
    disabled = _ifind_disabled()
    if disabled:
        return disabled
    try:
        return compact_ifind_json(_ifind_client().snapshot(
            codes=codes, indicators=indicators, start_time=start_time, end_time=end_time
        ))
    except Exception as exc:  # noqa: BLE001
        return f"IFIND_ERROR: {type(exc).__name__}: {exc}"


@mcp.tool()
def ifind_basic_data(codes: str, indicators: str, params: str = "") -> str:
    """Query iFinD basic/fundamental indicators; separate multiple indicators/param groups with semicolons."""
    disabled = _ifind_disabled()
    if disabled:
        return disabled
    try:
        return compact_ifind_json(_ifind_client().basic_data(codes=codes, indicators=indicators, params=params))
    except Exception as exc:  # noqa: BLE001
        return f"IFIND_ERROR: {type(exc).__name__}: {exc}"


@mcp.tool()
def ifind_date_sequence(
    codes: str,
    indicators: str,
    params: str,
    start_date: str,
    end_date: str,
    as_of_date: str,
    fill: str = "Blank",
    days: str = "Tradedays",
    interval: str = "D",
) -> str:
    """Query iFinD historical series with an explicit research cutoff; future end_date is rejected."""
    disabled = _ifind_disabled()
    if disabled:
        return disabled
    if end_date.replace("-", "") > as_of_date.replace("-", ""):
        return f"IFIND_PIT_REJECTED: end_date={end_date} 晚于研究截止日 as_of_date={as_of_date}"
    try:
        return compact_ifind_json(_ifind_client().date_sequence(
            codes=codes, indicators=indicators, params=params, start_date=start_date, end_date=end_date,
            fill=fill, days=days, interval=interval,
        ))
    except Exception as exc:  # noqa: BLE001
        return f"IFIND_ERROR: {type(exc).__name__}: {exc}"


@mcp.custom_route("/health", methods=["GET"])
async def health(_request):
    return JSONResponse({
        "status": "ok",
        "service": "tradingagents-finance-mcp",
        "ifind_enabled": bool(get_config().get("ifind_enabled", False)),
        "ifind_configured": bool(get_config().get("ifind_refresh_token")),
    })


def main():
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
