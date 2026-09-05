from __future__ import annotations

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


@mcp.tool()
def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Get PIT-safe A-share OHLCV data."""
    return _invoke(lc_get_stock_data, symbol=symbol, start_date=start_date, end_date=end_date)


@mcp.tool()
def get_indicators(symbol: str, indicator: str, curr_date: str, look_back_days: int = 30) -> str:
    """Get a technical indicator as of the research cutoff date."""
    return _invoke(
        lc_get_indicators,
        symbol=symbol,
        indicator=indicator,
        curr_date=curr_date,
        look_back_days=look_back_days,
    )


@mcp.tool()
def get_verified_market_snapshot(symbol: str, curr_date: str, look_back_days: int = 30) -> str:
    """Get a deterministic verified market snapshot for numeric claims."""
    return _invoke(
        lc_get_verified_market_snapshot,
        symbol=symbol,
        curr_date=curr_date,
        look_back_days=look_back_days,
    )


@mcp.tool()
def get_fundamentals(ticker: str, curr_date: str) -> str:
    """Get PIT-safe valuation/fundamental summary."""
    return _invoke(lc_get_fundamentals, ticker=ticker, curr_date=curr_date)


@mcp.tool()
def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    """Get a disclosed balance sheet available by the cutoff date."""
    return _invoke(lc_get_balance_sheet, ticker=ticker, freq=freq, curr_date=curr_date)


@mcp.tool()
def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    """Get a disclosed cash-flow statement available by the cutoff date."""
    return _invoke(lc_get_cashflow, ticker=ticker, freq=freq, curr_date=curr_date)


@mcp.tool()
def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    """Get a disclosed income statement available by the cutoff date."""
    return _invoke(lc_get_income_statement, ticker=ticker, freq=freq, curr_date=curr_date)


@mcp.tool()
def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Get company announcements/news available no later than end_date."""
    return _invoke(lc_get_news, ticker=ticker, start_date=start_date, end_date=end_date)


@mcp.tool()
def get_global_news(curr_date: str, look_back_days: int | None = None, limit: int | None = None) -> str:
    """Get macro/global news available by curr_date."""
    return _invoke(
        lc_get_global_news,
        curr_date=curr_date,
        look_back_days=look_back_days,
        limit=limit,
    )


@mcp.tool()
def get_insider_transactions(ticker: str, curr_date: str | None = None) -> str:
    """Get management/related-person holding changes available by curr_date."""
    return _invoke(lc_get_insider_transactions, ticker=ticker, curr_date=curr_date)


@mcp.tool()
def get_macro_indicators(indicator: str, curr_date: str, look_back_days: int | None = None) -> str:
    """Get a Chinese macro indicator with explicit research cutoff."""
    return _invoke(
        lc_get_macro_indicators,
        indicator=indicator,
        curr_date=curr_date,
        look_back_days=look_back_days,
    )


@mcp.tool()
def search_company_knowledge(\n    query: str,\n    ticker: str,\n    as_of_date: str,\n    top_k: int = 6,\n    doc_type: str | None = None,\n) -> str:\n    """Hybrid RAG over company documents with publish-date PIT filtering."""
    return _invoke(
        lc_search_company_knowledge,
        query=query,
        ticker=ticker,
        as_of_date=as_of_date,
        top_k=top_k,\n        doc_type=doc_type,\n    )\n\n\n@mcp.custom_route("/health", methods=["GET"])
async def health(_request):
    return JSONResponse({"status": "ok", "service": "tradingagents-finance-mcp"})


def main():
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
