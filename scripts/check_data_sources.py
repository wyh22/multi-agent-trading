"""Free A-share data-source preflight."""

from __future__ import annotations
import sys
from pathlib import Path
import argparse
from datetime import date, timedelta
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)
load_dotenv(PROJECT_ROOT / ".env")


def _status(name: str, fn, *, required: bool = True) -> bool:
    print(f"\n[CHECK] {name}")
    try:
        result = fn()
        print("[OK]", str(result)[:800].replace("\n", " | "))
        return True
    except Exception as exc:
        print(f"[{'FAIL' if required else 'WARN'}] {type(exc).__name__}: {exc}")
        return not required


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="600519.SH")
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()

    from tradingagents.dataflows.akshare_financials import get_free_balance_sheet, get_free_cashflow, get_free_income_statement
    from tradingagents.dataflows.baostock import get_baostock_fundamentals, get_baostock_stock_data
    from tradingagents.dataflows.cninfo import get_cninfo_news
    from tradingagents.dataflows.interface import route_to_vendor

    as_of = date.fromisoformat(args.date)
    market_start = (as_of - timedelta(days=30)).isoformat()
    news_start = (as_of - timedelta(days=14)).isoformat()
    announcement_start = (as_of - timedelta(days=120)).isoformat()

    results = [
        _status("BaoStock OHLCV", lambda: get_baostock_stock_data(args.ticker, market_start, args.date, curr_date=args.date)),
        _status("BaoStock PIT valuation fundamentals", lambda: get_baostock_fundamentals(args.ticker, args.date)),
        _status("AKShare/Sina balance sheet", lambda: get_free_balance_sheet(args.ticker, "quarterly", args.date)),
        _status("AKShare/Sina income statement", lambda: get_free_income_statement(args.ticker, "quarterly", args.date)),
        _status("AKShare/Sina cashflow", lambda: get_free_cashflow(args.ticker, "quarterly", args.date)),
        _status("CNInfo announcements", lambda: get_cninfo_news(args.ticker, announcement_start, args.date, curr_date=args.date), required=False),
        _status("Router: get_stock_data", lambda: route_to_vendor("get_stock_data", args.ticker, market_start, args.date)),
        _status("Router: get_indicators(rsi)", lambda: route_to_vendor("get_indicators", args.ticker, "rsi", args.date, 30)),
        _status("Router: get_fundamentals", lambda: route_to_vendor("get_fundamentals", args.ticker, args.date)),
        _status("Router: get_news", lambda: route_to_vendor("get_news", args.ticker, start_date=news_start, end_date=args.date), required=False),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
