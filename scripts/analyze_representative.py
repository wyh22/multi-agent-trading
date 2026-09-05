"""Run the existing 7-Agent graph from a Representative Research Pool row."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
root_str = str(PROJECT_ROOT)
if root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "从 representative_research_pool.csv 选择一个 ticker，"
            "携带 selection provenance 运行现有 7-Agent 深度研究。"
        )
    )
    parser.add_argument("--pool-csv", required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    pool_path = Path(args.pool_csv).expanduser()
    if not pool_path.exists():
        raise FileNotFoundError(f"Research Pool CSV 不存在: {pool_path}")

    frame = pd.read_csv(pool_path)
    required = {"ticker", "research_context"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Research Pool CSV 缺少列: {sorted(missing)}")

    ticker = str(args.ticker).strip().upper()
    matches = frame[frame["ticker"].astype(str).str.upper() == ticker]
    if matches.empty:
        available = ", ".join(frame["ticker"].astype(str).head(20))
        raise ValueError(
            f"{ticker} 不在该 Research Pool 中。前20个可用 ticker: {available}"
        )

    row = matches.iloc[0]
    candidate_context = str(row.get("research_context", "") or "")
    print(
        f"Research Entry: {ticker} | "
        f"sector={row.get('sector_name', '')} | "
        f"representative_score={row.get('representative_score', '')}"
    )
    print(
        "Selection provenance 会传入 Agent，但被明确标记为 NOT evidence；"
        "市场/新闻/基本面结论仍必须由工具独立验证。"
    )

    graph = TradingAgentsGraph(config=DEFAULT_CONFIG)
    state, signal = graph.propagate(
        ticker,
        args.date,
        candidate_context=candidate_context,
    )
    output_dir = graph.save_reports(
        state,
        ticker,
        save_path=args.output or None,
    )

    print(f"\nSignal: {signal}")
    print(f"Audit Status: {state.get('audit_status', '')}")
    print(f"Reports: {output_dir}")
    print("\nFinal Decision:\n")
    print(state.get("final_trade_decision", ""))
    if state.get("audit_report"):
        print("\n--- Audit ---\n")
        print(state["audit_report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
