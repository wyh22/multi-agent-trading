"""A 股大盘、行业与自动研究候选发现入口。"""

from __future__ import annotations
import argparse, os, sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
root_str = str(PROJECT_ROOT)
if root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)
os.environ.setdefault("TQDM_DISABLE", "1")

from tradingagents.discovery.market import analyze_market_regime
from tradingagents.discovery.pipeline import run_discovery, write_discovery_report
from tradingagents.discovery.sectors import analyze_sectors


def _print_frame(df, n=20):
    if df is None or df.empty:
        print("（无数据）")
        return
    with __import__("pandas").option_context("display.max_columns", None, "display.width", 180):
        print(df.head(n).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="A股大盘/行业/自动研究候选发现（不调用 LLM）")
    parser.add_argument("--mode", choices=["market", "sectors", "all"], default="all")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--sectors", type=int, default=4)
    parser.add_argument("--per-sector", type=int, default=35)
    parser.add_argument("--max-sector-picks", type=int, default=0)
    parser.add_argument("--quality-pool", type=int, default=0)
    parser.add_argument("--quality-weight", type=float, default=0.25)
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--allow-historical-membership", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.mode == "market":
        result = analyze_market_regime(args.date)
        print(f"市场环境: {result.regime} | 综合分={result.score:.1f}/100")
        print(result.summary)
        _print_frame(result.indices, args.top)
        return 0

    market = analyze_market_regime(args.date)
    if args.mode == "sectors":
        sectors = analyze_sectors(args.date, market_regime=market.regime, top_n=args.top)
        print(f"市场环境: {market.regime} | 综合分={market.score:.1f}/100")
        _print_frame(sectors.sectors, args.top)
        return 0

    result = run_discovery(
        args.date,
        sector_count=args.sectors,
        per_sector=args.per_sector,
        top_n=args.top,
        max_shortlist_per_sector=(args.max_sector_picks or None),
        quality_pool_size=(args.quality_pool or None),
        quality_weight=args.quality_weight,
        quality_enabled=not args.skip_quality,
        strict_pit=not args.allow_historical_membership,
    )
    print(f"Market Regime: {result.market.regime} | score={result.market.score:.1f}/100")
    print("排名靠前的行业：")
    _print_frame(result.sectors.sectors, args.sectors)
    print("\n研究候选清单：")
    if result.stocks.sector_quotas:
        print("行业分布（软上限）：", ", ".join(f"{k}:{v}" for k, v in result.stocks.sector_quotas.items()))
    print(f"基本面二筛：候选池={result.stocks.quality_pool_size}，有效财务={result.stocks.quality_scored_size}")
    _print_frame(result.stocks.candidates, args.top)

    if args.output:
        out = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = PROJECT_ROOT / "reports" / f"discovery_{args.date.replace('-', '')}_{stamp}"
    report = write_discovery_report(result, out)
    print(f"\n已保存：{report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
