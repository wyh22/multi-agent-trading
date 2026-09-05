"""A 股市场环境与行业研究优先级发现入口。"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
root_str = str(PROJECT_ROOT)
if root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)
os.environ.setdefault("TQDM_DISABLE", "1")

from tradingagents.discovery.market import analyze_market_regime
from tradingagents.discovery.pipeline import (
    run_discovery,
    run_research_pool,
    run_stock_discovery_legacy,
    write_discovery_report,
    write_research_pool_report,
)
from tradingagents.discovery.sectors import analyze_sectors


def _print_frame(df, n=20):
    if df is None or df.empty:
        print("（无数据）")
        return
    with __import__("pandas").option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
    ):
        print(df.head(n).to_string(index=False))


def _legacy_stock_mode(args) -> int:
    result = run_stock_discovery_legacy(
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
    print(
        f"[LEGACY] Market Regime: {result.market.regime} | "
        f"score={result.market.score:.1f}/100"
    )
    print("旧版股票 Research Shortlist：")
    if result.stocks.sector_quotas:
        print(
            "行业分布（软上限）：",
            ", ".join(
                f"{key}:{value}"
                for key, value in result.stocks.sector_quotas.items()
            ),
        )
    print(
        "基本面二筛："
        f"候选池={result.stocks.quality_pool_size}，"
        f"有效财务={result.stocks.quality_scored_size}"
    )
    _print_frame(result.stocks.candidates, args.top)
    print(
        "\n提示：legacy-stock 仅用于与旧版 Sector-first hard gate "
        "股票筛选进行对比；默认主链已经切换为行业发现。"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A股大盘/行业研究优先级发现。默认不调用 LLM；"
            "可选 LightGBM Sector Ranker。"
        )
    )
    parser.add_argument(
        "--mode",
        choices=["market", "sectors", "all", "pool", "legacy-stock"],
        default="all",
    )
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument(
        "--top",
        type=int,
        default=6,
        help="行业发现模式下输出 Top-K 行业；legacy-stock 下表示股票 Top-N。",
    )
    parser.add_argument(
        "--ml-model",
        default=None,
        help="可选 LightGBM Booster 模型路径；未提供时使用确定性 Rule Rank。",
    )
    parser.add_argument(
        "--ml-weight",
        type=float,
        default=0.5,
        help="ML percentile score 与 Rule Score 的融合权重，范围 0~1。",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--representatives-per-sector",
        type=int,
        default=2,
        help="pool 模式下每个 Top 行业选择多少只代表性研究入口。",
    )
    parser.add_argument(
        "--component-limit",
        type=int,
        default=20,
        help="pool 模式下每个行业最多读取多少只当前申万成分股。",
    )

    # Legacy stock-screen arguments are intentionally kept so previous commands
    # still work when the caller explicitly chooses --mode legacy-stock.
    parser.add_argument("--sectors", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--per-sector", type=int, default=35, help=argparse.SUPPRESS)
    parser.add_argument("--max-sector-picks", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--quality-pool", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--quality-weight", type=float, default=0.25, help=argparse.SUPPRESS)
    parser.add_argument("--skip-quality", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--allow-historical-membership",
        action="store_true",
        help=(
            "允许 pool/legacy-stock 在历史日期使用当前申万成分信息；"
            "仅调试用，会引入幸存者偏差。"
        ),
    )

    args = parser.parse_args()

    if args.mode == "market":
        result = analyze_market_regime(args.date)
        print(f"市场环境: {result.regime} | 综合分={result.score:.1f}/100")
        print(result.summary)
        _print_frame(result.indices, args.top)
        return 0

    if args.mode == "sectors":
        market = analyze_market_regime(args.date)
        sectors = analyze_sectors(
            args.date,
            market_regime=market.regime,
            top_n=args.top,
        )
        print(f"市场环境: {market.regime} | 综合分={market.score:.1f}/100")
        _print_frame(sectors.sectors, args.top)
        return 0

    if args.mode == "legacy-stock":
        return _legacy_stock_mode(args)

    if args.mode == "pool":
        result = run_research_pool(
            args.date,
            sector_top_n=args.top,
            representatives_per_sector=args.representatives_per_sector,
            component_limit=args.component_limit,
            strict_pit=not args.allow_historical_membership,
            ml_model_path=args.ml_model,
            ml_weight=args.ml_weight,
        )
        print(
            f"Market Regime: {result.discovery.market.regime} | "
            f"score={result.discovery.market.score:.1f}/100"
        )
        print("\nSector Research Shortlist：")
        _print_frame(result.discovery.sectors.sectors, args.top)
        print("\nRepresentative Research Entries：")
        shown = result.representatives.representatives.drop(
            columns=["research_context"],
            errors="ignore",
        )
        _print_frame(
            shown,
            args.top * args.representatives_per_sector,
        )
        if args.output:
            out = Path(args.output)
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = (
                PROJECT_ROOT
                / "reports"
                / f"research_pool_{args.date.replace('-', '')}_{stamp}"
            )
        report = write_research_pool_report(result, out)
        print(f"\n已保存：{report}")
        print(
            "代表股只是研究入口，不是买入推荐。CSV 中的 research_context "
            "可传给 /analyze 的 candidate_context，让 7-Agent 知道研究来源。"
        )
        return 0

    result = run_discovery(
        args.date,
        top_n=args.top,
        ml_model_path=args.ml_model,
        ml_weight=args.ml_weight,
    )
    print(
        f"Market Regime: {result.market.regime} | "
        f"score={result.market.score:.1f}/100"
    )
    print(
        "Style 权重：",
        ", ".join(
            f"{key}={float(value):.0%}"
            for key, value in result.metadata.get("style_weights", {}).items()
        ),
    )
    print(
        f"排名来源: {result.metadata.get('rank_source', 'rule')} | "
        f"行业横截面={result.metadata.get('sector_universe_size', len(result.sector_universe))}"
    )
    print("\nSector Research Shortlist：")
    shown_cols = [
        "sector_code",
        "sector_name",
        "ret_20d",
        "ret_60d",
        "dividend_yield",
        "momentum_score",
        "valuation_score",
        "dividend_score",
        "liquidity_score",
        "primary_style",
        "style_profile",
        "rule_score",
        "ml_score",
        "sector_score",
    ]
    shown_cols = [
        column
        for column in shown_cols
        if column in result.sectors.sectors.columns
    ]
    _print_frame(result.sectors.sectors[shown_cols], args.top)

    if args.output:
        out = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = (
            PROJECT_ROOT
            / "reports"
            / f"sector_discovery_{args.date.replace('-', '')}_{stamp}"
        )
    report = write_discovery_report(result, out)
    print(f"\n已保存：{report}")
    print(
        "下一步：可运行 --mode pool 自动生成代表性 Research Entries，"
        "再交给 7-Agent 单股研究。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
