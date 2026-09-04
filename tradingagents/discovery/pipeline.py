from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

from .market import analyze_market_regime
from .models import DiscoveryResult
from .screener import load_sector_components, screen_stocks
from .sectors import analyze_sectors


def _df_to_markdown(df) -> str:
    if df is None or df.empty:
        return ""
    shown = df.copy()
    for col in shown.columns:
        shown[col] = shown[col].map(lambda v: "" if v is None or str(v) == "nan" else v)
    headers = [str(c) for c in shown.columns]
    def esc(v): return str(v).replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(map(esc, headers)) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(esc(v) for v in row) + " |")
    return "\n".join(lines)


def _validate_stock_discovery_date(as_of_date: str, strict_pit: bool) -> None:
    if not strict_pit:
        return
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    if as_of < date.today() - timedelta(days=7):
        raise ValueError(
            "严格 PIT 模式下，自动选股仅支持当前/最近交易日（7天内）。"
            "申万 index_component_sw 只提供当前成分股及计入日期，无法完整恢复历史退出成分，"
            "直接用于历史日期会产生幸存者偏差。市场和板块分析仍可对历史日期运行。"
        )


def run_discovery(
    as_of_date: str,
    *,
    sector_count: int = 4,
    per_sector: int = 35,
    top_n: int = 10,
    max_shortlist_per_sector: int | None = None,
    quality_pool_size: int | None = None,
    quality_weight: float = 0.25,
    quality_enabled: bool = True,
    strict_pit: bool = True,
) -> DiscoveryResult:
    market = analyze_market_regime(as_of_date)
    sector_result = analyze_sectors(
        as_of_date, market_regime=market.regime, top_n=max(sector_count, 10)
    )
    _validate_stock_discovery_date(as_of_date, strict_pit)
    top_sectors = sector_result.sectors.head(sector_count)
    components = load_sector_components(top_sectors, as_of_date, max_per_sector=per_sector)
    stock_kwargs = {
        "market_regime": market.regime,
        "top_n": top_n,
        "max_shortlist_per_sector": max_shortlist_per_sector,
        "quality_pool_size": quality_pool_size,
        "quality_weight": quality_weight,
    }
    if not quality_enabled:
        stock_kwargs["quality_loader"] = None
    stocks = screen_stocks(components, as_of_date, **stock_kwargs)
    return DiscoveryResult(
        as_of_date=as_of_date, market=market, sectors=sector_result, stocks=stocks,
        metadata={
            "sector_count": sector_count, "per_sector": per_sector, "top_n": top_n,
            "max_shortlist_per_sector": max_shortlist_per_sector,
            "quality_pool_size": stocks.quality_pool_size,
            "quality_scored_size": stocks.quality_scored_size,
            "quality_weight": quality_weight, "quality_enabled": quality_enabled,
            "strict_pit": strict_pit,
            "method": "市场环境 -> 申万行业排名与趋势惩罚 -> BaoStock 量化筛选 -> 行业中性估值 -> PIT 轻量基本面二筛 -> 行业软上限",
        },
    )


def discovery_markdown(result: DiscoveryResult) -> str:
    market = result.market
    sectors = result.sectors.sectors
    stocks = result.stocks.candidates
    lines = [
        f"# A股自动投研候选发现报告 — {result.as_of_date}", "",
        "> 本报告为研究候选筛选，不构成投资建议或收益承诺。自动选股阶段不调用 LLM；",
        "> 先用大盘/行业/量价估值因子缩小范围，再建议把 Top 候选交给现有 Multi-Agent 做深度研究。", "",
        "## 1. 大盘环境", "",
        f"- Market Regime: **{market.regime}**", f"- 综合分: **{market.score:.1f}/100**",
        f"- 摘要: {market.summary}", "",
        _df_to_markdown(market.indices) if not market.indices.empty else "无可用指数数据。", "",
        "## 2. 申万一级行业排名", "",
        _df_to_markdown(sectors) if not sectors.empty else "无可用行业数据。", "",
        "## 3. Research Shortlist", "",
        f"- 行业成分初始股票池: {result.stocks.universe_size}",
        f"- 完成因子评分: {result.stocks.scored_size}",
        ("- Shortlist 行业分布（Soft Cap）: " + ", ".join(f"{k} {v}只" for k,v in result.stocks.sector_quotas.items())
         if result.stocks.sector_quotas else "- Shortlist 行业分布: 无可用数据"),
        f"- 轻量基本面二筛: Pool {result.stocks.quality_pool_size} 只 / 有效财务 {result.stocks.quality_scored_size} 只", "",
        _df_to_markdown(stocks) if not stocks.empty else "没有通过过滤的候选股票。", "",
        "## 4. 评分解释", "",
        "- Momentum: 20/60 日收益 + MA20/MA60 趋势。",
        "- Valuation: PE(TTM)/PB 按申万一级行业内横截面排名；PE<=0 不参与便宜度评分。",
        "- Liquidity: 20 日平均成交额与换手率。",
        "- Risk: 20 日年化波动率与 60 日最大回撤。",
        "- Quality: Quant Top Pool 使用 BaoStock PIT 季度财务做轻量二筛。",
        "- Diversification: Soft Sector Cap 限制单行业最大集中度，不为低分行业强制预留名额。", "",
        "## 5. 下一步", "",
        "对 Top 3~5 候选逐只运行 `python -m cli.main analyze`，用 7-Agent 证据约束流程完成深度报告。",
    ]
    warnings = market.warnings + result.sectors.warnings + result.stocks.warnings
    if warnings:
        lines += ["", "## 数据/质量提示", ""] + [f"- {w}" for w in warnings]
    return "\n".join(lines).strip() + "\n"


def write_discovery_report(result: DiscoveryResult, output_dir: str | Path) -> Path:
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    result.market.indices.to_csv(out/"market_indices.csv",index=False)
    result.sectors.sectors.to_csv(out/"sector_ranking.csv",index=False)
    result.stocks.candidates.to_csv(out/"stock_shortlist.csv",index=False)
    report=out/"discovery_report.md"; report.write_text(discovery_markdown(result),encoding="utf-8")
    (out/"metadata.json").write_text(json.dumps({
        "as_of_date":result.as_of_date,"market_regime":result.market.regime,"market_score":result.market.score,**result.metadata
    },ensure_ascii=False,indent=2),encoding="utf-8")
    return report
