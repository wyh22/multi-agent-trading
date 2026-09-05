from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from .market import analyze_market_regime
from .models import (
    DiscoveryResult,
    MarketRegimeResult,
    ResearchPoolResult,
    SectorDiscoveryResult,
    SectorRankingResult,
)
from .representatives import select_representative_stocks
from .screener import load_sector_components, screen_stocks
from .sector_ranker import LightGBMSectorRanker, SectorRanker, blend_sector_scores
from .sectors import analyze_sectors, sector_style_weights


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    shown = df.copy()
    for col in shown.columns:
        shown[col] = shown[col].map(
            lambda v: "" if v is None or str(v) == "nan" else v
        )
    headers = [str(c) for c in shown.columns]

    def esc(v):
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(map(esc, headers)) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(esc(v) for v in row) + " |")
    return "\n".join(lines)


def run_discovery(
    as_of_date: str,
    *,
    top_n: int = 6,
    ml_model_path: str | None = None,
    ml_weight: float = 0.5,
    ranker: SectorRanker | None = None,
    market_analyzer: Callable[..., MarketRegimeResult] = analyze_market_regime,
    sector_analyzer: Callable[..., SectorRankingResult] = analyze_sectors,
) -> SectorDiscoveryResult:
    """Run the primary sector-first discovery pipeline.

    The main discovery output is now a Top-K sector research shortlist. It no
    longer hard-gates the stock universe or claims that one cross-industry factor
    model can identify the globally best stocks.

    A LightGBM ranker is optional. When enabled, its same-day cross-sectional
    percentile score is blended with the deterministic rule score; the rule score
    is always retained for auditability.
    """

    if top_n <= 0:
        raise ValueError("top_n 必须大于 0")
    if ranker is not None and ml_model_path:
        raise ValueError("ranker 与 ml_model_path 只能提供一个")

    market = market_analyzer(as_of_date)
    raw_sector_result = sector_analyzer(
        as_of_date,
        market_regime=market.regime,
        top_n=0,
    )
    ranked = raw_sector_result.sectors.copy()
    if ranked.empty:
        raise RuntimeError("没有可用的申万一级行业排名")

    active_ranker = ranker
    if active_ranker is None and ml_model_path:
        active_ranker = LightGBMSectorRanker(ml_model_path)

    ranked = blend_sector_scores(
        ranked,
        ranker=active_ranker,
        ml_weight=ml_weight if active_ranker is not None else 0.0,
    )
    shortlist = ranked.head(min(int(top_n), len(ranked))).reset_index(drop=True)

    sector_result = SectorRankingResult(
        as_of_date=raw_sector_result.as_of_date,
        current_data_date=raw_sector_result.current_data_date,
        anchor_20d_date=raw_sector_result.anchor_20d_date,
        anchor_60d_date=raw_sector_result.anchor_60d_date,
        sectors=shortlist,
        warnings=list(raw_sector_result.warnings),
    )
    return SectorDiscoveryResult(
        as_of_date=as_of_date,
        market=market,
        sectors=sector_result,
        sector_universe=ranked,
        metadata={
            "top_n": len(shortlist),
            "sector_universe_size": len(ranked),
            "style_weights": sector_style_weights(market.regime),
            "rank_source": (
                "rule+ml"
                if active_ranker is not None and ml_weight > 0
                else "rule"
            ),
            "ml_weight": float(ml_weight) if active_ranker is not None else 0.0,
            "ml_model_path": str(ml_model_path) if ml_model_path else None,
            "method": (
                "Market Regime -> SW L1 full cross-section -> "
                "Momentum/Value/Dividend/Liquidity style scores -> "
                "regime-aware rule rank -> optional ML rerank -> Top-K sectors"
            ),
        },
    )




def run_research_pool(
    as_of_date: str,
    *,
    sector_top_n: int = 4,
    representatives_per_sector: int = 2,
    component_limit: int = 20,
    strict_pit: bool = True,
    ml_model_path: str | None = None,
    ml_weight: float = 0.5,
    ranker: SectorRanker | None = None,
    representative_selector=select_representative_stocks,
) -> ResearchPoolResult:
    """Bridge sector discovery to concrete single-stock research entry points.

    The representative layer is deliberately *not* an investment-quality model:
    it selects liquid, index-representative and data-complete names within each
    already-selected sector.  In strict PIT mode, recent/current component
    membership is required because the public SW component interface cannot
    reconstruct historical exits reliably.
    """

    if sector_top_n <= 0:
        raise ValueError("sector_top_n 必须大于 0")
    if representatives_per_sector <= 0:
        raise ValueError("representatives_per_sector 必须大于 0")

    discovery = run_discovery(
        as_of_date,
        top_n=sector_top_n,
        ml_model_path=ml_model_path,
        ml_weight=ml_weight,
        ranker=ranker,
    )
    _validate_component_membership_date(as_of_date, strict_pit)
    reps = representative_selector(
        discovery.sectors.sectors,
        as_of_date,
        representatives_per_sector=representatives_per_sector,
        component_limit=component_limit,
    )
    return ResearchPoolResult(
        as_of_date=as_of_date,
        discovery=discovery,
        representatives=reps,
        metadata={
            "sector_top_n": len(discovery.sectors.sectors),
            "representatives_per_sector": representatives_per_sector,
            "component_limit": component_limit,
            "strict_pit": strict_pit,
            "method": (
                "Sector Discovery -> within-sector representativeness/liquidity/"
                "relative-strength/data-coverage routing -> 7-Agent entry points"
            ),
        },
    )


def research_pool_markdown(result: ResearchPoolResult) -> str:
    discovery = result.discovery
    reps = result.representatives.representatives
    lines = [
        f"# A股行业发现与代表性研究池 — {result.as_of_date}",
        "",
        "> 本报告先确定行业研究优先级，再为每个行业选择少量代表性股票作为深度研究入口。",
        "> 代表性评分不等于投资评分；它不使用 ROE、利润增长或估值质量来判断未来收益。",
        "",
        "## 1. Market Regime",
        "",
        f"- Regime: **{discovery.market.regime}**",
        f"- Score: **{discovery.market.score:.1f}/100**",
        f"- Summary: {discovery.market.summary}",
        "",
        "## 2. Sector Research Shortlist",
        "",
        _df_to_markdown(discovery.sectors.sectors)
        if not discovery.sectors.sectors.empty
        else "无可用行业。",
        "",
        "## 3. Representative Research Entries",
        "",
        f"- 行业成分候选数: {result.representatives.universe_size}",
        f"- 完成历史/流动性校验: {result.representatives.scored_size}",
        f"- 最终研究入口: {len(reps)}",
        "",
        _df_to_markdown(
            reps.drop(columns=["research_context"], errors="ignore")
        ) if not reps.empty else "没有可用代表性股票。",
        "",
        "## 4. Representative Score Contract",
        "",
        "- 35% 行业权重：衡量公司对该行业指数的代表性。",
        "- 30% 流动性：使用 20 日平均成交额做行业内排名。",
        "- 20% 行业内相对强弱：20/60 日收益仅用于选择更有研究价值的活跃代表。",
        "- 15% 数据完整性：优先选择后续 7-Agent 能获得更完整证据的标的。",
        "- 不使用 PE/PB/ROE/利润增长做代表股评分，避免把研究路由重新变成隐藏的跨行业选股模型。",
        "",
        "## 5. 7-Agent Handoff",
        "",
        "每个代表股都附带 research_context。该上下文只描述“为什么进入研究池”，"
        "会被明确标记为 selection prior，而不是事实证据；Analyst 必须通过工具独立验证，"
        "并允许最终结论否定该行业先验。",
    ]
    warnings = (
        list(discovery.market.warnings)
        + list(discovery.sectors.warnings)
        + list(result.representatives.warnings)
    )
    if warnings:
        lines += ["", "## 数据/质量提示", ""] + [f"- {w}" for w in warnings]
    return "\n".join(lines).strip() + "\n"


def write_research_pool_report(
    result: ResearchPoolResult,
    output_dir: str | Path,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.discovery.market.indices.to_csv(out / "market_indices.csv", index=False)
    result.discovery.sector_universe.to_csv(out / "sector_ranking.csv", index=False)
    result.discovery.sectors.sectors.to_csv(out / "sector_shortlist.csv", index=False)
    result.representatives.representatives.to_csv(
        out / "representative_research_pool.csv",
        index=False,
    )
    report = out / "research_pool_report.md"
    report.write_text(research_pool_markdown(result), encoding="utf-8")
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "as_of_date": result.as_of_date,
                "market_regime": result.discovery.market.regime,
                "market_score": result.discovery.market.score,
                **result.discovery.metadata,
                **result.metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def discovery_markdown(result: SectorDiscoveryResult) -> str:
    market = result.market
    sectors = result.sectors.sectors
    weights = result.metadata.get("style_weights", {})

    lines = [
        f"# A股行业研究发现报告 — {result.as_of_date}",
        "",
        "> 本报告输出的是行业研究优先级，不是个股买入清单，也不构成投资建议或收益承诺。",
        "> Market Regime 只改变 Style 权重，不再把未进入 Top 行业的股票永久排除。",
        "",
        "## 1. 大盘环境",
        "",
        f"- Market Regime: **{market.regime}**",
        f"- 综合分: **{market.score:.1f}/100**",
        f"- 摘要: {market.summary}",
        "",
        _df_to_markdown(market.indices)
        if not market.indices.empty
        else "无可用指数数据。",
        "",
        "## 2. Sector Research Shortlist",
        "",
        (
            "- 全量行业横截面: "
            f"{result.metadata.get('sector_universe_size', len(result.sector_universe))} "
            "个申万一级行业"
        ),
        f"- 输出 Top-K: {len(sectors)}",
        f"- 排名来源: {result.metadata.get('rank_source', 'rule')}",
        (
            "- Regime Style 权重: "
            + ", ".join(
                f"{key}={float(value):.0%}"
                for key, value in weights.items()
            )
            if weights
            else "- Regime Style 权重: 无"
        ),
        "",
        _df_to_markdown(sectors) if not sectors.empty else "没有可用行业。",
        "",
        "## 3. 评分解释",
        "",
        "- Momentum: 1/20/60 日行业相对动量，允许成长/科技板块依靠趋势强度进入研究池。",
        "- Value: PE/PB 横截面便宜度，仅作为一种 Style，不再让所有行业都按低估值标准参加同一场考试。",
        "- Dividend: 行业股息率横截面排名，使高股息/防御型板块拥有独立优势。",
        "- Liquidity: 换手率与成交额占比，衡量行业资金活跃度。",
        "- Regime: Risk-On / Neutral / Risk-Off 只动态调整 Style 权重，不作为行业资格硬门槛。",
        "- Trend Penalty: 仅对持续弱趋势做软惩罚，不直接删除行业。",
        "- Optional ML: 可插入 LightGBM Ranker；模型分数先转为当日横截面百分位，再与 Rule Score 融合。",
        "",
        "## 4. 下一步",
        "",
        "从 Top 行业中选择代表性、流动性和数据覆盖较好的股票，再运行 python -m cli.main analyze 交给 7-Agent 做证据约束的单股深度研究。",
    ]

    warnings = list(market.warnings) + list(result.sectors.warnings)
    if warnings:
        lines += ["", "## 数据/质量提示", ""] + [f"- {w}" for w in warnings]
    return "\n".join(lines).strip() + "\n"


def write_discovery_report(
    result: SectorDiscoveryResult,
    output_dir: str | Path,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.market.indices.to_csv(out / "market_indices.csv", index=False)
    result.sector_universe.to_csv(out / "sector_ranking.csv", index=False)
    result.sectors.sectors.to_csv(out / "sector_shortlist.csv", index=False)
    report = out / "discovery_report.md"
    report.write_text(discovery_markdown(result), encoding="utf-8")
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "as_of_date": result.as_of_date,
                "market_regime": result.market.regime,
                "market_score": result.market.score,
                **result.metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


# ---------------------------------------------------------------------------
# Legacy stock discovery
# ---------------------------------------------------------------------------

def _validate_component_membership_date(as_of_date: str, strict_pit: bool) -> None:
    if not strict_pit:
        return
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    if as_of < date.today() - timedelta(days=7):
        raise ValueError(
            "严格 PIT 模式下，涉及申万个股成分的 Representative Pool / legacy-stock "
            "仅支持当前或最近交易日（7天内）。index_component_sw 只提供当前成分及计入日期，"
            "无法完整恢复历史退出成分；用于更早日期会产生幸存者偏差。"
            "纯 Sector Discovery 不依赖个股成分恢复，因此历史日期仍可运行。"
        )


def run_stock_discovery_legacy(
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
    """Retain the previous sector-hard-gate stock screen for comparison only."""

    market = analyze_market_regime(as_of_date)
    sector_result = analyze_sectors(
        as_of_date,
        market_regime=market.regime,
        top_n=max(sector_count, 10),
    )
    _validate_component_membership_date(as_of_date, strict_pit)
    top_sectors = sector_result.sectors.head(sector_count)
    components = load_sector_components(
        top_sectors,
        as_of_date,
        max_per_sector=per_sector,
    )
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
        as_of_date=as_of_date,
        market=market,
        sectors=sector_result,
        stocks=stocks,
        metadata={
            "legacy": True,
            "sector_count": sector_count,
            "per_sector": per_sector,
            "top_n": top_n,
            "max_shortlist_per_sector": max_shortlist_per_sector,
            "quality_pool_size": stocks.quality_pool_size,
            "quality_scored_size": stocks.quality_scored_size,
            "quality_weight": quality_weight,
            "quality_enabled": quality_enabled,
            "strict_pit": strict_pit,
            "method": (
                "LEGACY: Market -> hard Top sectors -> stock quant screen -> "
                "PIT quality screen -> sector soft cap"
            ),
        },
    )
