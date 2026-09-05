"""Representative-stock selection for sector-first research.

This module does *not* try to find the globally best stocks.  It picks a small
number of liquid, representative and data-complete names from already selected
sectors so the expensive 7-Agent graph has concrete research entry points.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from tradingagents.dataflows.baostock import load_factor_history_batch_baostock

from .models import RepresentativeSelectionResult
from .screener import load_sector_components


def _pct(series: pd.Series) -> pd.Series:
    """0-100 percentile score where larger raw values are better."""

    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    out = pd.Series(50.0, index=values.index, dtype=float)
    if valid.empty or valid.nunique() <= 1:
        return out
    out.loc[valid.index] = valid.rank(method="average", pct=True) * 100.0
    return out.clip(0.0, 100.0)


def _safe_return(close: pd.Series, periods: int) -> float:
    values = pd.to_numeric(close, errors="coerce").dropna()
    if len(values) <= periods:
        return float("nan")
    base = float(values.iloc[-periods - 1])
    return float(values.iloc[-1] / base - 1.0) if base else float("nan")


def _history_metrics(frame: pd.DataFrame) -> dict | None:
    if frame is None or frame.empty or "Close" not in frame:
        return None

    df = frame.sort_values("Date") if "Date" in frame else frame.copy()
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < 25:
        return None

    def mean_tail(column: str, n: int = 20) -> float:
        values = pd.to_numeric(
            df.get(column, pd.Series(index=df.index, dtype=float)),
            errors="coerce",
        ).dropna().tail(n)
        return float(values.mean()) if not values.empty else float("nan")

    def latest(column: str) -> float:
        values = pd.to_numeric(
            df.get(column, pd.Series(index=df.index, dtype=float)),
            errors="coerce",
        ).dropna()
        return float(values.iloc[-1]) if not values.empty else float("nan")

    daily = close.pct_change().dropna()
    data_date = "unknown"
    if "Date" in df:
        parsed = pd.to_datetime(df["Date"], errors="coerce")
        if parsed.notna().any():
            data_date = parsed.max().strftime("%Y-%m-%d")

    return {
        "ret_20d": _safe_return(close, 20),
        "ret_60d": _safe_return(close, 60),
        "avg_amount_20d": mean_tail("Amount"),
        "avg_turnover_20d": mean_tail("Turnover_Rate"),
        "vol_20d": (
            float(daily.tail(20).std(ddof=0) * np.sqrt(252))
            if not daily.empty
            else float("nan")
        ),
        "is_st": latest("Is_ST"),
        "data_date": data_date,
        "history_points": int(len(close)),
    }


def _within_sector_score(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Rank a feature within each selected sector.

    Representative selection is intentionally within-sector: a bank and a
    semiconductor company do not compete on valuation or accounting structure.
    """

    result = pd.Series(50.0, index=frame.index, dtype=float)
    for _, indices in frame.groupby("sector_code", dropna=False).groups.items():
        idx = list(indices)
        result.loc[idx] = _pct(frame.loc[idx, column])
    return result


def _selection_reason(row: pd.Series) -> str:
    dimensions = {
        "行业代表性": float(row.get("weight_score", 50.0)),
        "流动性": float(row.get("liquidity_score", 50.0)),
        "行业内相对强弱": float(row.get("relative_strength_score", 50.0)),
        "数据完整性": float(row.get("data_coverage_score", 50.0)),
    }
    top = sorted(dimensions.items(), key=lambda item: item[1], reverse=True)[:2]
    return " + ".join(label for label, _ in top)


def _research_context(row: pd.Series) -> str:
    """Render provenance passed to 7-Agent as a prior, never as evidence."""

    sector_name = str(row.get("sector_name", "") or "")
    sector_code = str(row.get("sector_code", "") or "")
    style = str(row.get("style_profile", "") or row.get("primary_style", "") or "")
    reason = str(row.get("selection_reason", "") or "")
    score = float(row.get("representative_score", 0.0) or 0.0)

    return (
        "Research-origin context (selection provenance only; NOT investment evidence). "
        f"This ticker was selected as a representative research entry for "
        f"{sector_name}({sector_code}). "
        f"Sector style context: {style or 'unspecified'}. "
        f"Representative score: {score:.2f}/100; selection reason: {reason or 'unspecified'}. "
        "The score only reflects sector weight, liquidity, within-sector relative "
        "strength and data completeness. It does not prove valuation quality, "
        "fundamental quality, future return, or a Buy thesis. All Analysts must "
        "independently verify tool evidence and may reject the sector-selection prior."
    )


def select_representative_stocks(
    sectors: pd.DataFrame,
    as_of_date: str,
    *,
    representatives_per_sector: int = 2,
    component_limit: int = 20,
    min_avg_amount: float = 20_000_000.0,
    lookback_days: int = 120,
    component_fetcher: Callable[[str], pd.DataFrame] | None = None,
    history_loader: Callable[..., dict[str, pd.DataFrame]] = (
        load_factor_history_batch_baostock
    ),
) -> RepresentativeSelectionResult:
    """Select research representatives from each already-ranked sector.

    Score contract:
    - 35% current SW index weight: sector representativeness;
    - 30% 20-day average amount: liquidity;
    - 20% 20/60-day relative strength within the sector;
    - 15% data coverage.

    No PE/PB/ROE/profit-growth factor is used here.  The function is a research
    routing layer, not a second hidden stock-picking model.
    """

    if representatives_per_sector <= 0:
        raise ValueError("representatives_per_sector 必须大于 0")
    if component_limit < representatives_per_sector:
        raise ValueError("component_limit 不能小于 representatives_per_sector")
    if sectors is None or sectors.empty:
        return RepresentativeSelectionResult(
            as_of_date=as_of_date,
            representatives=pd.DataFrame(),
            universe_size=0,
            scored_size=0,
            warnings=["行业 Shortlist 为空，无法选择代表性股票"],
        )

    components = load_sector_components(
        sectors,
        as_of_date,
        max_per_sector=component_limit,
        component_fetcher=component_fetcher,
    )
    if components.empty:
        return RepresentativeSelectionResult(
            as_of_date=as_of_date,
            representatives=pd.DataFrame(),
            universe_size=0,
            scored_size=0,
            warnings=["所选行业没有可用成分股"],
        )

    # Carry discovery provenance into the representative layer without putting
    # the sector score itself into the representative score.
    sector_meta_cols = [
        column
        for column in [
            "sector_code",
            "primary_style",
            "style_profile",
            "rule_score",
            "ml_score",
            "sector_score",
        ]
        if column in sectors.columns
    ]
    sector_meta = sectors[sector_meta_cols].drop_duplicates("sector_code")
    components = components.drop(
        columns=[
            column
            for column in ["sector_score"]
            if column in components.columns
        ]
    ).merge(sector_meta, on="sector_code", how="left")

    histories = history_loader(
        components["ticker"].astype(str).tolist(),
        as_of_date,
        lookback_days=lookback_days,
    )

    rows: list[dict] = []
    for _, meta in components.iterrows():
        metrics = _history_metrics(histories.get(str(meta["ticker"])))
        if metrics is None or metrics.get("is_st") == 1:
            continue
        amount = metrics.get("avg_amount_20d")
        if pd.notna(amount) and float(amount) < float(min_avg_amount):
            continue
        row = meta.to_dict()
        row.update(metrics)
        rows.append(row)

    scored = pd.DataFrame(rows)
    if scored.empty:
        return RepresentativeSelectionResult(
            as_of_date=as_of_date,
            representatives=scored,
            universe_size=len(components),
            scored_size=0,
            warnings=["成分股历史数据不足或流动性过滤后为空"],
        )

    raw_weight = pd.to_numeric(
        scored.get("index_weight", pd.Series(index=scored.index, dtype=float)),
        errors="coerce",
    )
    scored["index_weight"] = raw_weight
    if "sector_score" not in scored.columns:
        scored["sector_score"] = 50.0
    scored["weight_score"] = _within_sector_score(
        scored.assign(index_weight_numeric=raw_weight),
        "index_weight_numeric",
    )

    log_amount = np.log1p(
        pd.to_numeric(scored["avg_amount_20d"], errors="coerce").clip(lower=0)
    )
    scored["amount_log"] = log_amount
    scored["liquidity_score"] = _within_sector_score(scored, "amount_log")

    scored["ret20_score"] = _within_sector_score(scored, "ret_20d")
    scored["ret60_score"] = _within_sector_score(scored, "ret_60d")
    scored["relative_strength_score"] = (
        0.65 * scored["ret20_score"] + 0.35 * scored["ret60_score"]
    ).clip(0.0, 100.0)

    coverage_columns = [
        "index_weight",
        "ret_20d",
        "ret_60d",
        "avg_amount_20d",
        "avg_turnover_20d",
        "vol_20d",
    ]
    coverage = pd.DataFrame(
        {
            column: pd.to_numeric(
                scored.get(column, pd.Series(index=scored.index, dtype=float)),
                errors="coerce",
            ).notna()
            for column in coverage_columns
        },
        index=scored.index,
    )
    scored["data_coverage_score"] = coverage.mean(axis=1) * 100.0

    scored["representative_score"] = (
        0.35 * scored["weight_score"]
        + 0.30 * scored["liquidity_score"]
        + 0.20 * scored["relative_strength_score"]
        + 0.15 * scored["data_coverage_score"]
    ).clip(0.0, 100.0)

    selected_parts = []
    for _, group in scored.groupby("sector_code", sort=False):
        selected_parts.append(
            group.sort_values(
                ["representative_score", "index_weight"],
                ascending=[False, False],
                na_position="last",
            ).head(representatives_per_sector)
        )
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values(
        ["sector_score", "representative_score"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    selected["selection_reason"] = selected.apply(_selection_reason, axis=1)
    selected["research_context"] = selected.apply(_research_context, axis=1)
    selected["research_label"] = "Representative Research Entry"

    keep = [
        "ticker",
        "code",
        "name",
        "sector_code",
        "sector_name",
        "primary_style",
        "style_profile",
        "sector_score",
        "index_weight",
        "ret_20d",
        "ret_60d",
        "avg_amount_20d",
        "avg_turnover_20d",
        "history_points",
        "data_date",
        "weight_score",
        "liquidity_score",
        "relative_strength_score",
        "data_coverage_score",
        "representative_score",
        "selection_reason",
        "research_label",
        "research_context",
    ]
    keep = [column for column in keep if column in selected.columns]

    warnings = [
        (
            "代表股评分仅用于分配研究入口，不使用估值/财务质量因子，"
            "不得解释为预期收益或投资评级。"
        )
    ]
    return RepresentativeSelectionResult(
        as_of_date=as_of_date,
        representatives=selected[keep],
        universe_size=len(components),
        scored_size=len(scored),
        warnings=warnings,
    )
