from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from .models import SectorRankingResult


def _check_akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("AKShare 未安装，请执行 `pip install -U akshare`") from exc
    return ak


def _as_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _normalize_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    rename = {
        "指数代码": "sector_code",
        "指数名称": "sector_name",
        "发布日期": "data_date",
        "收盘指数": "close",
        "涨跌幅": "change_pct",
        "换手率": "turnover",
        "市盈率": "pe",
        "市净率": "pb",
        "股息率": "dividend_yield",
        "成交额占比": "amount_share",
    }
    out = df.rename(columns=rename).copy()
    required = ["sector_code", "sector_name", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"申万行业数据缺少列: {missing}; 实际列={list(df.columns)}")
    for col in [
        "close",
        "change_pct",
        "turnover",
        "pe",
        "pb",
        "dividend_yield",
        "amount_share",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["sector_code"] = out["sector_code"].astype(str).str.replace(".SI", "", regex=False)
    out["sector_name"] = out["sector_name"].astype(str)
    if "data_date" in out.columns:
        out["data_date"] = pd.to_datetime(out["data_date"], errors="coerce")
    return out


def _fetch_day_near(
    target: date,
    *,
    fetcher: Callable[[str, str, str], pd.DataFrame] | None = None,
    max_back_days: int = 10,
):
    if fetcher is None:
        ak = _check_akshare()

        def fetcher(symbol: str, start_date: str, end_date: str):
            return ak.index_analysis_daily_sw(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )

    last_error: Exception | None = None
    for offset in range(max_back_days + 1):
        d = target - timedelta(days=offset)
        ymd = d.strftime("%Y%m%d")
        try:
            normalized = _normalize_snapshot(fetcher("一级行业", ymd, ymd))
            if not normalized.empty:
                return d, normalized
        except Exception as exc:
            last_error = exc
    suffix = f": {last_error}" if last_error else ""
    raise RuntimeError(f"无法取得 {target} 附近的申万一级行业日报{suffix}")


def _pct_rank(series: pd.Series, *, higher_better: bool = True) -> pd.Series:
    """Cross-sectional percentile score where 100 always means better."""

    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    out = pd.Series(50.0, index=s.index, dtype=float)
    if valid.empty or valid.nunique() <= 1:
        return out
    out.loc[valid.index] = valid.rank(
        method="average",
        pct=True,
        ascending=higher_better,
    ) * 100.0
    return out.clip(0.0, 100.0)


def sector_style_weights(market_regime: str) -> dict[str, float]:
    """Regime-aware style weights; regime changes preferences, not eligibility."""

    regime = market_regime.lower().replace("_", "-")
    if "risk-on" in regime:
        return {
            "momentum": 0.55,
            "valuation": 0.10,
            "dividend": 0.05,
            "liquidity": 0.30,
        }
    if "risk-off" in regime:
        return {
            "momentum": 0.25,
            "valuation": 0.20,
            "dividend": 0.35,
            "liquidity": 0.20,
        }
    return {
        "momentum": 0.40,
        "valuation": 0.20,
        "dividend": 0.15,
        "liquidity": 0.25,
    }


def _sector_trend_penalty(
    ret20: pd.Series,
    ret60: pd.Series,
    market_regime: str,
) -> pd.Series:
    """Soft penalty only; a sector is never removed solely because of trend."""

    r20 = pd.to_numeric(ret20, errors="coerce")
    r60 = pd.to_numeric(ret60, errors="coerce")
    penalty = pd.Series(0.0, index=r20.index, dtype=float)
    penalty += np.select(
        [r60 <= -0.15, r60 <= -0.10, r60 <= -0.05],
        [14.0, 9.0, 4.5],
        default=0.0,
    )
    penalty += np.select(
        [r20 <= -0.08, r20 <= -0.05, r20 <= -0.02],
        [6.0, 3.5, 1.5],
        default=0.0,
    )
    penalty += ((r20 < 0) & (r60 < 0)).astype(float) * 1.5

    regime = market_regime.lower().replace("_", "-")
    if "risk-off" in regime:
        penalty *= 1.10
    elif "risk-on" in regime:
        penalty *= 0.80
    return penalty.clip(0.0, 24.0)


def _style_labels(out: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    columns = {
        "Momentum": "momentum_score",
        "Value": "valuation_score",
        "Dividend": "dividend_score",
        "Liquidity": "liquidity_score",
    }
    score_frame = pd.DataFrame(
        {
            label: pd.to_numeric(out[column], errors="coerce").fillna(50.0)
            for label, column in columns.items()
        },
        index=out.index,
    )
    primary = score_frame.idxmax(axis=1)
    profile = score_frame.apply(
        lambda row: " + ".join(row.sort_values(ascending=False).index[:2]),
        axis=1,
    )
    return primary, profile


def rank_sector_snapshots(
    current: pd.DataFrame,
    anchor20: pd.DataFrame,
    anchor60: pd.DataFrame,
    *,
    market_regime: str = "Neutral",
) -> pd.DataFrame:
    """Rank all SW level-1 sectors with explicit style dimensions.

    The old implementation used sector ranking as a hard gate for downstream
    stock selection and then reused sector score again inside stock scoring.
    This function now treats sector discovery as an independent cross-sectional
    research problem.  Every sector stays eligible; Market Regime only changes
    style weights.
    """

    cur = _normalize_snapshot(current) if "指数代码" in current.columns else current.copy()
    a20 = _normalize_snapshot(anchor20) if "指数代码" in anchor20.columns else anchor20.copy()
    a60 = _normalize_snapshot(anchor60) if "指数代码" in anchor60.columns else anchor60.copy()

    a20 = a20[["sector_code", "close"]].rename(columns={"close": "close_20_anchor"})
    a60 = a60[["sector_code", "close"]].rename(columns={"close": "close_60_anchor"})
    out = (
        cur.merge(a20, on="sector_code", how="left")
        .merge(a60, on="sector_code", how="left")
    )
    out["ret_20d"] = out["close"] / out["close_20_anchor"] - 1.0
    out["ret_60d"] = out["close"] / out["close_60_anchor"] - 1.0

    p20 = _pct_rank(out["ret_20d"], higher_better=True)
    p60 = _pct_rank(out["ret_60d"], higher_better=True)
    p1 = _pct_rank(
        out.get("change_pct", pd.Series(index=out.index, dtype=float)),
        higher_better=True,
    )
    pturn = _pct_rank(
        out.get("turnover", pd.Series(index=out.index, dtype=float)),
        higher_better=True,
    )
    pamount = _pct_rank(
        out.get("amount_share", pd.Series(index=out.index, dtype=float)),
        higher_better=True,
    )

    pe = pd.to_numeric(
        out.get("pe", pd.Series(index=out.index, dtype=float)),
        errors="coerce",
    ).where(lambda s: s > 0)
    pb = pd.to_numeric(
        out.get("pb", pd.Series(index=out.index, dtype=float)),
        errors="coerce",
    ).where(lambda s: s > 0)
    dividend = pd.to_numeric(
        out.get("dividend_yield", pd.Series(index=out.index, dtype=float)),
        errors="coerce",
    ).where(lambda s: s >= 0)

    # Four explicit styles let high-growth/high-momentum sectors and high-yield
    # defensive sectors win for different reasons instead of taking one exam.
    out["momentum_score"] = (0.50 * p20 + 0.35 * p60 + 0.15 * p1).clip(0, 100)
    out["valuation_score"] = (
        0.60 * _pct_rank(pe, higher_better=False)
        + 0.40 * _pct_rank(pb, higher_better=False)
    ).clip(0, 100)
    out["dividend_score"] = _pct_rank(dividend, higher_better=True).clip(0, 100)
    out["liquidity_score"] = (0.65 * pturn + 0.35 * pamount).clip(0, 100)

    weights = sector_style_weights(market_regime)
    out["rule_score_raw"] = (
        weights["momentum"] * out["momentum_score"]
        + weights["valuation"] * out["valuation_score"]
        + weights["dividend"] * out["dividend_score"]
        + weights["liquidity"] * out["liquidity_score"]
    ).clip(0, 100)

    out["trend_penalty"] = _sector_trend_penalty(
        out["ret_20d"],
        out["ret_60d"],
        market_regime,
    )
    out["rule_score"] = (out["rule_score_raw"] - out["trend_penalty"]).clip(0, 100)

    # Compatibility aliases: old consumers expect sector_score(_raw).
    out["sector_score_raw"] = out["rule_score_raw"]
    out["sector_score"] = out["rule_score"]

    primary, profile = _style_labels(out)
    out["primary_style"] = primary
    out["style_profile"] = profile
    out["trend_quality"] = np.select(
        [
            (out["ret_20d"] > 0) & (out["ret_60d"] > 0),
            out["ret_60d"] <= -0.10,
            (out["ret_20d"] < 0) & (out["ret_60d"] < 0),
        ],
        ["strong", "weak", "downtrend"],
        default="mixed",
    )

    keep = [
        "sector_code",
        "sector_name",
        "close",
        "change_pct",
        "ret_20d",
        "ret_60d",
        "turnover",
        "amount_share",
        "pe",
        "pb",
        "dividend_yield",
        "momentum_score",
        "valuation_score",
        "dividend_score",
        "liquidity_score",
        "primary_style",
        "style_profile",
        "rule_score_raw",
        "trend_penalty",
        "rule_score",
        "sector_score_raw",
        "trend_quality",
        "sector_score",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep].sort_values("sector_score", ascending=False).reset_index(drop=True)


def analyze_sectors(
    as_of_date: str,
    *,
    market_regime: str = "Neutral",
    top_n: int = 10,
    fetcher=None,
) -> SectorRankingResult:
    as_of = _as_date(as_of_date)
    current_date, current = _fetch_day_near(as_of, fetcher=fetcher)
    d20, anchor20 = _fetch_day_near(as_of - timedelta(days=30), fetcher=fetcher)
    d60, anchor60 = _fetch_day_near(as_of - timedelta(days=90), fetcher=fetcher)
    ranked = rank_sector_snapshots(
        current,
        anchor20,
        anchor60,
        market_regime=market_regime,
    )
    if top_n > 0:
        ranked = ranked.head(top_n).reset_index(drop=True)
    return SectorRankingResult(
        as_of_date=as_of_date,
        current_data_date=current_date.isoformat(),
        anchor_20d_date=d20.isoformat(),
        anchor_60d_date=d60.isoformat(),
        sectors=ranked,
    )
