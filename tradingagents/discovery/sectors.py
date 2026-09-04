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
        "指数代码": "sector_code", "指数名称": "sector_name", "发布日期": "data_date",
        "收盘指数": "close", "涨跌幅": "change_pct", "换手率": "turnover",
        "市盈率": "pe", "市净率": "pb", "股息率": "dividend_yield", "成交额占比": "amount_share",
    }
    out = df.rename(columns=rename).copy()
    required = ["sector_code", "sector_name", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"申万行业数据缺少列: {missing}; 实际列={list(df.columns)}")
    for col in ["close", "change_pct", "turnover", "pe", "pb", "dividend_yield", "amount_share"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["sector_code"] = out["sector_code"].astype(str).str.replace(".SI", "", regex=False)
    out["sector_name"] = out["sector_name"].astype(str)
    if "data_date" in out.columns:
        out["data_date"] = pd.to_datetime(out["data_date"], errors="coerce")
    return out


def _fetch_day_near(target: date, *, fetcher: Callable[[str, str, str], pd.DataFrame] | None = None, max_back_days: int = 10):
    if fetcher is None:
        ak = _check_akshare()
        def fetcher(symbol: str, start_date: str, end_date: str):
            return ak.index_analysis_daily_sw(symbol=symbol, start_date=start_date, end_date=end_date)

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
    s = pd.to_numeric(series, errors="coerce")
    ranked = s.rank(pct=True, ascending=higher_better) * 100.0
    if not higher_better:
        ranked = s.rank(pct=True, ascending=False) * 100.0
    return ranked.fillna(50.0)


def _sector_trend_penalty(ret20: pd.Series, ret60: pd.Series, market_regime: str) -> pd.Series:
    r20 = pd.to_numeric(ret20, errors="coerce")
    r60 = pd.to_numeric(ret60, errors="coerce")
    penalty = pd.Series(0.0, index=r20.index, dtype=float)
    penalty += np.select([r60 <= -0.15, r60 <= -0.10, r60 <= -0.05], [18.0, 12.0, 6.0], default=0.0)
    penalty += np.select([r20 <= -0.08, r20 <= -0.05, r20 <= -0.02], [8.0, 5.0, 2.5], default=0.0)
    penalty += ((r20 < 0) & (r60 < 0)).astype(float) * 2.0
    regime = market_regime.lower()
    if "risk-off" in regime or "risk_off" in regime:
        penalty *= 1.20
    elif "risk-on" in regime or "risk_on" in regime:
        penalty *= 0.85
    return penalty.clip(0.0, 30.0)


def rank_sector_snapshots(current: pd.DataFrame, anchor20: pd.DataFrame, anchor60: pd.DataFrame, *, market_regime: str = "Neutral") -> pd.DataFrame:
    cur = _normalize_snapshot(current) if "指数代码" in current.columns else current.copy()
    a20 = _normalize_snapshot(anchor20) if "指数代码" in anchor20.columns else anchor20.copy()
    a60 = _normalize_snapshot(anchor60) if "指数代码" in anchor60.columns else anchor60.copy()
    a20 = a20[["sector_code", "close"]].rename(columns={"close": "close_20_anchor"})
    a60 = a60[["sector_code", "close"]].rename(columns={"close": "close_60_anchor"})
    out = cur.merge(a20, on="sector_code", how="left").merge(a60, on="sector_code", how="left")
    out["ret_20d"] = out["close"] / out["close_20_anchor"] - 1.0
    out["ret_60d"] = out["close"] / out["close_60_anchor"] - 1.0

    p20 = _pct_rank(out["ret_20d"], higher_better=True)
    p60 = _pct_rank(out["ret_60d"], higher_better=True)
    p1 = _pct_rank(out.get("change_pct", pd.Series(index=out.index, dtype=float)), higher_better=True)
    pturn = _pct_rank(out.get("turnover", pd.Series(index=out.index, dtype=float)), higher_better=True)
    pe = pd.to_numeric(out.get("pe", pd.Series(index=out.index, dtype=float)), errors="coerce").where(lambda s: s > 0)
    pb = pd.to_numeric(out.get("pb", pd.Series(index=out.index, dtype=float)), errors="coerce").where(lambda s: s > 0)
    pval = 0.6 * _pct_rank(pe, higher_better=False) + 0.4 * _pct_rank(pb, higher_better=False)

    regime = market_regime.lower()
    if "risk-on" in regime or "risk_on" in regime:
        weights = (0.42, 0.28, 0.10, 0.12, 0.08)
    elif "risk-off" in regime or "risk_off" in regime:
        weights = (0.25, 0.20, 0.05, 0.15, 0.35)
    else:
        weights = (0.35, 0.25, 0.08, 0.12, 0.20)

    out["sector_score_raw"] = (
        weights[0] * p20 + weights[1] * p60 + weights[2] * p1 + weights[3] * pturn + weights[4] * pval
    ).clip(0, 100)
    out["trend_penalty"] = _sector_trend_penalty(out["ret_20d"], out["ret_60d"], market_regime)
    out["sector_score"] = (out["sector_score_raw"] - out["trend_penalty"]).clip(0, 100)
    out["momentum_score"] = (0.6 * p20 + 0.4 * p60).clip(0, 100)
    out["valuation_score"] = pval.clip(0, 100)
    out["trend_quality"] = np.select(
        [(out["ret_20d"] > 0) & (out["ret_60d"] > 0), (out["ret_60d"] <= -0.10), (out["ret_20d"] < 0) & (out["ret_60d"] < 0)],
        ["strong", "weak", "downtrend"], default="mixed",
    )
    keep = ["sector_code","sector_name","close","change_pct","ret_20d","ret_60d","turnover","pe","pb","dividend_yield","momentum_score","valuation_score","sector_score_raw","trend_penalty","trend_quality","sector_score"]
    keep = [c for c in keep if c in out.columns]
    return out[keep].sort_values("sector_score", ascending=False).reset_index(drop=True)


def analyze_sectors(as_of_date: str, *, market_regime: str = "Neutral", top_n: int = 10, fetcher=None) -> SectorRankingResult:
    as_of = _as_date(as_of_date)
    current_date, current = _fetch_day_near(as_of, fetcher=fetcher)
    d20, anchor20 = _fetch_day_near(as_of - timedelta(days=30), fetcher=fetcher)
    d60, anchor60 = _fetch_day_near(as_of - timedelta(days=90), fetcher=fetcher)
    ranked = rank_sector_snapshots(current, anchor20, anchor60, market_regime=market_regime)
    if top_n > 0:
        ranked = ranked.head(top_n).reset_index(drop=True)
    return SectorRankingResult(as_of_date=as_of_date,current_data_date=current_date.isoformat(),anchor_20d_date=d20.isoformat(),anchor_60d_date=d60.isoformat(),sectors=ranked)
