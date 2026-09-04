from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

from tradingagents.dataflows.baostock import load_index_ohlcv_baostock

from .models import MarketRegimeResult


INDEX_UNIVERSE = {
    "上证综指": "000001.SH",
    "沪深300": "000300.SH",
    "中证500": "000905.SH",
    "中证1000": "000852.SH",
    "创业板指": "399006.SZ",
}


def _safe_return(close: pd.Series, periods: int) -> float:
    values = pd.to_numeric(close, errors="coerce").dropna()
    if len(values) <= periods:
        return float("nan")
    base = float(values.iloc[-periods - 1])
    last = float(values.iloc[-1])
    if base == 0:
        return float("nan")
    return last / base - 1.0


def _max_drawdown(close: pd.Series, window: int = 60) -> float:
    s = pd.to_numeric(close, errors="coerce").dropna().tail(window)
    if len(s) < 2:
        return float("nan")
    peak = s.cummax()
    dd = s / peak - 1.0
    return float(dd.min())


def _index_metrics(frame: pd.DataFrame) -> dict[str, float | str]:
    if frame.empty or "Close" not in frame:
        raise ValueError("指数行情为空或缺少 Close 列")
    df = frame.copy().sort_values("Date") if "Date" in frame else frame.copy()
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < 20:
        raise ValueError("指数行情不足 20 个交易日")

    daily = close.pct_change().dropna()
    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean()) if len(close) >= 60 else float(close.mean())
    last = float(close.iloc[-1])
    vol20 = float(daily.tail(20).std(ddof=0) * math.sqrt(252)) if len(daily) else float("nan")

    return {
        "close": last,
        "ret_5d": _safe_return(close, 5),
        "ret_20d": _safe_return(close, 20),
        "ret_60d": _safe_return(close, 60),
        "ma20_gap": (last / ma20 - 1.0) if ma20 else float("nan"),
        "ma60_gap": (last / ma60 - 1.0) if ma60 else float("nan"),
        "vol_20d": vol20,
        "max_drawdown_60d": _max_drawdown(close, 60),
        "data_date": (
            pd.to_datetime(df["Date"], errors="coerce").max().strftime("%Y-%m-%d")
            if "Date" in df and pd.to_datetime(df["Date"], errors="coerce").notna().any()
            else "unknown"
        ),
    }


def _score_index(row: pd.Series | dict) -> float:
    get = row.get
    ret20 = float(get("ret_20d", 0.0) or 0.0)
    ret60 = float(get("ret_60d", 0.0) or 0.0)
    gap20 = float(get("ma20_gap", 0.0) or 0.0)
    gap60 = float(get("ma60_gap", 0.0) or 0.0)
    vol = float(get("vol_20d", 0.25) or 0.25)
    dd = float(get("max_drawdown_60d", -0.10) or -0.10)

    score = 50.0
    score += float(np.clip(ret20 * 220.0, -14.0, 14.0))
    score += float(np.clip(ret60 * 100.0, -12.0, 12.0))
    score += 6.0 if gap20 >= 0 else -6.0
    score += 6.0 if gap60 >= 0 else -6.0
    score -= float(np.clip(max(vol - 0.25, 0.0) * 25.0, 0.0, 7.0))
    score -= float(np.clip(max(abs(min(dd, 0.0)) - 0.10, 0.0) * 30.0, 0.0, 7.0))
    return float(np.clip(score, 0.0, 100.0))


def classify_regime(score: float) -> str:
    if score >= 60:
        return "Risk-On"
    if score <= 40:
        return "Risk-Off"
    return "Neutral"


def analyze_market_regime(
    as_of_date: str,
    *,
    loader: Callable[..., pd.DataFrame] = load_index_ohlcv_baostock,
    lookback_days: int = 220,
) -> MarketRegimeResult:
    rows: list[dict] = []
    warnings: list[str] = []

    for name, symbol in INDEX_UNIVERSE.items():
        try:
            frame = loader(symbol, as_of_date, lookback_days=lookback_days)
            metrics = _index_metrics(frame)
            metrics.update({"index_name": name, "symbol": symbol})
            metrics["index_score"] = _score_index(metrics)
            rows.append(metrics)
        except Exception as exc:
            warnings.append(f"{name}({symbol}) 获取/计算失败: {exc}")

    if not rows:
        raise RuntimeError("没有可用的大盘指数数据，无法判断 Market Regime")

    df = pd.DataFrame(rows)
    score = float(df["index_score"].mean())
    regime = classify_regime(score)
    up20 = int((pd.to_numeric(df["ret_20d"], errors="coerce") > 0).sum())
    total = len(df)
    avg20 = float(pd.to_numeric(df["ret_20d"], errors="coerce").mean())
    avg60 = float(pd.to_numeric(df["ret_60d"], errors="coerce").mean())
    summary = (
        f"{regime}（综合分 {score:.1f}/100）；"
        f"{up20}/{total} 个核心指数 20 日收益为正，"
        f"核心指数平均 20 日收益 {avg20:.2%}、60 日收益 {avg60:.2%}。"
    )
    cols = [
        "index_name", "symbol", "data_date", "close", "ret_5d", "ret_20d",
        "ret_60d", "ma20_gap", "ma60_gap", "vol_20d", "max_drawdown_60d",
        "index_score",
    ]
    return MarketRegimeResult(
        as_of_date=as_of_date,
        regime=regime,
        score=round(score, 2),
        indices=df[cols].sort_values("index_score", ascending=False).reset_index(drop=True),
        summary=summary,
        warnings=warnings,
    )
