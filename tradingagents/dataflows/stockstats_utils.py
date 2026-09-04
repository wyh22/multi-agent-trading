"""Technical-indicator helpers backed by BaoStock A-share OHLCV."""
from __future__ import annotations

from typing import Annotated

import pandas as pd
from stockstats import wrap

from .baostock import load_ohlcv_baostock
from .errors import NoMarketDataError


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return data
    out = data.copy()
    if "Date" not in out.columns:
        out = out.reset_index()
        if "Date" not in out.columns and "date" in out.columns:
            out = out.rename(columns={"date": "Date"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date")
    price_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in out.columns]
    out[price_cols] = out[price_cols].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=["Close"]).copy()
    out[price_cols] = out[price_cols].ffill().bfill()
    return out.reset_index(drop=True)


def load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Load a multi-year A-share OHLCV window ending at `curr_date`."""
    data = load_ohlcv_baostock(symbol, curr_date, lookback_days=5 * 366)
    cleaned = _clean_dataframe(data)
    if cleaned is None or cleaned.empty:
        raise NoMarketDataError(symbol, symbol, f"截止 {curr_date} 无可用 BaoStock OHLCV 数据")
    cutoff = pd.Timestamp(curr_date).normalize()
    return cleaned[cleaned["Date"] <= cutoff].reset_index(drop=True)


def filter_financials_by_date(data: pd.DataFrame, curr_date: str) -> pd.DataFrame:
    """Drop statement-period columns after the research cutoff date."""
    if not curr_date or data is None or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date)
    parsed = pd.to_datetime(data.columns, errors="coerce")
    return data.loc[:, parsed <= cutoff]


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "A-share ticker"],
        indicator: Annotated[str, "stockstats indicator name"],
        curr_date: Annotated[str, "research cutoff date, YYYY-mm-dd"],
    ):
        data = load_ohlcv(symbol, curr_date)
        df = wrap(data.copy())
        df[indicator]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        cutoff = pd.Timestamp(curr_date).normalize()
        rows = df[df["Date"].dt.normalize() == cutoff]
        if rows.empty:
            return "N/A: Not a trading day (weekend or holiday)"
        value = rows.iloc[-1][indicator]
        return "N/A" if pd.isna(value) else value


_INDICATOR_DESCRIPTIONS = {
    "close_50_sma": "50 SMA: medium-term trend and dynamic support/resistance.",
    "close_200_sma": "200 SMA: long-term trend benchmark.",
    "close_10_ema": "10 EMA: responsive short-term trend measure.",
    "macd": "MACD: momentum from the difference between fast and slow EMAs.",
    "macds": "MACD signal line.",
    "macdh": "MACD histogram.",
    "rsi": "RSI: momentum/overbought-oversold indicator.",
    "boll": "Bollinger middle band.",
    "boll_ub": "Bollinger upper band.",
    "boll_lb": "Bollinger lower band.",
    "atr": "ATR: volatility measured by average true range.",
    "vwma": "VWMA: volume-weighted moving average.",
    "mfi": "MFI: price-and-volume money-flow momentum indicator.",
}


def _get_stock_stats_bulk(symbol: str, indicator: str, curr_date: str) -> dict[str, str]:
    data = load_ohlcv(symbol, curr_date)
    df = wrap(data.copy())
    df[indicator]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    result: dict[str, str] = {}
    for _, row in df.iterrows():
        value = row[indicator]
        result[str(row["Date"])] = "N/A" if pd.isna(value) else str(value)
    return result


def get_stockstats_indicator(symbol: str, indicator: str, curr_date: str) -> str:
    try:
        return str(StockstatsUtils.get_stock_stats(symbol, indicator, curr_date))
    except NoMarketDataError:
        raise
    except Exception:
        return ""


def get_stock_stats_indicators_window(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int = 30,
) -> str:
    """Return one supported indicator over a calendar look-back window."""
    indicator = indicator.strip().lower()
    if indicator not in _INDICATOR_DESCRIPTIONS:
        raise ValueError(
            f"Indicator {indicator} is not supported. "
            f"Please choose from: {list(_INDICATOR_DESCRIPTIONS)}"
        )

    cutoff = pd.Timestamp(curr_date).normalize()
    start = cutoff - pd.Timedelta(days=max(1, int(look_back_days)))
    values = _get_stock_stats_bulk(symbol, indicator, curr_date)

    lines = []
    current = cutoff
    while current >= start:
        key = current.strftime("%Y-%m-%d")
        lines.append(f"{key}: {values.get(key, 'N/A: Not a trading day (weekend or holiday)')}")
        current -= pd.Timedelta(days=1)

    return (
        f"## {indicator} values from {start.strftime('%Y-%m-%d')} "
        f"to {cutoff.strftime('%Y-%m-%d')}:\n\n"
        + "\n".join(lines)
        + "\n\n"
        + _INDICATOR_DESCRIPTIONS[indicator]
    )
