"""Free A-share financial statements via AKShare/Sina Finance."""
from __future__ import annotations

import logging

import pandas as pd

from .asof import enrich_with_metadata
from .config import get_config
from .errors import NoMarketDataError, VendorNotConfiguredError
from .symbol_utils import normalize_a_share_symbol

logger = logging.getLogger(__name__)


def _check_akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise VendorNotConfiguredError("AKShare 未安装，请执行 `pip install -U akshare`。") from exc
    return ak


def _codes(ticker: str) -> tuple[str, str]:
    canonical = normalize_a_share_symbol(ticker)
    code, exchange = canonical.split(".", 1)
    sina_prefix = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(exchange.upper(), "")
    if not sina_prefix:
        raise NoMarketDataError(ticker, canonical, f"无法映射新浪市场前缀: {exchange}")
    return canonical, f"{sina_prefix}{code}"


def _asof(curr_date: str | None) -> pd.Timestamp:
    return pd.Timestamp(curr_date or pd.Timestamp.today().date()).normalize()


def _is_historical(curr_date: str | None) -> bool:
    return bool(curr_date and pd.Timestamp(curr_date).normalize() < pd.Timestamp.today().normalize())


def _parse_update_date(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.replace(" T", " ", regex=False).str.strip()
    return pd.to_datetime(raw, errors="coerce")


def _filter_sina(df: pd.DataFrame, curr_date: str | None, label: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "更新日期" not in df.columns:
        if get_config().get("strict_asof", True) and _is_historical(curr_date):
            raise NoMarketDataError(label, label, "新浪财报缺少更新日期，无法验证历史 PIT")
        return df.copy()
    cutoff = _asof(curr_date)
    update = _parse_update_date(df["更新日期"])
    mask = update.notna() & (update.dt.normalize() <= cutoff)
    out = df.loc[mask].copy()
    out["_effective_update"] = update.loc[mask]
    return out.sort_values("_effective_update", ascending=False).drop(columns="_effective_update")


def _filter_report_period(df: pd.DataFrame, curr_date: str | None, freq: str) -> pd.DataFrame:
    out = df.copy()
    report_col = next((c for c in ("报告日", "report_date") if c in out.columns), None)
    if report_col:
        dates = pd.to_datetime(out[report_col], errors="coerce")
        out = out.loc[dates.notna() & (dates.dt.normalize() <= _asof(curr_date))].copy()
        if freq.lower() == "annual":
            out = out.loc[dates.loc[out.index].dt.strftime("%m-%d").eq("12-31")]
    return out


def _latest_date(df: pd.DataFrame) -> str | None:
    for col in ("更新日期", "报告日", "report_date"):
        if col in df.columns and not df.empty:
            dates = _parse_update_date(df[col]) if col == "更新日期" else pd.to_datetime(df[col], errors="coerce")
            dates = dates.dropna()
            if not dates.empty:
                return dates.max().strftime("%Y-%m-%d")
    return None


def _select_wide_columns(df: pd.DataFrame, keywords: tuple[str, ...]) -> pd.DataFrame:
    fixed = [c for c in ("报告日", "类型", "更新日期", "币种") if c in df.columns]
    selected = list(fixed)
    for col in df.columns:
        if col not in selected and any(key in str(col) for key in keywords):
            selected.append(col)
    if len(selected) <= len(fixed):
        selected += [c for c in df.columns if c not in selected][:12]
    return df[selected[:22]].head(8)


def _statement(ticker, freq, curr_date, *, sina_symbol, title, keywords) -> str:
    ak = _check_akshare()
    canonical, sina_code = _codes(ticker)
    cutoff = _asof(curr_date)
    try:
        df = ak.stock_financial_report_sina(stock=sina_code, symbol=sina_symbol)
        if df is None or df.empty:
            raise ValueError("新浪财报返回空数据")
        df = _filter_sina(df, curr_date, f"{title}/{canonical}")
        df = _filter_report_period(df, curr_date, freq)
        if df.empty:
            raise ValueError("截止研究日无可用财报")
        display = _select_wide_columns(df, keywords)
        return enrich_with_metadata(
            f"# {canonical} {title} ({freq})\n"
            "# 数据源: AKShare / 新浪财经\n"
            "# PIT: 按更新日期截止过滤\n\n"
            + display.to_csv(index=False),
            vendor="akshare-sina",
            as_of_date=cutoff.strftime("%Y-%m-%d"),
            data_date=_latest_date(df),
        )
    except NoMarketDataError:
        raise
    except Exception as exc:
        logger.warning("%s 新浪财报失败: %s", canonical, exc)
        raise NoMarketDataError(
            ticker, canonical, f"AKShare/新浪财经 {title} 不可用: {exc}"
        ) from exc


_BALANCE_KEYS = ("货币资金", "应收", "存货", "流动资产", "固定资产", "资产总计", "总资产", "流动负债", "负债合计", "总负债", "股东权益", "所有者权益")
_INCOME_KEYS = ("营业收入", "营业总收入", "营业成本", "营业利润", "利润总额", "净利润", "归属于母公司", "基本每股收益")
_CASH_KEYS = ("经营活动产生的现金流量净额", "投资活动产生的现金流量净额", "筹资活动产生的现金流量净额", "现金及现金等价物净增加额", "期末现金")


def get_free_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    return _statement(ticker, freq, curr_date, sina_symbol="资产负债表", title="资产负债表", keywords=_BALANCE_KEYS)


def get_free_income_statement(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    return _statement(ticker, freq, curr_date, sina_symbol="利润表", title="利润表", keywords=_INCOME_KEYS)


def get_free_cashflow(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    return _statement(ticker, freq, curr_date, sina_symbol="现金流量表", title="现金流量表", keywords=_CASH_KEYS)
