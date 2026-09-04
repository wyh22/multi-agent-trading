"""BaoStock A-share PIT-safe data adapter."""
from __future__ import annotations
import io, logging, threading
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Annotated
import pandas as pd

from .asof import enrich_with_metadata
from .errors import NoMarketDataError, VendorNotConfiguredError
from .symbol_utils import normalize_a_share_symbol

logger = logging.getLogger(__name__)
_BS_LOCK = threading.RLock()
_DAILY_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
_OHLCV_FIELDS = "date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg"


def _check_baostock():
    try:
        import baostock as bs
    except ImportError as exc:
        raise VendorNotConfiguredError("BaoStock 未安装。请执行 `pip install -U baostock`；BaoStock 无需 API Token。") from exc
    return bs


def _to_baostock_code(symbol: str) -> tuple[str, str]:
    canonical = normalize_a_share_symbol(symbol)
    code, exchange = canonical.split(".", 1)
    prefix = {"SH":"sh","SZ":"sz","BJ":"bj"}.get(exchange.upper())
    if not prefix:
        raise NoMarketDataError(symbol, canonical, f"BaoStock 不支持交易所 {exchange}")
    return canonical, f"{prefix}.{code}"


def _adjustflag(adjust: str | None) -> str:
    return {"hfq":"1","qfq":"2","":"3","none":"3","raw":"3"}.get((adjust or "").strip().lower(), "2")


@contextmanager
def _session():
    bs = _check_baostock()
    with _BS_LOCK:
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            login = bs.login()
        if getattr(login, "error_code", "1") != "0":
            raise ConnectionError(f"BaoStock 登录失败: {getattr(login, 'error_msg', 'unknown error')}")
        try:
            yield bs
        finally:
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    bs.logout()
            except Exception:
                logger.debug("BaoStock logout failed", exc_info=True)


def _result_to_frame(rs) -> pd.DataFrame:
    if getattr(rs, "error_code", "1") != "0":
        raise ConnectionError(f"BaoStock 查询失败: {getattr(rs, 'error_msg', 'unknown error')}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=list(rs.fields))


def _query_history(symbol, start_date, end_date, *, fields, adjust="qfq"):
    canonical, bs_code = _to_baostock_code(symbol)
    with _session() as bs:
        rs = bs.query_history_k_data_plus(bs_code, fields, start_date=start_date, end_date=end_date, frequency="d", adjustflag=_adjustflag(adjust))
        df = _result_to_frame(rs)
    if df.empty:
        raise NoMarketDataError(symbol, canonical, f"{start_date} 至 {end_date} 无 BaoStock 数据")
    return canonical, df


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    rename = {"date":"Date","open":"Open","high":"High","low":"Low","close":"Close","preclose":"Pre_Close","volume":"Volume","amount":"Amount","turn":"Turnover_Rate","pctChg":"Change_Pct","peTTM":"PE_TTM","pbMRQ":"PB_MRQ","psTTM":"PS_TTM","pcfNcfTTM":"PCF_NCF_TTM","tradestatus":"Trade_Status","isST":"Is_ST"}
    out = df.rename(columns=rename).copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for col in ["Open","High","Low","Close","Pre_Close","Volume","Amount","Turnover_Rate","Change_Pct","PE_TTM","PB_MRQ","PS_TTM","PCF_NCF_TTM"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def load_ohlcv_baostock(symbol: str, curr_date: str, lookback_days: int = 450, adjust: str = "qfq") -> pd.DataFrame:
    cutoff = pd.Timestamp(curr_date).normalize()
    start = (cutoff - pd.Timedelta(days=max(lookback_days, 30))).strftime("%Y-%m-%d")
    canonical, raw = _query_history(symbol, start, cutoff.strftime("%Y-%m-%d"), fields=_OHLCV_FIELDS, adjust=adjust)
    df = _normalize_history(raw)
    df = df[df["Date"] <= cutoff]
    if df.empty:
        raise NoMarketDataError(symbol, canonical, f"截止 {curr_date} 无 OHLCV 数据")
    return df[[c for c in ("Date","Open","High","Low","Close","Volume") if c in df]].copy()


def load_index_ohlcv_baostock(index_code: str, curr_date: str, lookback_days: int = 220) -> pd.DataFrame:
    cutoff = pd.Timestamp(curr_date).normalize()
    start = (cutoff - pd.Timedelta(days=max(lookback_days, 90))).strftime("%Y-%m-%d")
    canonical, raw = _query_history(index_code, start, cutoff.strftime("%Y-%m-%d"), fields="date,code,open,high,low,close,preclose,volume,amount,adjustflag", adjust="")
    df = _normalize_history(raw)
    df = df[df["Date"] <= cutoff]
    if df.empty:
        raise NoMarketDataError(index_code, canonical, f"截止 {curr_date} 无指数行情")
    return df[[c for c in ["Date","Open","High","Low","Close","Pre_Close","Volume","Amount"] if c in df]].copy()


def load_factor_history_batch_baostock(symbols: list[str], curr_date: str, lookback_days: int = 150, adjust: str = "qfq") -> dict[str, pd.DataFrame]:
    cutoff = pd.Timestamp(curr_date).normalize()
    start = (cutoff - pd.Timedelta(days=max(lookback_days, 90))).strftime("%Y-%m-%d")
    result = {}
    with _session() as bs:
        for symbol in dict.fromkeys(symbols):
            try:
                canonical, bs_code = _to_baostock_code(symbol)
                rs = bs.query_history_k_data_plus(bs_code, _DAILY_FIELDS, start_date=start, end_date=cutoff.strftime("%Y-%m-%d"), frequency="d", adjustflag=_adjustflag(adjust))
                raw = _result_to_frame(rs)
                if raw.empty: continue
                df = _normalize_history(raw)
                df = df[df["Date"] <= cutoff]
                if not df.empty: result[canonical] = df.reset_index(drop=True)
            except Exception as exc:
                logger.warning("BaoStock 批量因子行情跳过 %s: %s", symbol, exc)
    return result


def get_baostock_stock_data(symbol: Annotated[str,"A股代码"], start_date: str, end_date: str, adjust: str = "qfq", curr_date: str | None = None) -> str:
    cutoff = min(pd.Timestamp(end_date), pd.Timestamp(curr_date or end_date)).normalize()
    canonical, raw = _query_history(symbol, start_date, cutoff.strftime("%Y-%m-%d"), fields=_OHLCV_FIELDS, adjust=adjust)
    df = _normalize_history(raw)
    df = df[(df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= cutoff)]
    if df.empty: raise NoMarketDataError(symbol, canonical, "无交易数据")
    latest = df["Date"].max().strftime("%Y-%m-%d")
    render = df.copy(); render["Date"] = render["Date"].dt.strftime("%Y-%m-%d")
    return enrich_with_metadata(f"# {canonical} A股日线数据 (BaoStock)\n\n"+render.to_csv(index=False), vendor="baostock", as_of_date=cutoff.strftime("%Y-%m-%d"), data_date=latest)


def get_baostock_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    cutoff = pd.Timestamp(curr_date or pd.Timestamp.today().date()).normalize()
    start = (cutoff - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    canonical, raw = _query_history(ticker, start, cutoff.strftime("%Y-%m-%d"), fields=_DAILY_FIELDS, adjust="")
    df = _normalize_history(raw); df = df[df["Date"] <= cutoff]
    if "Trade_Status" in df.columns:
        tradable = df[df["Trade_Status"].astype(str) == "1"]
        if not tradable.empty: df = tradable
    if df.empty: raise NoMarketDataError(ticker, canonical, "无估值数据")
    row = df.iloc[-1]; data_date = row["Date"].strftime("%Y-%m-%d")
    def fmt(name, digits=4):
        value=row.get(name)
        return "unavailable" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"
    lines=[f"## {canonical} 基础面/估值上下文 (BaoStock)",f"- 最近交易日: {data_date}",f"- 收盘价: {fmt('Close',2)}",f"- 换手率(%): {fmt('Turnover_Rate')}",f"- PE(TTM): {fmt('PE_TTM')}",f"- PB(MRQ): {fmt('PB_MRQ')}",f"- PS(TTM): {fmt('PS_TTM')}",f"- PCF(NCF, TTM): {fmt('PCF_NCF_TTM')}"]
    return enrich_with_metadata("\n".join(lines),vendor="baostock",as_of_date=cutoff.strftime("%Y-%m-%d"),data_date=data_date)
