"""CNInfo official-announcement adapter."""
from __future__ import annotations
import pandas as pd
from .asof import enrich_with_metadata
from .errors import NoMarketDataError, VendorNotConfiguredError
from .symbol_utils import normalize_a_share_symbol

def _check_akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise VendorNotConfiguredError("AKShare 未安装，请执行 `pip install -U akshare`。") from exc
    return ak

def get_cninfo_news(ticker: str,start_date: str,end_date: str,limit: int=50,curr_date: str|None=None)->str:
    ak=_check_akshare()
    canonical=normalize_a_share_symbol(ticker)
    code=canonical.split(".")[0]
    cutoff=min(pd.Timestamp(end_date),pd.Timestamp(curr_date or end_date)).normalize()
    start=pd.Timestamp(start_date).normalize()
    df=ak.stock_zh_a_disclosure_report_cninfo(symbol=code,market="沪深京",keyword="",category="",start_date=start.strftime("%Y%m%d"),end_date=cutoff.strftime("%Y%m%d"))
    if df is None or df.empty:
        raise NoMarketDataError(ticker,canonical,"巨潮资讯在该 PIT 窗口内无公告")
    if "公告时间" in df.columns:
        dates=pd.to_datetime(df["公告时间"],errors="coerce")
        mask=dates.notna()&(dates.dt.normalize()>=start)&(dates.dt.normalize()<=cutoff)
        df=df.loc[mask].copy()
    if df.empty:
        raise NoMarketDataError(ticker,canonical,"巨潮资讯过滤后无 PIT-safe 公告")
    df=df.head(max(1,int(limit)))
    lines=[f"## {canonical} 巨潮资讯正式公告"]
    for _,row in df.iterrows():
        lines.append(f"- [{row.get('公告时间','')}] {row.get('公告标题','')} {row.get('公告链接','')}".strip())
    valid_dates=pd.to_datetime(df.get("公告时间"),errors="coerce") if "公告时间" in df else pd.Series(dtype="datetime64[ns]")
    data_date=valid_dates.dropna().max().strftime("%Y-%m-%d") if not valid_dates.dropna().empty else None
    return enrich_with_metadata("\n".join(lines),vendor="cninfo",as_of_date=cutoff.strftime("%Y-%m-%d"),data_date=data_date)
