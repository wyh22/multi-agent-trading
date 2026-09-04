from __future__ import annotations

"""PIT-aware lightweight financial-quality screen for discovery candidates."""

import io
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Iterable
import pandas as pd

from tradingagents.dataflows.symbol_utils import normalize_a_share_symbol


def _check_baostock():
    try:
        import baostock as bs
    except ImportError as exc:
        raise RuntimeError("BaoStock 未安装，请执行 `pip install -U baostock`") from exc
    return bs


def _bs_code(symbol: str) -> tuple[str, str]:
    canonical=normalize_a_share_symbol(symbol)
    code,exchange=canonical.split(".",1)
    prefix={"SH":"sh","SZ":"sz","BJ":"bj"}.get(exchange.upper())
    if not prefix: raise ValueError(f"不支持的 A 股市场: {symbol}")
    return canonical,f"{prefix}.{code}"


@contextmanager
def _session():
    bs=_check_baostock()
    with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
        login=bs.login()
    if getattr(login,"error_code","1")!="0":
        raise ConnectionError(getattr(login,"error_msg","BaoStock login failed"))
    try: yield bs
    finally:
        try:
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()): bs.logout()
        except Exception: pass


def _frame(rs)->pd.DataFrame:
    if getattr(rs,"error_code","1")!="0":
        return pd.DataFrame()
    rows=[]
    while rs.next(): rows.append(rs.get_row_data())
    return pd.DataFrame(rows,columns=list(rs.fields))


def _latest_pit(df: pd.DataFrame,as_of_date: str)->pd.Series|None:
    if df is None or df.empty or "pubDate" not in df.columns:return None
    out=df.copy()
    pub=pd.to_datetime(out["pubDate"],errors="coerce")
    stat=pd.to_datetime(out.get("statDate"),errors="coerce") if "statDate" in out else pd.Series(pd.NaT,index=out.index)
    cutoff=pd.Timestamp(as_of_date).normalize()
    mask=pub.notna()&(pub.dt.normalize()<=cutoff)
    out=out.loc[mask].copy()
    if out.empty:return None
    out["_pub"]=pub.loc[out.index];out["_stat"]=stat.loc[out.index]
    return out.sort_values(["_stat","_pub"],ascending=False).iloc[0]


def _num(row,*names):
    if row is None:return float("nan")
    for name in names:
        if name in row.index:
            v=pd.to_numeric(pd.Series([row.get(name)]),errors="coerce").iloc[0]
            if pd.notna(v):return float(v)
    return float("nan")


def _quarters(as_of_date: str,n: int=8):
    d=pd.Timestamp(as_of_date);q=(d.month-1)//3+1;y=d.year
    out=[]
    for _ in range(n):
        out.append((y,q));q-=1
        if q==0:q=4;y-=1
    return out


def load_quality_metrics_batch_baostock(symbols: Iterable[str],as_of_date: str)->pd.DataFrame:
    rows=[]
    with _session() as bs:
        for symbol in dict.fromkeys(symbols):
            canonical,code=_bs_code(symbol)
            best=None
            for year,quarter in _quarters(as_of_date):
                profit=_latest_pit(_frame(bs.query_profit_data(code=code,year=year,quarter=quarter)),as_of_date)
                growth=_latest_pit(_frame(bs.query_growth_data(code=code,year=year,quarter=quarter)),as_of_date)
                cash=_latest_pit(_frame(bs.query_cash_flow_data(code=code,year=year,quarter=quarter)),as_of_date)
                balance=_latest_pit(_frame(bs.query_balance_data(code=code,year=year,quarter=quarter)),as_of_date)
                candidates=[r for r in (profit,growth,cash,balance) if r is not None]
                if not candidates:continue
                pub=max(str(r.get("pubDate","")) for r in candidates)
                stat=max(str(r.get("statDate","")) for r in candidates)
                net_profit=_num(profit,"netProfit")
                cfo=_num(cash,"CFOToOR","CFOToNP")
                if pd.isna(cfo):
                    cfo_net=_num(cash,"netCashFlowsOperAct")
                    cfo=cfo_net/net_profit if pd.notna(cfo_net) and pd.notna(net_profit) and net_profit!=0 else float("nan")
                best={
                    "ticker":canonical,"quality_period":stat,"quality_pub_date":pub,
                    "roe":_num(profit,"roeAvg","roe"),
                    "net_profit_yoy":_num(growth,"YOYNI","YOYNetProfit","netProfitYOY"),
                    "cfo_to_np":cfo,
                    "liability_to_asset":_num(balance,"liabilityToAsset","liabilityToAssetRatio"),
                }
                break
            if best:rows.append(best)
    return pd.DataFrame(rows)
