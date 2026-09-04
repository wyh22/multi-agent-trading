from __future__ import annotations

from typing import Callable, Iterable
import math
import numpy as np
import pandas as pd

from tradingagents.dataflows.baostock import load_factor_history_batch_baostock
from tradingagents.dataflows.symbol_utils import normalize_a_share_symbol
from .models import StockScreenResult
from .quality import load_quality_metrics_batch_baostock


def _check_akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("AKShare 未安装，请执行 `pip install -U akshare`") from exc
    return ak


def _canonical(code: str) -> str:
    return normalize_a_share_symbol(str(code).zfill(6))


def _is_excluded_name(name: str) -> bool:
    upper=str(name).upper().replace(" ","")
    return "ST" in upper or "退" in upper


def load_sector_components(
    sectors: pd.DataFrame, as_of_date: str, *,
    max_per_sector: int=35, component_fetcher: Callable[[str],pd.DataFrame]|None=None
)->pd.DataFrame:
    if component_fetcher is None:
        ak=_check_akshare()
        component_fetcher=lambda symbol: ak.index_component_sw(symbol=symbol)
    frames=[];as_of=pd.Timestamp(as_of_date).normalize()
    for _,sector in sectors.iterrows():
        code=str(sector["sector_code"]).replace(".SI","");name=str(sector["sector_name"])
        raw=component_fetcher(code)
        if raw is None or raw.empty:continue
        df=raw.rename(columns={"证券代码":"code","证券名称":"name","最新权重":"index_weight","计入日期":"entry_date"}).copy()
        if "code" not in df or "name" not in df:continue
        df["code"]=df["code"].astype(str).str.extract(r"(\d{6})",expand=False)
        df=df[df["code"].notna()&~df["name"].map(_is_excluded_name)]
        if "entry_date" in df:
            df["entry_date"]=pd.to_datetime(df["entry_date"],errors="coerce")
            df=df[df["entry_date"].isna()|(df["entry_date"]<=as_of)]
        if "index_weight" in df:
            df["index_weight"]=pd.to_numeric(df["index_weight"],errors="coerce")
            df=df.sort_values("index_weight",ascending=False)
        df=df.head(max_per_sector)
        df["ticker"]=df["code"].map(_canonical);df["sector_code"]=code;df["sector_name"]=name
        df["sector_score"]=float(sector.get("sector_score",50.0))
        keep=[c for c in ["ticker","code","name","sector_code","sector_name","sector_score","index_weight","entry_date"] if c in df]
        frames.append(df[keep])
    if not frames:
        return pd.DataFrame(columns=["ticker","code","name","sector_code","sector_name","sector_score"])
    out=pd.concat(frames,ignore_index=True)
    return out.sort_values(["sector_score","index_weight"],ascending=[False,False],na_position="last").drop_duplicates("ticker").reset_index(drop=True)


def _safe_return(close: pd.Series,periods: int)->float:
    s=pd.to_numeric(close,errors="coerce").dropna()
    if len(s)<=periods:return float("nan")
    base=float(s.iloc[-periods-1])
    return float(s.iloc[-1]/base-1.0) if base else float("nan")


def _max_drawdown(close: pd.Series,window: int=60)->float:
    s=pd.to_numeric(close,errors="coerce").dropna().tail(window)
    return float((s/s.cummax()-1.0).min()) if len(s)>=2 else float("nan")


def _metrics_from_history(frame: pd.DataFrame)->dict|None:
    if frame is None or frame.empty or "Close" not in frame:return None
    df=frame.sort_values("Date") if "Date" in frame else frame.copy()
    close=pd.to_numeric(df["Close"],errors="coerce").dropna()
    if len(close)<25:return None
    daily=close.pct_change().dropna();last=float(close.iloc[-1]);ma20=float(close.tail(20).mean())
    ma60=float(close.tail(60).mean()) if len(close)>=60 else float(close.mean())
    def latest(col):
        s=pd.to_numeric(df.get(col,pd.Series(dtype=float)),errors="coerce").dropna()
        return float(s.iloc[-1]) if not s.empty else float("nan")
    def mean20(col):
        s=pd.to_numeric(df.get(col,pd.Series(dtype=float)),errors="coerce").dropna().tail(20)
        return float(s.mean()) if not s.empty else float("nan")
    return {
        "last_close":last,"ret_5d":_safe_return(close,5),"ret_20d":_safe_return(close,20),"ret_60d":_safe_return(close,60),
        "trend_20":1.0 if last>=ma20 else 0.0,"trend_60":1.0 if last>=ma60 else 0.0,
        "vol_20d":float(daily.tail(20).std(ddof=0)*np.sqrt(252)) if len(daily) else float("nan"),
        "max_drawdown_60d":_max_drawdown(close,60),"avg_amount_20d":mean20("Amount"),
        "avg_turnover_20d":mean20("Turnover_Rate"),"pe_ttm":latest("PE_TTM"),"pb_mrq":latest("PB_MRQ"),
        "ps_ttm":latest("PS_TTM"),"is_st":latest("Is_ST"),
        "data_date":pd.to_datetime(df["Date"],errors="coerce").max().strftime("%Y-%m-%d") if "Date" in df else "unknown",
    }


def _pct(series: pd.Series,higher_better: bool=True)->pd.Series:
    s=pd.to_numeric(series,errors="coerce");out=pd.Series(50.0,index=s.index,dtype=float);valid=s.dropna()
    if len(valid)<2 or valid.nunique()<2:return out
    ranks=valid.rank(method="average",ascending=higher_better)
    out.loc[valid.index]=((ranks-1)/(len(valid)-1)*100).clip(0,100)
    return out


def _sector_pct(frame,values,*,higher_better):
    global_score=_pct(values,higher_better)
    if "sector_name" not in frame:return global_score
    result=global_score.copy()
    for _,idx in frame["sector_name"].fillna("UNKNOWN").astype(str).groupby(frame["sector_name"].fillna("UNKNOWN").astype(str)).groups.items():
        idx=list(idx);local=pd.to_numeric(values.loc[idx],errors="coerce")
        if local.notna().sum()>=3 and local.nunique(dropna=True)>=2:
            result.loc[idx]=_pct(local,higher_better)
    return result.fillna(50).clip(0,100)


def _weighted(parts):
    index=parts[0][0].index;num=pd.Series(0.0,index=index);den=pd.Series(0.0,index=index)
    for score,valid,w in parts:
        mask=valid.fillna(False)&score.notna();num.loc[mask]+=score.loc[mask]*w;den.loc[mask]+=w
    out=pd.Series(50.0,index=index);mask=den>0;out.loc[mask]=num.loc[mask]/den.loc[mask]
    return out.clip(0,100)


def score_stock_metrics(metrics: pd.DataFrame,market_regime: str)->pd.DataFrame:
    out=metrics.copy()
    p20=_pct(out["ret_20d"],True);p60=_pct(out["ret_60d"],True)
    ptrend=(pd.to_numeric(out["trend_20"],errors="coerce").fillna(0)+pd.to_numeric(out["trend_60"],errors="coerce").fillna(0))/2*100
    momentum=0.45*p20+0.35*p60+0.20*ptrend
    raw_pe=pd.to_numeric(out["pe_ttm"],errors="coerce");raw_pb=pd.to_numeric(out["pb_mrq"],errors="coerce")
    pe_valid=raw_pe>0;pb_valid=raw_pb>0
    pe=raw_pe.where(pe_valid);pb=raw_pb.where(pb_valid)
    out["valuation_quality_flag"]=np.select([raw_pe.isna(),raw_pe<0,raw_pe==0],["missing_pe","negative_pe","zero_pe"],default="positive_pe")
    pe_s=_sector_pct(out,pe,higher_better=False);pb_s=_sector_pct(out,pb,higher_better=False)
    valuation=_weighted([(pe_s,pe_valid,0.65),(pb_s,pb_valid,0.35)])
    amount=np.log1p(pd.to_numeric(out["avg_amount_20d"],errors="coerce").clip(lower=0))
    liquidity=0.75*_pct(amount,True)+0.25*_pct(out["avg_turnover_20d"],True)
    risk=0.55*_pct(out["vol_20d"],False)+0.45*_pct(out["max_drawdown_60d"],True)
    sector=pd.to_numeric(out["sector_score"],errors="coerce").fillna(50).clip(0,100)
    regime=market_regime.lower()
    weights={"momentum":.42,"valuation":.12,"liquidity":.14,"risk":.12,"sector":.20} if "risk-on" in regime else (
        {"momentum":.20,"valuation":.20,"liquidity":.10,"risk":.30,"sector":.20} if "risk-off" in regime else
        {"momentum":.32,"valuation":.17,"liquidity":.13,"risk":.18,"sector":.20})
    out["momentum_score"]=momentum.clip(0,100);out["valuation_score"]=valuation
    out["liquidity_score"]=liquidity.clip(0,100);out["risk_score"]=risk.clip(0,100)
    out["final_score"]=(weights["momentum"]*out["momentum_score"]+weights["valuation"]*out["valuation_score"]+weights["liquidity"]*out["liquidity_score"]+weights["risk"]*out["risk_score"]+weights["sector"]*sector).clip(0,100)
    out["quant_score"]=out["final_score"]
    return out.sort_values("final_score",ascending=False).reset_index(drop=True)


def _rank_quality(df: pd.DataFrame)->pd.DataFrame:
    out=df.copy()
    specs=[("roe",True,.30),("net_profit_yoy",True,.25),("cfo_to_np",True,.25),("liability_to_asset",False,.20)]
    parts=[];coverage=pd.Series(0.0,index=out.index)
    for col,higher,w in specs:
        vals=pd.to_numeric(out.get(col,pd.Series(index=out.index,dtype=float)),errors="coerce")
        valid=vals.notna();coverage+=valid.astype(float);parts.append((_sector_pct(out,vals,higher_better=higher),valid,w))
    out["quality_score"]=_weighted(parts);out["quality_coverage"]=coverage/len(specs)
    out["quality_flag"]=np.where(out["quality_coverage"]<.5,"low_quality_coverage","ok")
    return out


def _diversify(scored: pd.DataFrame,top_n: int,max_per_sector: int|None=None):
    ranked=scored.sort_values("final_score",ascending=False).reset_index(drop=True);target=min(top_n,len(ranked))
    cap=max_per_sector or max(1,math.ceil(max(1,target)*.30))
    chosen=[];counts={}
    while len(chosen)<target:
        added=False
        for idx,row in ranked.iterrows():
            if idx in chosen:continue
            sector=str(row.get("sector_name","UNKNOWN"))
            if counts.get(sector,0)>=cap:continue
            chosen.append(idx);counts[sector]=counts.get(sector,0)+1;added=True
            if len(chosen)>=target:break
        if not added:cap+=1
    return ranked.loc[chosen].sort_values("final_score",ascending=False).reset_index(drop=True),counts


def screen_stocks(
    components: pd.DataFrame,as_of_date: str,*,market_regime: str,top_n: int=10,
    max_shortlist_per_sector: int|None=None,
    history_loader: Callable[...,dict[str,pd.DataFrame]]=load_factor_history_batch_baostock,
    quality_loader: Callable[[Iterable[str],str],pd.DataFrame]|None=load_quality_metrics_batch_baostock,
    lookback_days: int=150,min_avg_amount: float=20_000_000.0,quality_pool_size: int|None=None,quality_weight: float=.25
)->StockScreenResult:
    if components.empty:
        return StockScreenResult(as_of_date,market_regime,pd.DataFrame(),0,0,["候选股票池为空"])
    histories=history_loader(components["ticker"].tolist(),as_of_date,lookback_days=lookback_days)
    rows=[]
    for _,meta in components.iterrows():
        m=_metrics_from_history(histories.get(meta["ticker"]))
        if not m or m.get("is_st")==1:continue
        if pd.notna(m.get("avg_amount_20d")) and float(m["avg_amount_20d"])<min_avg_amount:continue
        row=meta.to_dict();row.update(m);rows.append(row)
    metrics=pd.DataFrame(rows)
    if metrics.empty:
        return StockScreenResult(as_of_date,market_regime,metrics,len(components),0,["候选股历史数据不足或全部被过滤"])
    scored=score_stock_metrics(metrics,market_regime)
    pool_n=min(len(scored),max(top_n,quality_pool_size or max(top_n*3,24)))
    pool,_=_diversify(scored,pool_n,None)
    quality=pd.DataFrame()
    if quality_loader and not pool.empty:
        try:quality=quality_loader(pool["ticker"].astype(str).tolist(),as_of_date)
        except Exception:quality=pd.DataFrame()
    pool["quant_score"]=pool["final_score"]
    if not quality.empty:
        pool=pool.merge(quality.drop_duplicates("ticker"),on="ticker",how="left")
        pool=_rank_quality(pool)
        eff=float(max(0,min(1,quality_weight)))*pd.to_numeric(pool["quality_coverage"],errors="coerce").fillna(0)
        pool["quality_effective_weight"]=eff
        pool["final_score"]=pool["quant_score"]*(1-eff)+pool["quality_score"]*eff
    else:
        pool["quality_score"]=50.0;pool["quality_coverage"]=0.0;pool["quality_effective_weight"]=0.0;pool["quality_flag"]="quality_unavailable"
    result,counts=_diversify(pool,top_n,max_shortlist_per_sector)
    result["research_label"]=pd.cut(result["final_score"],[-np.inf,55,65,75,np.inf],labels=["Observe","Watch","Candidate","Strong Candidate"],right=False).astype(str)
    return StockScreenResult(as_of_date,market_regime,result,len(components),len(scored),[f"Shortlist 行业分布（Soft Cap）: {counts}"],counts,len(pool),int((pool["quality_coverage"]>0).sum()))
