"""Token-free China macro provider backed by AKShare public-data interfaces."""
from __future__ import annotations
import re
import pandas as pd
from .asof import enrich_with_metadata
from .errors import NoMarketDataError, VendorNotConfiguredError

_SERIES={
    "cpi":("macro_china_cpi","中国 CPI"),"ppi":("macro_china_ppi","中国 PPI"),
    "gdp":("macro_china_gdp","中国 GDP"),"real_gdp":("macro_china_gdp","中国 GDP"),
    "pmi":("macro_china_pmi","中国 PMI"),"lpr":("macro_china_lpr","中国 LPR"),
    "money_supply":("macro_china_money_supply","中国货币供应量"),"m2":("macro_china_money_supply","中国货币供应量"),
    "unemployment":("macro_china_urban_unemployment","中国城镇调查失业率"),
    "unemployment_rate":("macro_china_urban_unemployment","中国城镇调查失业率"),
}
_MAX_STALENESS_DAYS={"cpi":120,"ppi":120,"pmi":120,"lpr":150,"money_supply":120,"m2":120,"unemployment":150,"unemployment_rate":150,"gdp":450,"real_gdp":450}
_PERIOD_LAG_DAYS={"cpi":15,"ppi":15,"pmi":7,"money_supply":20,"m2":20,"unemployment":20,"unemployment_rate":20,"gdp":45,"real_gdp":45,"lpr":0}
_PERIOD_COLUMNS={"月份","统计月份","季度","年份","报告期"}

def _check_akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise VendorNotConfiguredError("AKShare 未安装，请执行 `pip install -U akshare`。") from exc
    return ak

def _parse_cn_date(value):
    if value is None or (isinstance(value,float) and pd.isna(value)): return None
    if isinstance(value,pd.Timestamp): return value
    text=str(value).strip()
    if not text: return None
    m=re.search(r"(?P<y>20\d{2})\s*(?:年|[-/.])\s*(?P<m>\d{1,2})\s*(?:月|[-/.])\s*(?P<d>\d{1,2})",text)
    if m:
        try:return pd.Timestamp(int(m.group("y")),int(m.group("m")),int(m.group("d")))
        except ValueError:pass
    m=re.search(r"(?P<y>20\d{2})\s*年?[-/.]?\s*(?P<m>\d{1,2})\s*月?",text)
    if m:
        try:return pd.Timestamp(int(m.group("y")),int(m.group("m")),1)
        except ValueError:pass
    qmap={"一":1,"二":2,"三":3,"四":4}
    q=re.search(r"(?P<y>20\d{2}).*?(?:第)?(?P<q>[一二三四1-4])\s*(?:季度|Q)",text,re.I) or re.search(r"(?P<y>20\d{2})\s*[Qq](?P<q>[1-4])",text)
    if q:
        qr=q.group("q"); quarter=qmap.get(qr,int(qr) if qr.isdigit() else 1)
        return pd.Timestamp(int(q.group("y")),quarter*3,1)+pd.offsets.MonthEnd(0)
    y=re.fullmatch(r"\s*(20\d{2})\s*年?\s*",text)
    if y:return pd.Timestamp(int(y.group(1)),12,31)
    parsed=pd.to_datetime(re.sub(r"[./]","-",text.replace("日","")),errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)

def _date_column(df):
    preferred=("发布日期","发布时间","更新日期","日期","月份","统计月份","统计时间","季度","年份","报告期","TRADE_DATE","时间","date","Date")
    return next((c for c in preferred if c in df.columns),None)

def get_akshare_macro_data(indicator: str,curr_date: str,look_back_days: int|None=None)->str:
    key=indicator.strip().lower().replace("-","_").replace(" ","_")
    if key not in _SERIES:
        return f"DATA_UNAVAILABLE: AKShare 中国宏观当前支持: {', '.join(sorted(_SERIES))}; 收到 {indicator!r}。"
    func_name,title=_SERIES[key]
    ak=_check_akshare(); df=getattr(ak,func_name)()
    if df is None or df.empty: raise NoMarketDataError(indicator,indicator,f"{func_name} 返回空数据")
    cutoff=pd.Timestamp(curr_date).normalize(); date_col=_date_column(df); work=df.copy(); data_date=None
    if date_col:
        period_dates=work[date_col].map(_parse_cn_date)
        lag=_PERIOD_LAG_DAYS.get(key,0) if date_col in _PERIOD_COLUMNS else 0
        available=period_dates.map(lambda x:(x+pd.Timedelta(days=lag)).normalize() if x is not None else pd.NaT)
        valid=available.notna()&(available<=cutoff)
        work=work.loc[valid].copy(); work["_period_date"]=period_dates.loc[valid]; work["_available_date"]=available.loc[valid]
        work=work.sort_values("_available_date")
        if look_back_days and not work.empty:
            work=work[work["_available_date"]>=cutoff-pd.Timedelta(days=int(look_back_days))]
        if not work.empty:
            latest_period=pd.Timestamp(work["_period_date"].dropna().max()); latest_available=pd.Timestamp(work["_available_date"].dropna().max())
            data_date=latest_period.strftime("%Y-%m-%d"); age=int((cutoff-latest_available).days); max_age=_MAX_STALENESS_DAYS.get(key,180)
            if age>max_age: raise NoMarketDataError(indicator,indicator,f"{func_name} 最新可用期为 {data_date}，距 {curr_date} 约 {age} 天，超过 {max_age} 天新鲜度阈值")
        work=work.drop(columns=["_period_date","_available_date"],errors="ignore")
    elif cutoff<pd.Timestamp.today().normalize():
        raise NoMarketDataError(indicator,indicator,f"{func_name} 缺少可解析日期列，不能用于历史 PIT")
    if work.empty: raise NoMarketDataError(indicator,indicator,f"截止 {curr_date} 无可用宏观数据")
    return enrich_with_metadata(f"## {title}\n# AKShare 接口: {func_name}\n# 日期字段: {date_col or 'unavailable'}\n\n"+work.tail(20).to_csv(index=False),vendor="akshare-macro",as_of_date=curr_date,data_date=data_date)
