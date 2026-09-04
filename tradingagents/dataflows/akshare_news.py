"""A-share news and management-holding context via AKShare public interfaces."""
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


def get_akshare_global_news(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """Return recent 财联社 global-market telegrams available by `curr_date`."""
    ak = _check_akshare()
    cutoff = pd.Timestamp(curr_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    lookback = max(1, int(look_back_days or 7))
    max_items = max(1, int(limit or 10))

    df = ak.stock_info_global_cls(symbol="全部")
    if df is None or df.empty:
        raise NoMarketDataError("global_news", "global_news", "财联社电报接口返回空数据")

    dates = pd.to_datetime(df.get("发布日期"), errors="coerce")
    if "发布时间" in df.columns:
        stamps = pd.to_datetime(
            dates.dt.strftime("%Y-%m-%d") + " " + df["发布时间"].astype(str),
            errors="coerce",
        )
    else:
        stamps = dates

    start = cutoff - pd.Timedelta(days=lookback)
    mask = stamps.notna() & (stamps >= start) & (stamps <= cutoff)
    work = df.loc[mask].copy()
    work["_published_at"] = stamps.loc[mask]
    work = work.sort_values("_published_at", ascending=False).head(max_items)
    if work.empty:
        raise NoMarketDataError(
            "global_news", "global_news",
            f"截止 {curr_date} 的最近 {lookback} 天内无可用财联社电报",
        )

    lines = ["## 财联社全球市场电报"]
    for _, row in work.iterrows():
        ts = pd.Timestamp(row["_published_at"]).strftime("%Y-%m-%d %H:%M:%S")
        title = str(row.get("标题", "") or "").strip()
        content = str(row.get("内容", "") or "").strip()
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"- [{ts}] {title}" + (f" — {content}" if content else ""))

    data_date = pd.Timestamp(work["_published_at"].max()).strftime("%Y-%m-%d")
    return enrich_with_metadata(
        "\n".join(lines),
        vendor="akshare-cls",
        as_of_date=curr_date,
        data_date=data_date,
    )


def get_akshare_insider_transactions(ticker: str, curr_date: str | None = None) -> str:
    """Return A-share management/related-person holding changes by the cutoff date."""
    ak = _check_akshare()
    canonical = normalize_a_share_symbol(ticker)
    code = canonical.split(".", 1)[0]
    cutoff = pd.Timestamp(curr_date or pd.Timestamp.today().date()).normalize()

    df = ak.stock_hold_management_detail_em()
    if df is None or df.empty:
        raise NoMarketDataError(ticker, canonical, "高管持股变动接口返回空数据")

    code_col = df["代码"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    dates = pd.to_datetime(df["日期"], errors="coerce")
    mask = code_col.eq(code) & dates.notna() & (dates.dt.normalize() <= cutoff)
    work = df.loc[mask].copy()
    work["_date"] = dates.loc[mask]
    work = work.sort_values("_date", ascending=False).head(50)
    if work.empty:
        raise NoMarketDataError(ticker, canonical, f"截止 {cutoff.date()} 无高管持股变动记录")

    keep = [
        c for c in (
            "日期", "代码", "名称", "变动人", "变动股数", "成交均价", "变动金额",
            "变动原因", "变动比例", "变动后持股数", "董监高人员姓名", "职务",
            "变动人与董监高的关系",
        )
        if c in work.columns
    ]
    display = work[keep].copy()
    return enrich_with_metadata(
        f"## {canonical} 高管及相关人员持股变动\n\n" + display.to_csv(index=False),
        vendor="akshare-eastmoney",
        as_of_date=cutoff.strftime("%Y-%m-%d"),
        data_date=work["_date"].max().strftime("%Y-%m-%d"),
    )
