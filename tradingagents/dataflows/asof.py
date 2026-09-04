"""asof 模块。"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Iterable

import pandas as pd

from .config import get_config

logger = logging.getLogger(__name__)

_KNOWN_DATA_DATE_COLS = {"trade_date", "Date", "date", "Trading_date", "交易日期"}
_KNOWN_END_DATE_COLS = {"end_date", "EndDate", "endDate", "REPORT_DATE", "报告期"}
_KNOWN_ANN_DATE_COLS = {
    "ann_date", "AnnDate", "annDate", "披露日期",
    "f_ann_date", "FAnnDate", "f_annDate", "实际披露日期",
    "NOTICE_DATE", "UPDATE_DATE", "公告日期",
}


def _parse_date(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            if value.endswith(".0") and value[:-2].isdigit():
                value = value[:-2]
            if value.isdigit() and len(value) in {6, 8}:
                fmt = "%Y%m%d" if len(value) == 8 else "%y%m%d"
                parsed = pd.to_datetime(value, format=fmt, errors="coerce")
                return None if pd.isna(parsed) else pd.Timestamp(parsed)
        if isinstance(value, (datetime, pd.Timestamp)):
            return pd.Timestamp(value)
        parsed = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(parsed) else pd.Timestamp(parsed)
    except Exception:
        return None


def _has_any(df: pd.DataFrame, cols: set[str]) -> str | None:
    for col in df.columns:
        if col in cols:
            return col
    return None


def _existing_columns(df: pd.DataFrame, candidates: Iterable[str | None]) -> list[str]:
    out: list[str] = []
    for col in candidates:
        if col and col in df.columns and col not in out:
            out.append(col)
    return out


def _strict_enabled(strict: bool | None) -> bool:
    if strict is not None:
        return bool(strict)
    return bool(get_config().get("strict_asof", True))


def _parse_cutoff(trade_date: str, *, label: str, strict: bool) -> pd.Timestamp | None:
    cutoff = _parse_date(trade_date)
    if cutoff is not None:
        return cutoff.normalize()
    message = f"[asof] 无法解析 trade_date={trade_date!r}，无法验证 {label} 的 PIT 安全性。"
    if strict:
        raise PITVerificationError(message)
    logger.warning(message)
    return None


def _verify_filtered_dates(parsed_dates, cutoff, *, label, trade_date, cutoff_kind):
    valid = parsed_dates.dropna()
    future = valid[valid > cutoff]
    if future.empty:
        return
    raise FutureDataError(
        label=label, trade_date=trade_date, cutoff_kind=cutoff_kind,
        n_rows=len(future), sample="\n".join(str(v) for v in future.head(3).tolist()),
    )


def filter_asof_rows(
    df: pd.DataFrame, trade_date: str, *,
    data_date_col: str | None = None, ann_date_col: str | None = None,
    end_date_col: str | None = None, strict: bool | None = None, label: str = "data",
) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    strict_enabled = _strict_enabled(strict)
    cutoff = _parse_cutoff(trade_date, label=label, strict=strict_enabled)
    if cutoff is None:
        return df

    ann_actual = ann_date_col or _has_any(df, _KNOWN_ANN_DATE_COLS)
    data_actual = data_date_col or _has_any(df, _KNOWN_DATA_DATE_COLS)
    end_actual = end_date_col or _has_any(df, _KNOWN_END_DATE_COLS)
    if ann_actual and ann_actual in df.columns:
        cutoff_col, cutoff_kind = ann_actual, "ann_date"
    elif data_actual and data_actual in df.columns:
        cutoff_col, cutoff_kind = data_actual, "data_date"
    else:
        message = (
            f"[asof] {label} 仅发现报告期列 {end_actual!r}；报告期不代表披露日，无法验证 PIT 安全性。"
            if end_actual else f"[asof] {label} 没有可识别的数据日期/披露日期列，无法验证 PIT 安全性。"
        )
        if strict_enabled:
            raise PITVerificationError(message)
        logger.warning(message)
        return df

    parsed = df[cutoff_col].map(_parse_date)
    valid = parsed.notna()
    past = valid & (parsed <= cutoff)
    result = df.loc[past].copy()
    result["_asof_parsed"] = parsed.loc[past]
    result = result.sort_values("_asof_parsed", ascending=False)
    if strict_enabled:
        _verify_filtered_dates(result["_asof_parsed"], cutoff, label=label, trade_date=trade_date, cutoff_kind=cutoff_kind)
    return result.drop(columns=["_asof_parsed"]).reset_index(drop=True)


def filter_disclosed_rows(
    df: pd.DataFrame, trade_date: str, *,
    f_ann_date_col: str | None = None, ann_date_col: str | None = None,
    end_date_col: str | None = None, strict: bool | None = None, label: str = "financials",
) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    strict_enabled = _strict_enabled(strict)
    cutoff = _parse_cutoff(trade_date, label=label, strict=strict_enabled)
    if cutoff is None:
        return df
    disclosure_cols = _existing_columns(df, [
        f_ann_date_col, ann_date_col,
        "f_ann_date","FAnnDate","f_annDate","实际披露日期",
        "ann_date","AnnDate","annDate","披露日期","NOTICE_DATE","UPDATE_DATE","公告日期",
    ])
    if not disclosure_cols:
        end_actual = end_date_col or _has_any(df, _KNOWN_END_DATE_COLS)
        message = f"[asof] {label} 缺少披露日期列" + (f"（仅有报告期 {end_actual!r}）" if end_actual else "") + "，无法安全地做历史财报 PIT 过滤。"
        if strict_enabled:
            raise PITVerificationError(message)
        logger.warning(message)
        return df
    parsed_frame = pd.concat([df[col].map(_parse_date).rename(col) for col in disclosure_cols], axis=1)
    for col in parsed_frame.columns:
        parsed_frame[col] = pd.to_datetime(parsed_frame[col], errors="coerce")
    effective_ann = parsed_frame.apply(lambda row: row.dropna().max() if row.notna().any() else pd.NaT, axis=1)
    effective_ann = pd.to_datetime(effective_ann, errors="coerce")
    past = effective_ann.notna() & (effective_ann <= cutoff)
    result = df.loc[past].copy()
    result["_effective_ann"] = effective_ann.loc[past]
    result = result.sort_values("_effective_ann", ascending=False)
    if strict_enabled:
        _verify_filtered_dates(result["_effective_ann"], cutoff, label=label, trade_date=trade_date, cutoff_kind="effective_disclosure_date")
    return result.drop(columns=["_effective_ann"]).reset_index(drop=True)


def latest_trade_date(df: pd.DataFrame, trade_date: str) -> str | None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    data_col = _has_any(df, _KNOWN_DATA_DATE_COLS)
    if not data_col:
        return None
    parsed = df[data_col].map(_parse_date).dropna()
    if parsed.empty:
        return None
    cutoff = _parse_date(trade_date)
    valid = parsed[parsed <= cutoff] if cutoff is not None else parsed
    return None if valid.empty else str(valid.max().date())


class PITVerificationError(Exception):
    pass


class FutureDataError(Exception):
    def __init__(self, label: str, trade_date: str, cutoff_kind: str, n_rows: int, sample: str):
        self.label, self.trade_date, self.cutoff_kind = label, trade_date, cutoff_kind
        self.n_rows, self.sample = n_rows, sample
        super().__init__(
            f"[strict_asof] {label}: 分析日 {trade_date} 后仍残留 {n_rows} 条未来数据"
            f"（按 {cutoff_kind} 判断）。样本：\n{sample}"
        )


def enrich_with_metadata(payload: Any, *, vendor: str, as_of_date: str, data_date: str | None = None, quality_flag: str = "ok") -> Any:
    meta_line = f"# Meta: vendor={vendor}; as_of_date={as_of_date}; data_date={data_date or 'N/A'}; quality_flag={quality_flag}\n"
    if isinstance(payload, str):
        return meta_line + payload if payload.startswith("#") else meta_line + "\n" + payload
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["_meta"] = {"vendor": vendor, "as_of_date": as_of_date, "data_date": data_date, "quality_flag": quality_flag}
        return payload
    return payload
