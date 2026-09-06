from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Callable

import pandas as pd

from tradingagents.dataflows.symbol_utils import normalize_a_share_symbol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusCandidate:
    ticker: str
    title: str
    publish_date: str
    url: str
    doc_type: str


def _compact_title(value: str) -> str:
    return "".join(str(value or "").split()).lower()


def _is_primary_report(title: str, *, year: int, kind: str) -> bool:
    normalized = _compact_title(title)
    if any(word in normalized for word in ("摘要", "英文", "更正公告", "取消")):
        return False
    if kind == "annual":
        return f"{year}年年度报告" in normalized
    if kind == "semiannual":
        return (
            f"{year}年半年度报告" in normalized
            or f"{year}年中期报告" in normalized
        )
    if kind == "q1":
        return f"{year}年第一季度报告" in normalized
    raise ValueError(f"unsupported report kind: {kind}")


def _is_investor_relation(title: str) -> bool:
    normalized = _compact_title(title)
    return any(
        token in normalized
        for token in (
            "投资者关系活动记录",
            "投资者关系管理记录",
            "调研活动信息",
        )
    )


def _normalize_url(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        return "https:" + raw
    if raw.startswith("/"):
        return "https://static.cninfo.com.cn" + raw
    if raw.startswith("http://"):
        return "https://" + raw[len("http://") :]
    return raw


def _rows(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    work = df.copy()
    if "公告时间" in work.columns:
        work["_publish_date"] = pd.to_datetime(
            work["公告时间"],
            errors="coerce",
        )
    else:
        work["_publish_date"] = pd.NaT
    work = work.sort_values("_publish_date", ascending=False)
    return work.to_dict(orient="records")


def select_high_value_disclosures(
    df: pd.DataFrame,
    *,
    ticker: str,
    annual_year: int,
    interim_year: int,
    max_docs: int = 3,
) -> list[CorpusCandidate]:
    """Select a small high-value filing set for generic equity RAG.

    Priority:
    1) previous-year annual report;
    2) current-year semiannual report;
    3) latest current-year investor-relations record;
       fallback: current-year Q1 report.

    The logic is generic and does not contain stock/sector-specific keywords.
    """

    canonical = normalize_a_share_symbol(ticker)
    rows = _rows(df)
    selected: list[CorpusCandidate] = []

    def pick(predicate: Callable[[str], bool], doc_type: str):
        for row in rows:
            title = str(row.get("公告标题") or row.get("标题") or "").strip()
            if not predicate(title):
                continue
            url = _normalize_url(
                str(
                    row.get("公告链接")
                    or row.get("链接")
                    or row.get("url")
                    or ""
                )
            )
            stamp = pd.to_datetime(
                row.get("_publish_date") or row.get("公告时间"),
                errors="coerce",
            )
            if not url or pd.isna(stamp):
                continue
            return CorpusCandidate(
                ticker=canonical,
                title=title,
                publish_date=pd.Timestamp(stamp).date().isoformat(),
                url=url,
                doc_type=doc_type,
            )
        return None

    annual = pick(
        lambda title: _is_primary_report(
            title,
            year=annual_year,
            kind="annual",
        ),
        "annual_report",
    )
    if annual:
        selected.append(annual)

    semiannual = pick(
        lambda title: _is_primary_report(
            title,
            year=interim_year,
            kind="semiannual",
        ),
        "semiannual_report",
    )
    if semiannual:
        selected.append(semiannual)

    investor_relation = pick(
        _is_investor_relation,
        "investor_relation",
    )
    if investor_relation:
        selected.append(investor_relation)
    else:
        q1 = pick(
            lambda title: _is_primary_report(
                title,
                year=interim_year,
                kind="q1",
            ),
            "quarterly_report",
        )
        if q1:
            selected.append(q1)

    # Stable de-duplication in case the upstream title list contains duplicates.
    unique: dict[str, CorpusCandidate] = {}
    for item in selected:
        unique.setdefault(item.url, item)
    return list(unique.values())[: max(1, int(max_docs))]


def fetch_cninfo_disclosures(
    ticker: str,
    *,
    start_date: str,
    end_date: str,
    attempts: int = 2,
    retry_sleep_seconds: float = 1.5,
) -> pd.DataFrame:
    """Fetch the official CNInfo disclosure index with bounded retries."""

    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "AKShare is required for corpus discovery; install project agent dependencies."
        ) from exc

    canonical = normalize_a_share_symbol(ticker)
    code = canonical.split(".", 1)[0]
    last_error: Exception | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            result = ak.stock_zh_a_disclosure_report_cninfo(
                symbol=code,
                market="沪深京",
                keyword="",
                category="",
                start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
                end_date=pd.Timestamp(end_date).strftime("%Y%m%d"),
            )
            if result is None:
                return pd.DataFrame()
            return result
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt + 1 < max(1, int(attempts)):
                time.sleep(max(0.0, float(retry_sleep_seconds)))
    raise RuntimeError(
        f"CNInfo disclosure discovery failed for {canonical}: {last_error}"
    )


def default_reporting_years(today: date | None = None) -> tuple[int, int]:
    current = today or date.today()
    return current.year - 1, current.year
