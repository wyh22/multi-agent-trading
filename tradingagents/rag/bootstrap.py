from __future__ import annotations

import html
import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Callable
from urllib.parse import parse_qs, urlparse

import pandas as pd

from tradingagents.dataflows.symbol_utils import normalize_a_share_symbol

logger = logging.getLogger(__name__)

CNINFO_HOST = "https://www.cninfo.com.cn"
CNINFO_STATIC_HOST = "https://static.cninfo.com.cn"
CNINFO_SEARCH_URL = f"{CNINFO_HOST}/new/hisAnnouncement/query"
CNINFO_STOCK_JSON_URL = f"{CNINFO_HOST}/new/data/szse_stock.json"
CNINFO_SEARCH_REFERER = (
    f"{CNINFO_HOST}/new/commonUrl/pageOfSearch?url=disclosure/list/search"
)
CNINFO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Origin": CNINFO_HOST,
    "Referer": CNINFO_SEARCH_REFERER,
}


@dataclass(frozen=True)
class CorpusCandidate:
    ticker: str
    title: str
    publish_date: str
    url: str
    doc_type: str


def _compact_title(value: str) -> str:
    cleaned = re.sub(r"<[^>]+>", "", str(value or ""))
    return "".join(html.unescape(cleaned).split()).lower()


def _display_title(value: str) -> str:
    cleaned = re.sub(r"<[^>]+>", "", str(value or ""))
    return " ".join(html.unescape(cleaned).split())


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


def _cninfo_static_url_from_adjunct(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        return "https:" + raw
    if raw.startswith("http://"):
        raw = "https://" + raw[len("http://") :]
    if raw.startswith("https://"):
        return raw
    return f"{CNINFO_STATIC_HOST}/{raw.lstrip('/')}"


def _cninfo_static_url_from_detail(value: str) -> str:
    parsed = urlparse(str(value or "").strip())
    if "cninfo.com.cn" not in parsed.netloc or "/new/disclosure/detail" not in parsed.path:
        return ""
    query = parse_qs(parsed.query)
    announcement_id = (query.get("announcementId") or [""])[0].strip()
    announcement_time = (query.get("announcementTime") or [""])[0].strip()
    publish_date = announcement_time[:10]
    if not announcement_id or not re.match(r"^\d{4}-\d{2}-\d{2}$", publish_date):
        return ""
    return f"{CNINFO_STATIC_HOST}/finalpage/{publish_date}/{announcement_id}.PDF"


def _normalize_url(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    detail_pdf_url = _cninfo_static_url_from_detail(raw)
    if detail_pdf_url:
        return detail_pdf_url
    if raw.startswith("//"):
        return "https:" + raw
    if raw.startswith("/"):
        return f"{CNINFO_STATIC_HOST}{raw}"
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

    def pick(
        predicate: Callable[[str], bool],
        doc_type: str,
        *,
        year_filter: int | None = None,
    ):
        for row in rows:
            title = _display_title(row.get("公告标题") or row.get("标题") or "")
            if not predicate(title):
                continue
            stamp = pd.to_datetime(
                row.get("_publish_date") or row.get("公告时间"),
                errors="coerce",
            )
            if pd.isna(stamp):
                continue
            if year_filter is not None and pd.Timestamp(stamp).year != year_filter:
                continue
            url = _normalize_url(
                str(
                    row.get("公告链接")
                    or row.get("链接")
                    or row.get("url")
                    or row.get("adjunctUrl")
                    or ""
                )
            )
            if not url:
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
        year_filter=interim_year,
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


def _fmt_cninfo_date(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _load_cninfo_org_id(code: str):
    import requests

    response = requests.get(
        CNINFO_STOCK_JSON_URL,
        headers=CNINFO_HEADERS,
        timeout=(10, 30),
    )
    response.raise_for_status()
    payload = response.json()
    for item in payload.get("stockList", []):
        if str(item.get("code") or "").zfill(6) == code:
            return str(item.get("orgId") or "").strip()
    return ""


def _cninfo_timestamp_to_string(value) -> str:
    stamp = pd.to_datetime(value, unit="ms", utc=True, errors="coerce")
    if pd.isna(stamp):
        stamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(stamp):
        return ""
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return pd.Timestamp(stamp).strftime("%Y-%m-%d %H:%M:%S")


def _format_cninfo_announcements(announcements: list[dict]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for item in announcements:
        title = _display_title(item.get("announcementTitle") or "")
        publish_time = _cninfo_timestamp_to_string(item.get("announcementTime"))
        adjunct_url = item.get("adjunctUrl") or ""
        static_url = _cninfo_static_url_from_adjunct(adjunct_url)
        if not static_url:
            announcement_id = str(item.get("announcementId") or "").strip()
            if announcement_id and publish_time:
                static_url = (
                    f"{CNINFO_STATIC_HOST}/finalpage/"
                    f"{publish_time[:10]}/{announcement_id}.PDF"
                )
        rows.append(
            {
                "代码": str(item.get("secCode") or ""),
                "简称": str(item.get("secName") or ""),
                "公告标题": title,
                "公告时间": publish_time,
                "公告链接": static_url,
                "详情链接": (
                    f"{CNINFO_HOST}/new/disclosure/detail?"
                    f"stockCode={item.get('secCode')}&"
                    f"announcementId={item.get('announcementId')}&"
                    f"orgId={item.get('orgId')}&"
                    f"announcementTime={publish_time}"
                ),
                "announcementId": str(item.get("announcementId") or ""),
                "orgId": str(item.get("orgId") or ""),
                "adjunctUrl": str(adjunct_url),
            }
        )
    return pd.DataFrame(rows)


def _fetch_cninfo_disclosures_direct(
    code: str,
    *,
    start_date: str,
    end_date: str,
    page_size: int = 30,
    max_pages: int = 20,
) -> pd.DataFrame:
    import requests

    org_id = _load_cninfo_org_id(code)
    stock_item = f"{code},{org_id}" if org_id else code
    base_payload = {
        "pageNum": "1",
        "pageSize": str(max(1, int(page_size))),
        "column": "szse",
        "tabName": "fulltext",
        "plate": "",
        "stock": stock_item,
        "searchkey": "",
        "secid": "",
        "category": "",
        "trade": "",
        "seDate": f"{_fmt_cninfo_date(start_date)}~{_fmt_cninfo_date(end_date)}",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }

    session = requests.Session()
    session.headers.update(CNINFO_HEADERS)
    response = session.post(
        CNINFO_SEARCH_URL,
        data=base_payload,
        timeout=(10, 30),
    )
    response.raise_for_status()
    payload = response.json()
    total = int(payload.get("totalAnnouncement") or 0)
    page_count = min(
        max(1, math.ceil(total / int(base_payload["pageSize"]))),
        max(1, int(max_pages)),
    )
    announcements = list(payload.get("announcements") or [])

    for page_num in range(2, page_count + 1):
        page_payload = dict(base_payload, pageNum=str(page_num))
        response = session.post(
            CNINFO_SEARCH_URL,
            data=page_payload,
            timeout=(10, 30),
        )
        response.raise_for_status()
        page_payload_json = response.json()
        announcements.extend(page_payload_json.get("announcements") or [])

    return _format_cninfo_announcements(announcements)


def _fetch_cninfo_disclosures_via_akshare(
    code: str,
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "AKShare is required for fallback corpus discovery; install project agent dependencies."
        ) from exc

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


def fetch_cninfo_disclosures(
    ticker: str,
    *,
    start_date: str,
    end_date: str,
    attempts: int = 2,
    retry_sleep_seconds: float = 1.5,
) -> pd.DataFrame:
    """Fetch the official CNInfo disclosure index with bounded retries.

    The primary path queries CNInfo directly and keeps downloadable static PDF
    URLs. AKShare remains only as a compatibility fallback because its public
    dataframe exposes CNInfo detail-page URLs, which are not reliable document
    download targets for ingestion.
    """

    canonical = normalize_a_share_symbol(ticker)
    code = canonical.split(".", 1)[0]
    last_error: Exception | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            result = _fetch_cninfo_disclosures_direct(
                code,
                start_date=start_date,
                end_date=end_date,
            )
            if result is None:
                return pd.DataFrame()
            return result
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt + 1 < max(1, int(attempts)):
                time.sleep(max(0.0, float(retry_sleep_seconds)))

    try:
        logger.warning(
            "direct CNInfo discovery failed for %s; falling back to AKShare: %s",
            canonical,
            last_error,
        )
        return _fetch_cninfo_disclosures_via_akshare(
            code,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"CNInfo disclosure discovery failed for {canonical}: {exc}"
        ) from exc


def default_reporting_years(today: date | None = None) -> tuple[int, int]:
    current = today or date.today()
    return current.year - 1, current.year
