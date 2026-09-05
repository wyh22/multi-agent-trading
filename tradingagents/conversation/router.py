from __future__ import annotations

import functools
import re
from dataclasses import dataclass

from tradingagents.dataflows.symbol_utils import normalize_a_share_symbol


_STOCK_CODE = re.compile(r"(?<!\d)(\d{6})(?:\.(SH|SZ|SS|SSE|SZSE|BJ|BSE))?(?!\d)", re.IGNORECASE)


@dataclass(frozen=True)
class ConversationRoute:
    intent: str
    ticker: str | None


def _normalize_match(match: re.Match[str]) -> str | None:
    code = match.group(1)
    suffix = (match.group(2) or "").upper()
    raw = f"{code}.{suffix}" if suffix else code
    try:
        return normalize_a_share_symbol(raw)
    except Exception:  # noqa: BLE001
        return None


@functools.lru_cache(maxsize=1)
def _a_share_name_map() -> list[tuple[str, str]]:
    """Best-effort company-name resolver used only when AkShare is available."""
    try:
        import akshare as ak

        df = ak.stock_info_a_code_name()
        if df is None or df.empty:
            return []
        code_col = next((c for c in df.columns if str(c).lower() in {"code", "证券代码", "股票代码"}), df.columns[0])
        name_col = next((c for c in df.columns if str(c).lower() in {"name", "证券简称", "股票简称"}), df.columns[1])
        pairs: list[tuple[str, str]] = []
        for code, name in zip(df[code_col].astype(str), df[name_col].astype(str)):
            name = name.strip()
            code = code.strip().zfill(6)
            if not name:
                continue
            try:
                pairs.append((name, normalize_a_share_symbol(code)))
            except Exception:  # noqa: BLE001
                continue
        return sorted(pairs, key=lambda item: len(item[0]), reverse=True)
    except Exception:  # noqa: BLE001
        return []


def extract_ticker(text: str) -> str | None:
    value = text or ""
    match = _STOCK_CODE.search(value)
    if match:
        return _normalize_match(match)

    name_resolution_words = ("分析", "研究", "股票", "公司", "公告", "财报", "股价", "估值", "比较", "怎么看", "怎么样", "如何")
    if len(value) <= 80 and any(word in value for word in name_resolution_words):
        for name, ticker in _a_share_name_map():
            if name in value:
                return ticker
    return None


def route_message(text: str, *, current_ticker: str | None = None, force_mode: str = "auto") -> ConversationRoute:
    normalized = (text or "").strip().lower()
    explicit_ticker = extract_ticker(text)
    ticker = explicit_ticker or current_ticker

    if force_mode != "auto":
        return ConversationRoute(force_mode, ticker)

    discovery_words = ("候选", "筛选", "选股", "股票池", "top10", "top 10", "找几只", "找出", "行业排名", "板块排名", "行业发现", "值得关注的行业", "值得关注的板块", "代表股", "代表性股票", "研究池")
    research_words = ("深度分析", "深度研究", "完整分析", "完整研究", "投研报告", "研报", "全面分析", "重新分析", "重新研究")
    knowledge_words = ("公告", "年报", "季报", "财报原文", "历史材料", "问询函", "知识库")

    if any(word in normalized for word in discovery_words):
        return ConversationRoute("discovery", ticker)
    if any(word in normalized for word in research_words):
        return ConversationRoute("research", ticker)
    if explicit_ticker and any(word in normalized for word in ("分析", "研究", "怎么看", "如何看")):
        return ConversationRoute("research", ticker)
    if any(word in normalized for word in knowledge_words):
        return ConversationRoute("tool_chat", ticker)
    return ConversationRoute("tool_chat", ticker)
