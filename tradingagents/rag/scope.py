from __future__ import annotations

from tradingagents.dataflows.symbol_utils import normalize_a_share_symbol

GLOBAL_KNOWLEDGE_SCOPE = "__GLOBAL__"


def normalize_knowledge_scope(value: str) -> str:
    """Normalize a company ticker or the shared market/policy knowledge scope."""

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("knowledge scope/ticker 不能为空")
    if raw.upper() in {
        "GLOBAL",
        "__GLOBAL__",
        "MARKET",
        "POLICY",
        "SHARED",
    }:
        return GLOBAL_KNOWLEDGE_SCOPE
    return normalize_a_share_symbol(raw)
