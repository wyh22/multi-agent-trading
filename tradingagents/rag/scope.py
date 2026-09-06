from __future__ import annotations

from dataclasses import dataclass

from tradingagents.dataflows.symbol_utils import normalize_a_share_symbol

SCOPE_TYPES = {
    "company",
    "industry",
    "market",
    "macro",
    "regulation",
}
SHARED_TICKER = "__SHARED__"


@dataclass(frozen=True)
class KnowledgeScope:
    scope_type: str
    scope_key: str
    ticker: str
    industry: str = ""

    @property
    def scope_id(self) -> str:
        return f"{self.scope_type}:{self.scope_key}"


def _clean_scope_type(value: str | None) -> str:
    scope_type = str(value or "company").strip().lower()
    aliases = {
        "global": "market",
        "shared": "market",
        "policy": "regulation",
        "regulatory": "regulation",
        "sector": "industry",
    }
    scope_type = aliases.get(scope_type, scope_type)
    if scope_type not in SCOPE_TYPES:
        raise ValueError(
            f"unsupported knowledge scope_type={scope_type!r}; "
            f"expected one of {sorted(SCOPE_TYPES)}"
        )
    return scope_type


def normalize_scope(
    *,
    scope_type: str = "company",
    scope_key: str | None = None,
    ticker: str | None = None,
    industry: str | None = None,
) -> KnowledgeScope:
    """Normalize hierarchical knowledge scope without stock-specific rules."""

    kind = _clean_scope_type(scope_type)
    raw_key = str(scope_key or "").strip()
    raw_ticker = str(ticker or "").strip()
    raw_industry = str(industry or "").strip()

    if kind == "company":
        canonical = normalize_a_share_symbol(raw_ticker or raw_key)
        return KnowledgeScope(
            scope_type="company",
            scope_key=canonical,
            ticker=canonical,
            industry=raw_industry,
        )

    if kind == "industry":
        key = raw_key or raw_industry
        if not key:
            raise ValueError("industry scope requires scope_key or industry")
        return KnowledgeScope(
            scope_type="industry",
            scope_key=key,
            ticker=SHARED_TICKER,
            industry=raw_industry or key,
        )

    default_keys = {
        "market": "CN_A",
        "macro": "CN",
        "regulation": "CN",
    }
    key = raw_key or default_keys[kind]
    return KnowledgeScope(
        scope_type=kind,
        scope_key=key,
        ticker=SHARED_TICKER,
        industry=raw_industry,
    )


def company_scope_id(ticker: str) -> str:
    return normalize_scope(ticker=ticker).scope_id


def default_shared_scope_ids() -> list[str]:
    return [
        normalize_scope(scope_type="market").scope_id,
        normalize_scope(scope_type="macro").scope_id,
        normalize_scope(scope_type="regulation").scope_id,
    ]


def industry_scope_id(industry: str) -> str:
    return normalize_scope(
        scope_type="industry",
        scope_key=industry,
    ).scope_id
