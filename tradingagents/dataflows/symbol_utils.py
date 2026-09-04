"""A-share symbol normalization helpers.

The public project is intentionally scoped to mainland A shares. Vendor-specific
symbol rewrites for unrelated markets are kept out of the core data path so all
market-data adapters receive one deterministic canonical form: `000001.SZ`,
`600000.SH`, or `830799.BJ`.
"""
from __future__ import annotations

import re

from .errors import NoMarketDataError as NoMarketDataError

_EXCHANGE_ALIASES = {
    "SH": "SH", "SS": "SH", "SSE": "SH", "SHSE": "SH", "XSHG": "SH",
    "SZ": "SZ", "SZSE": "SZ", "XSHE": "SZ",
    "BJ": "BJ", "BSE": "BJ",
}


def normalize_a_share_symbol(raw: str) -> str:
    """Normalize common mainland stock-code forms to `CODE.EXCHANGE`."""
    if not isinstance(raw, str) or not raw.strip():
        return raw

    value = raw.strip().upper().replace(" ", "")

    prefix = re.fullmatch(
        r"(SH|SS|SSE|SHSE|XSHG|SZ|SZSE|XSHE|BJ|BSE)(\d{6})", value
    )
    if prefix:
        exchange, code = prefix.groups()
        return f"{code}.{_EXCHANGE_ALIASES[exchange]}"

    suffix = re.fullmatch(
        r"(\d{6})\.(SH|SS|SSE|SHSE|XSHG|SZ|SZSE|XSHE|BJ|BSE)", value
    )
    if suffix:
        code, exchange = suffix.groups()
        return f"{code}.{_EXCHANGE_ALIASES[exchange]}"

    if re.fullmatch(r"\d{6}", value):
        if value[0] in ("6", "5", "9"):
            exchange = "SH"
        elif value[0] in ("0", "3", "1"):
            exchange = "SZ"
        elif value[0] in ("4", "8"):
            exchange = "BJ"
        else:
            exchange = "SH"
        return f"{value}.{exchange}"

    return value


def normalize_symbol(raw: str) -> str:
    """Backward-compatible alias for the A-share canonicalizer."""
    return normalize_a_share_symbol(raw)
