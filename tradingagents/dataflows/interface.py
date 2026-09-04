import logging

from .akshare_financials import (
    get_free_balance_sheet,
    get_free_cashflow,
    get_free_income_statement,
)
from .akshare_macro import get_akshare_macro_data
from .akshare_news import get_akshare_global_news, get_akshare_insider_transactions
from .alpha_vantage import (
    get_balance_sheet as get_alpha_vantage_balance_sheet,
    get_cashflow as get_alpha_vantage_cashflow,
    get_fundamentals as get_alpha_vantage_fundamentals,
    get_global_news as get_alpha_vantage_global_news,
    get_income_statement as get_alpha_vantage_income_statement,
    get_indicator as get_alpha_vantage_indicator,
    get_insider_transactions as get_alpha_vantage_insider_transactions,
    get_news as get_alpha_vantage_news,
    get_stock as get_alpha_vantage_stock,
)
from .baostock import get_baostock_fundamentals, get_baostock_stock_data
from .cninfo import get_cninfo_news
from .config import get_config
from .errors import NoMarketDataError, VendorNotConfiguredError, VendorRateLimitError
from .fred import get_macro_data as get_fred_macro_data
from .stockstats_utils import get_stock_stats_indicators_window

logger = logging.getLogger(__name__)


def _alpha_insider(ticker: str, curr_date: str | None = None):
    return get_alpha_vantage_insider_transactions(ticker)


TOOLS_CATEGORIES = {
    "core_stock_apis": {"description": "OHLCV stock price data", "tools": ["get_stock_data"]},
    "technical_indicators": {"description": "Technical analysis indicators", "tools": ["get_indicators"]},
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": ["get_fundamentals", "get_balance_sheet", "get_cashflow", "get_income_statement"],
    },
    "news_data": {
        "description": "News and management-holding data",
        "tools": ["get_news", "get_global_news", "get_insider_transactions"],
    },
    "macro_data": {"description": "Macroeconomic indicators", "tools": ["get_macro_indicators"]},
}

VENDOR_LIST = ["baostock", "akshare", "cninfo", "alpha_vantage", "fred"]

VENDOR_METHODS = {
    "get_stock_data": {
        "baostock": get_baostock_stock_data,
        "alpha_vantage": get_alpha_vantage_stock,
    },
    "get_indicators": {
        "baostock": get_stock_stats_indicators_window,
        "alpha_vantage": get_alpha_vantage_indicator,
    },
    "get_fundamentals": {
        "baostock": get_baostock_fundamentals,
        "alpha_vantage": get_alpha_vantage_fundamentals,
    },
    "get_balance_sheet": {
        "akshare": get_free_balance_sheet,
        "alpha_vantage": get_alpha_vantage_balance_sheet,
    },
    "get_cashflow": {
        "akshare": get_free_cashflow,
        "alpha_vantage": get_alpha_vantage_cashflow,
    },
    "get_income_statement": {
        "akshare": get_free_income_statement,
        "alpha_vantage": get_alpha_vantage_income_statement,
    },
    "get_news": {
        "cninfo": get_cninfo_news,
        "alpha_vantage": get_alpha_vantage_news,
    },
    "get_global_news": {
        "akshare": get_akshare_global_news,
        "alpha_vantage": get_alpha_vantage_global_news,
    },
    "get_insider_transactions": {
        "akshare": get_akshare_insider_transactions,
        "alpha_vantage": _alpha_insider,
    },
    "get_macro_indicators": {
        "akshare": get_akshare_macro_data,
        "fred": get_fred_macro_data,
    },
}


def get_category_for_method(method: str) -> str:
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")


def get_vendor(category: str, method: str | None = None) -> str:
    config = get_config()
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]
    return config.get("data_vendors", {}).get(category, "default")


def route_to_vendor(method: str, *args, **kwargs):
    """Route a tool call through the explicitly configured deterministic vendor chain."""
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in str(vendor_config).split(",")]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    all_available_vendors = list(VENDOR_METHODS[method])
    explicit = [v for v in primary_vendors if v and v != "default"]
    if explicit:
        vendor_chain = [v for v in explicit if v in VENDOR_METHODS[method]]
        if not vendor_chain:
            raise ValueError(
                f"Configured vendor(s) {explicit} not available for '{method}'. "
                f"Available: {all_available_vendors}."
            )
    else:
        vendor_chain = all_available_vendors

    last_no_data: NoMarketDataError | None = None
    first_error: Exception | None = None

    for vendor in vendor_chain:
        impl_func = VENDOR_METHODS[method][vendor]
        try:
            return impl_func(*args, **kwargs)
        except VendorRateLimitError:
            logger.warning("Vendor %r rate-limited for %s; trying next vendor.", vendor, method)
        except VendorNotConfiguredError as exc:
            logger.warning("Vendor %r not configured for %s; trying next vendor.", vendor, method)
            first_error = first_error or exc
        except NoMarketDataError as exc:
            last_no_data = exc
        except Exception as exc:
            logger.warning("Vendor %r failed for %s: %s", vendor, method, exc)
            first_error = first_error or exc

    if last_no_data is not None:
        sym = last_no_data.symbol
        canonical = last_no_data.canonical
        resolved = "" if canonical == sym else f" (resolved to '{canonical}')"
        reason = f" ({last_no_data.detail})" if last_no_data.detail else ""
        return (
            f"NO_DATA_AVAILABLE: No usable market data for '{sym}'{resolved} from "
            f"any configured vendor{reason}. Do not estimate or fabricate values."
        )

    if first_error is not None:
        if category == "macro_data" or method in {"get_global_news", "get_insider_transactions"}:
            return (
                f"DATA_UNAVAILABLE: optional {method} could not be retrieved "
                f"({first_error}). Proceed without it; do not fabricate values."
            )
        raise first_error

    raise RuntimeError(f"No available vendor for '{method}'")
