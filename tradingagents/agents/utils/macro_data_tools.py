from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_macro_indicators(
    indicator: Annotated[
        str,
        "China macro indicator: cpi, ppi, gdp, real_gdp, pmi, lpr, "
        "money_supply, m2, unemployment, or unemployment_rate.",
    ],
    curr_date: Annotated[str, "Research cutoff date in yyyy-mm-dd format"],
    look_back_days: Annotated[
        int | None, "Optional trailing window length in days"
    ] = None,
) -> str:
    """Retrieve a PIT-aware Chinese macro series through the configured vendor.

    The default A-share configuration uses AKShare and only returns observations
    available by curr_date. Unsupported indicators degrade to an explicit
    DATA_UNAVAILABLE response rather than fabricated values.
    """
    return route_to_vendor("get_macro_indicators", indicator, curr_date, look_back_days)
