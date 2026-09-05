from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class MarketRegimeResult:
    as_of_date: str
    regime: str
    score: float
    indices: pd.DataFrame
    summary: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class SectorRankingResult:
    as_of_date: str
    current_data_date: str
    anchor_20d_date: str
    anchor_60d_date: str
    sectors: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


@dataclass
class SectorDiscoveryResult:
    """Primary discovery output: rank sectors, not individual stocks.

    `sectors` is the Top-K research shortlist. `sector_universe` keeps the
    complete cross-sectional ranking for audit, comparison and optional model
    analysis.
    """

    as_of_date: str
    market: MarketRegimeResult
    sectors: SectorRankingResult
    sector_universe: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StockScreenResult:
    """Legacy stock-screen result retained for backwards-compatible experiments."""

    as_of_date: str
    regime: str
    candidates: pd.DataFrame
    universe_size: int
    scored_size: int
    warnings: list[str] = field(default_factory=list)
    sector_quotas: dict[str, int] = field(default_factory=dict)
    quality_pool_size: int = 0
    quality_scored_size: int = 0


@dataclass
class DiscoveryResult:
    """Legacy sector-first-then-stock discovery result.

    New application code should prefer :class:`SectorDiscoveryResult`.
    """

    as_of_date: str
    market: MarketRegimeResult
    sectors: SectorRankingResult
    stocks: StockScreenResult
    metadata: dict[str, Any] = field(default_factory=dict)
