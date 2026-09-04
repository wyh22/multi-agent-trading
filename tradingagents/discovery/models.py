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
class StockScreenResult:
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
    as_of_date: str
    market: MarketRegimeResult
    sectors: SectorRankingResult
    stocks: StockScreenResult
    metadata: dict[str, Any] = field(default_factory=dict)
