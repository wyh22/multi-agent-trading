"""A 股市场环境、行业发现、可选量化排序与协调智能体。"""

from .market import analyze_market_regime
from .pipeline import (
    run_discovery,
    run_research_pool,
    run_stock_discovery_legacy,
    write_discovery_report,
    write_research_pool_report,
)
from .representatives import select_representative_stocks
from .sector_ranker import (
    LightGBMSectorRanker,
    blend_sector_scores,
)
from .sectors import analyze_sectors

__all__ = [
    "DiscoveryCoordinatorAgent",
    "DiscoveryCoordinatorResult",
    "LightGBMSectorRanker",
    "analyze_market_regime",
    "analyze_sectors",
    "blend_sector_scores",
    "run_discovery",
    "run_research_pool",
    "run_stock_discovery_legacy",
    "select_representative_stocks",
    "write_discovery_report",
    "write_research_pool_report",
]


def __getattr__(name: str):
    if name in {"DiscoveryCoordinatorAgent", "DiscoveryCoordinatorResult"}:
        from .coordinator_agent import (
            DiscoveryCoordinatorAgent,
            DiscoveryCoordinatorResult,
        )
        return {
            "DiscoveryCoordinatorAgent": DiscoveryCoordinatorAgent,
            "DiscoveryCoordinatorResult": DiscoveryCoordinatorResult,
        }[name]
    raise AttributeError(name)
