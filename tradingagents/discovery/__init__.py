"""A 股市场环境、行业轮动、候选发现与协调智能体。"""

from .market import analyze_market_regime
from .pipeline import run_discovery, write_discovery_report
from .sectors import analyze_sectors

__all__ = [
    "DiscoveryCoordinatorAgent",
    "DiscoveryCoordinatorResult",
    "analyze_market_regime",
    "analyze_sectors",
    "run_discovery",
    "write_discovery_report",
]


def __getattr__(name: str):
    if name in {"DiscoveryCoordinatorAgent", "DiscoveryCoordinatorResult"}:
        from .coordinator_agent import DiscoveryCoordinatorAgent, DiscoveryCoordinatorResult
        return {"DiscoveryCoordinatorAgent": DiscoveryCoordinatorAgent, "DiscoveryCoordinatorResult": DiscoveryCoordinatorResult}[name]
    raise AttributeError(name)
