"""Backtest utilities.

Backtest evaluates post-decision market outcomes, while `tradingagents.evaluation`
focuses on Agent engineering quality such as tool choice, PIT constraints and report grounding.
"""

from .metrics import (
    BacktestMetrics,
    compute_backtest_metrics,
    direction_accuracy,
    excess_return,
    max_drawdown,
    mean_return,
    performance_by_rating,
    sharpe_ratio,
)
from .runner import BacktestConfig, BacktestResult, BacktestRunner, run_backtest

__all__ = [
    "BacktestMetrics",
    "BacktestConfig",
    "BacktestResult",
    "BacktestRunner",
    "compute_backtest_metrics",
    "direction_accuracy",
    "excess_return",
    "max_drawdown",
    "mean_return",
    "performance_by_rating",
    "run_backtest",
    "sharpe_ratio",
]
