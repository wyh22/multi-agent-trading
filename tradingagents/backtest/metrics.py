"""Deterministic metrics for batched research backtests."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass
class BacktestMetrics:
    total_decisions: int
    directional_decisions: int
    direction_accuracy: float
    mean_raw_return_pct: float
    mean_excess_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    per_rating: dict[str, dict[str, float]]
    raw_returns: list[float]
    excess_returns: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


_LONG = {"BUY", "OVERWEIGHT", "买入", "增持"}
_SHORT = {"SELL", "UNDERWEIGHT", "卖出", "减持"}


def _normalized_rating(value: str) -> str:
    return str(value).strip().upper()


def direction_accuracy(decisions: Sequence[str], returns: Sequence[float]) -> float:
    correct = 0
    directional = 0
    for decision, ret in zip(decisions, returns, strict=False):
        rating = _normalized_rating(decision)
        if rating in _LONG:
            directional += 1
            correct += int(ret > 0)
        elif rating in _SHORT:
            directional += 1
            correct += int(ret < 0)
    return correct / directional if directional else 0.0


def mean_return(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def excess_return(raw: Sequence[float], benchmark: Sequence[float]) -> list[float]:
    return [float(r - b) for r, b in zip(raw, benchmark, strict=False)]


def max_drawdown(equity_curve: Sequence[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = float(equity_curve[0])
    drawdown = 0.0
    for value in equity_curve:
        value = float(value)
        peak = max(peak, value)
        if peak > 0:
            drawdown = max(drawdown, (peak - value) / peak)
    return drawdown


def sharpe_ratio(
    returns: Sequence[float],
    *,
    periods_per_year: int = 252,
    risk_free: float = 0.02,
) -> float:
    if len(returns) < 2:
        return 0.0
    sigma = statistics.stdev(returns)
    if sigma == 0:
        return 0.0
    mean_period = statistics.mean(returns)
    risk_free_period = risk_free / periods_per_year
    return float((mean_period - risk_free_period) / sigma * math.sqrt(periods_per_year))


def performance_by_rating(
    decisions: Sequence[str],
    returns: Sequence[float],
    benchmark_returns: Sequence[float],
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[tuple[float, float]]] = {}
    for decision, raw, bench in zip(decisions, returns, benchmark_returns, strict=False):
        buckets.setdefault(str(decision), []).append((float(raw), float(bench)))

    result: dict[str, dict[str, float]] = {}
    for rating, pairs in buckets.items():
        raw_values = [raw for raw, _ in pairs]
        alpha_values = [raw - bench for raw, bench in pairs]
        result[rating] = {
            "count": float(len(pairs)),
            "mean_raw_pct": mean_return(raw_values) * 100,
            "mean_alpha_pct": mean_return(alpha_values) * 100,
        }
    return result


def compute_backtest_metrics(
    decisions: Sequence[str],
    raw_returns: Sequence[float],
    benchmark_returns: Sequence[float],
    *,
    periods_per_year: int = 252,
    risk_free: float = 0.02,
) -> BacktestMetrics:
    n = min(len(decisions), len(raw_returns), len(benchmark_returns))
    decisions = list(decisions[:n])
    raw_returns = [float(x) for x in raw_returns[:n]]
    benchmark_returns = [float(x) for x in benchmark_returns[:n]]

    alpha = excess_return(raw_returns, benchmark_returns)
    equity = [1.0]
    for ret in raw_returns:
        equity.append(equity[-1] * (1.0 + ret))

    directional = sum(
        1 for decision in decisions if _normalized_rating(decision) in (_LONG | _SHORT)
    )
    return BacktestMetrics(
        total_decisions=n,
        directional_decisions=directional,
        direction_accuracy=direction_accuracy(decisions, raw_returns),
        mean_raw_return_pct=mean_return(raw_returns) * 100,
        mean_excess_return_pct=mean_return(alpha) * 100,
        max_drawdown_pct=max_drawdown(equity) * 100,
        sharpe_ratio=sharpe_ratio(
            raw_returns,
            periods_per_year=periods_per_year,
            risk_free=risk_free,
        ),
        per_rating=performance_by_rating(decisions, raw_returns, benchmark_returns),
        raw_returns=raw_returns,
        excess_returns=alpha,
    )
