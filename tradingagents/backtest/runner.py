"""Batch outcome-evaluation runner for historical research decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence
import csv
import json

from .metrics import BacktestMetrics, compute_backtest_metrics


@dataclass
class BacktestConfig:
    tickers: Sequence[str]
    trade_dates: Sequence[str]
    holding_days: int = 5
    benchmark_ticker: str = "399001.SZ"
    asset_type: str = "stock"
    output_path: str | None = None


@dataclass
class BacktestResult:
    ticker: str
    trade_date: str
    rating: str
    raw_return_pct: float | None
    bench_return_pct: float | None
    alpha_pct: float | None
    actual_holding_days: int | None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class BacktestRunner:
    """Evaluate already-produced historical research ratings against later returns."""

    def __init__(
        self,
        config: BacktestConfig,
        research_fn: Callable[[str, str, str], str],
        fetch_returns_fn: Callable[
            [str, str, int, str],
            tuple[float | None, float | None, int | None],
        ],
    ):
        self.config = config
        self.research_fn = research_fn
        self.fetch_returns_fn = fetch_returns_fn

    def _evaluate_one(self, ticker: str, trade_date: str) -> BacktestResult:
        try:
            rating = self.research_fn(ticker, trade_date, self.config.asset_type)
            raw, bench, days = self.fetch_returns_fn(
                ticker,
                trade_date,
                self.config.holding_days,
                self.config.benchmark_ticker,
            )
        except Exception as exc:  # noqa: BLE001
            return BacktestResult(
                ticker=ticker,
                trade_date=trade_date,
                rating="ERROR",
                raw_return_pct=None,
                bench_return_pct=None,
                alpha_pct=None,
                actual_holding_days=None,
                error=f"{type(exc).__name__}: {exc}",
            )

        alpha = raw - bench if raw is not None and bench is not None else None
        return BacktestResult(
            ticker=ticker,
            trade_date=trade_date,
            rating=rating,
            raw_return_pct=raw * 100 if raw is not None else None,
            bench_return_pct=bench * 100 if bench is not None else None,
            alpha_pct=alpha * 100 if alpha is not None else None,
            actual_holding_days=days,
        )

    def run(self) -> tuple[list[BacktestResult], BacktestMetrics]:
        results = [
            self._evaluate_one(ticker, trade_date)
            for ticker in self.config.tickers
            for trade_date in self.config.trade_dates
        ]

        decisions: list[str] = []
        raw_returns: list[float] = []
        bench_returns: list[float] = []
        for item in results:
            if item.error or item.raw_return_pct is None or item.bench_return_pct is None:
                continue
            decisions.append(item.rating)
            raw_returns.append(item.raw_return_pct / 100.0)
            bench_returns.append(item.bench_return_pct / 100.0)

        metrics = compute_backtest_metrics(decisions, raw_returns, bench_returns)
        self._write_output(results, metrics)
        return results, metrics

    def _write_output(
        self,
        results: list[BacktestResult],
        metrics: BacktestMetrics,
    ) -> None:
        if not self.config.output_path:
            return

        path = Path(self.config.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".json":
            path.write_text(
                json.dumps(
                    {
                        "config": asdict(self.config),
                        "results": [item.to_dict() for item in results],
                        "metrics": metrics.to_dict(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(BacktestResult.__dataclass_fields__.keys()),
            )
            writer.writeheader()
            for item in results:
                writer.writerow(item.to_dict())


def run_backtest(
    config: BacktestConfig,
    research_fn: Callable[[str, str, str], str],
    fetch_returns_fn: Callable[
        [str, str, int, str],
        tuple[float | None, float | None, int | None],
    ],
) -> tuple[list[BacktestResult], BacktestMetrics]:
    return BacktestRunner(config, research_fn, fetch_returns_fn).run()
