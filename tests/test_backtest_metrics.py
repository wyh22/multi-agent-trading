from tradingagents.backtest.metrics import (
    compute_backtest_metrics,
    direction_accuracy,
    excess_return,
    max_drawdown,
)


def test_direction_accuracy_ignores_neutral_rating():
    decisions = ["Buy", "Sell", "Hold"]
    returns = [0.10, -0.05, 0.02]
    assert direction_accuracy(decisions, returns) == 1.0


def test_excess_return_and_drawdown_are_deterministic():
    assert excess_return([0.10, -0.02], [0.04, 0.01]) == [0.06, -0.03]
    assert round(max_drawdown([1.0, 1.1, 0.99, 1.2]), 6) == 0.1


def test_compute_backtest_metrics_reports_counts_and_alpha():
    result = compute_backtest_metrics(
        ["Buy", "Sell", "Hold"],
        [0.10, -0.05, 0.02],
        [0.04, -0.01, 0.01],
    )
    assert result.total_decisions == 3
    assert result.directional_decisions == 2
    assert result.direction_accuracy == 1.0
    assert round(result.mean_excess_return_pct, 6) == round((0.06 - 0.04 + 0.01) / 3 * 100, 6)
