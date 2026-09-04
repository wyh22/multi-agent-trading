from tradingagents.evaluation.dataset import EvalCase, ExpectedTrajectoryStep
from tradingagents.evaluation.evaluators import ReportQualityEvaluator, SingleStepEvaluator
from tradingagents.evaluation.trajectory import (
    ToolCallRecord,
    TrajectoryNode,
    TrajectoryRecord,
    score_trajectory,
)


def _case():
    return EvalCase(
        case_id="demo",
        ticker="600519",
        trade_date="2024-04-10",
        expected_analyst_reports=["market_report"],
        expected_trajectory=[
            ExpectedTrajectoryStep(
                analyst="Market Analyst",
                expected_tools=["get_stock_data"],
                forbidden_tools=["future_tool"],
            )
        ],
    )


def test_trajectory_passes_expected_tool_and_pit_date():
    case = _case()
    record = TrajectoryRecord(
        nodes=[
            TrajectoryNode(
                analyst="Market Analyst",
                tool_calls=[
                    ToolCallRecord(
                        "get_stock_data",
                        {"end_date": "2024-04-10"},
                    )
                ],
            )
        ]
    )
    scores = score_trajectory(
        record,
        case.expected_trajectory,
        trade_date=case.trade_date,
    )
    assert scores.trajectory_pass_rate == 1.0
    assert scores.future_date_violations == []


def test_single_step_detects_future_date_parameter():
    case = _case()
    record = TrajectoryRecord(
        nodes=[
            TrajectoryNode(
                analyst="Market Analyst",
                tool_calls=[
                    ToolCallRecord(
                        "get_stock_data",
                        {"end_date": "2024-04-11"},
                    )
                ],
            )
        ]
    )
    result = SingleStepEvaluator(case).evaluate(record)
    assert result.parameter_accuracy == 0.0


def test_report_quality_checks_numeric_grounding_and_future_dates():
    case = _case()
    evaluator = ReportQualityEvaluator(case)
    result = evaluator.evaluate(
        {
            "market_report": "截至 2024-04-10，收盘价为 100.0。",
            "final_trade_decision": "关键证据为收盘价 100.0。",
        },
        {},
    )
    assert result.report_nonempty
    assert result.no_future_explicit_dates
    assert result.numeric_grounding_rate == 1.0
    assert result.pass_rate == 1.0
