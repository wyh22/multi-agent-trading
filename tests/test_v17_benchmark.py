from pathlib import Path

import pandas as pd

from tradingagents.discovery.style_ic import build_style_ic_history
from tradingagents.evaluation.architecture import (
    ArchitectureBenchmarkCase,
    ArchitectureSystemRun,
    evaluate_architecture_run,
    load_architecture_benchmark,
    pit_parameter_score,
)
from tradingagents.evaluation.systems import SYSTEM_BUILDERS
from tradingagents.evaluation.telemetry import EvaluationTelemetryHandler, estimate_cost
from tradingagents.orchestration.schemas import CompletionAssessment


ROOT = Path(__file__).resolve().parents[1]


def test_style_ic_history_records_when_forward_label_becomes_available():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    rows = []
    for sector_idx in range(4):
        for date_idx, value in enumerate(dates):
            rows.append(
                {
                    "date": value.date().isoformat(),
                    "sector_code": f"S{sector_idx}",
                    "close": 100 + sector_idx * 5 + date_idx * (sector_idx + 1),
                    "momentum_score": 10 + sector_idx * 10,
                    "valuation_score": 40 - sector_idx * 5,
                    "dividend_score": 20 + sector_idx,
                    "liquidity_score": 30 + sector_idx * 2,
                }
            )
    panel = pd.DataFrame(rows)
    history = build_style_ic_history(
        panel,
        forward_periods=2,
        min_sector_count=3,
    )
    assert not history.empty
    assert "available_date" in history.columns
    assert all(
        pd.to_datetime(history["available_date"])
        > pd.to_datetime(history["date"])
    )


def test_pit_parameter_score_catches_future_tool_date():
    trace = [
        {
            "tool_name": "get_news",
            "arguments": {"end_date": "2026-09-04"},
        },
        {
            "tool_name": "get_stock_data",
            "arguments": {"as_of_date": "2026-09-10"},
        },
    ]
    violations, checked = pit_parameter_score(trace, "2026-09-05")
    assert checked == 2
    assert violations == 1


def test_architecture_eval_combines_routing_grounding_completion_and_pit():
    case = ArchitectureBenchmarkCase(
        case_id="case",
        user_query="分析现金流",
        ticker="600519.SH",
        as_of_date="2026-09-05",
        expected_actions=["delegate_agent"],
        expected_targets=["fundamentals"],
        required_dimensions=["现金流"],
        required_entities=["600519.SH"],
        critical_requirements=["现金流"],
    )
    run = ArchitectureSystemRun(
        case_id="case",
        system_name="dynamic-supervisor",
        answer=(
            "## Evidence Claims\n"
            "- [FACT] 经营现金流为正。"
        ),
        route_action="delegate_agent",
        route_target="fundamentals",
        telemetry={
            "llm_calls": 2,
            "tool_calls": 1,
            "tokens_in": 100,
            "tokens_out": 20,
        },
        evidence=[
            "## Evidence Claims\n- [FACT] 经营现金流为正。"
        ],
        tool_trace=[
            {
                "tool_name": "get_cashflow",
                "arguments": {"curr_date": "2026-09-05"},
            }
        ],
    )

    def complete(_contract, _observations):
        return CompletionAssessment(
            complete=True,
            completion_ratio=1.0,
            completed_items=["600519.SH::现金流"],
        )

    result = evaluate_architecture_run(
        case,
        run,
        completion_assessor=complete,
    )
    assert result.route_pass is True
    assert result.claim_grounding_rate == 1.0
    assert result.complete is True
    assert result.pit_parameter_violations == 0
    assert result.llm_calls == 2


def test_cost_estimate_uses_external_pricing_not_hardcoded_prices():
    telemetry = {
        "by_model": {
            "model-a": {
                "tokens_in": 1_000_000,
                "tokens_out": 500_000,
            }
        }
    }
    cost = estimate_cost(
        telemetry,
        {
            "model-a": {
                "input_per_million": 2.0,
                "output_per_million": 6.0,
            }
        },
    )
    assert cost == 5.0
    assert estimate_cost(telemetry, None) is None


def test_benchmark_dataset_and_systems_are_wired():
    cases = load_architecture_benchmark(
        ROOT / "evaluation" / "datasets" / "architecture_benchmark_v1.jsonl"
    )
    assert len(cases) >= 6
    assert {
        "single-agent-all-tools",
        "fixed-deep-research",
        "dynamic-supervisor",
    }.issubset(SYSTEM_BUILDERS)

    script = (
        ROOT / "scripts" / "run_agent_benchmark.py"
    ).read_text(encoding="utf-8")
    for artifact in (
        "raw_runs.jsonl",
        "per_case.csv",
        "summary.csv",
        "BENCHMARK_REPORT.md",
    ):
        assert artifact in script


def test_benchmark_capture_and_graph_callbacks_are_opt_in_and_traceable():
    conversation = (
        ROOT / "tradingagents" / "conversation" / "agent.py"
    ).read_text(encoding="utf-8")
    assert "capture_evidence: bool = False" in conversation
    assert '"evaluation_evidence"' in conversation
    assert '"evaluation_tool_trace"' in conversation

    graph = (
        ROOT / "tradingagents" / "graph" / "trading_graph.py"
    ).read_text(encoding="utf-8")
    assert "get_graph_args(callbacks=self.callbacks or None)" in graph


def test_telemetry_deduplicates_same_callback_run_id():
    handler = EvaluationTelemetryHandler()
    handler.on_chat_model_start(
        {"name": "model-a"},
        [[]],
        run_id="same-llm-run",
    )
    handler.on_chat_model_start(
        {"name": "model-a"},
        [[]],
        run_id="same-llm-run",
    )
    handler.on_tool_start(
        {"name": "tool-a"},
        "{}",
        run_id="same-tool-run",
    )
    handler.on_tool_start(
        {"name": "tool-a"},
        "{}",
        run_id="same-tool-run",
    )
    snapshot = handler.snapshot()
    assert snapshot["llm_calls"] == 1
    assert snapshot["tool_calls"] == 1
