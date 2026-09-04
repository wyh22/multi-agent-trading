"""Evaluation runner for deterministic Agent-quality reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .dataset import AgentEvalDataset, EvalCase, load_eval_dataset
from .evaluators import (
    EvaluationReport,
    ReportQualityEvaluator,
    SingleStepEvaluator,
    TrajectoryEvaluator,
)
from .trajectory import TrajectoryRecord


def run_single_eval_case(
    case: EvalCase,
    agent_run_fn: Callable[
        [EvalCase],
        tuple[dict[str, Any], dict[str, str], TrajectoryRecord],
    ],
) -> EvaluationReport:
    state, reports, record = agent_run_fn(case)

    single = SingleStepEvaluator(case).evaluate(record)
    trajectory = TrajectoryEvaluator(case).evaluate(record)
    quality = ReportQualityEvaluator(case).evaluate(state, reports)

    overall = (
        single.accuracy >= 0.8
        and single.parameter_accuracy >= 0.9
        and trajectory.trajectory_pass_rate >= 0.9
        and quality.pass_rate >= 0.8
    )
    notes: list[str] = []
    if single.accuracy < 0.8:
        notes.append(f"tool_selection={single.accuracy:.2f}")
    if single.parameter_accuracy < 0.9:
        notes.append(f"parameter_accuracy={single.parameter_accuracy:.2f}")
    if trajectory.trajectory_pass_rate < 0.9:
        notes.append(f"trajectory_pass={trajectory.trajectory_pass_rate:.2f}")
    if quality.pass_rate < 0.8:
        notes.append(f"report_quality={quality.pass_rate:.2f}")

    return EvaluationReport(
        case_id=case.case_id,
        single_step=single,
        trajectory=trajectory,
        report_quality=quality,
        overall_pass=overall,
        notes=notes,
    )


def run_evaluation(
    dataset: str | Path | AgentEvalDataset,
    agent_run_fn: Callable[
        [EvalCase],
        tuple[dict[str, Any], dict[str, str], TrajectoryRecord],
    ],
    output_path: str | Path | None = None,
) -> list[EvaluationReport]:
    if isinstance(dataset, (str, Path)):
        dataset = load_eval_dataset(dataset)

    results = [
        run_single_eval_case(case, agent_run_fn)
        for case in dataset.cases
    ]
    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "\n".join(
                json.dumps(item.to_dict(), ensure_ascii=False)
                for item in results
            ),
            encoding="utf-8",
        )
    return results
