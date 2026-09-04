"""Offline evaluators for tool use, PIT constraints and report grounding."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from .dataset import EvalCase
from .trajectory import TrajectoryRecord, TrajectoryScores, score_trajectory

_DATE_RE = re.compile(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?")


@dataclass
class SingleStepResult:
    total_tool_calls: int = 0
    correct_tool_selection: int = 0
    invalid_tool_calls: int = 0
    checked_date_params: int = 0
    valid_date_params: int = 0
    accuracy: float = 1.0
    parameter_accuracy: float = 1.0


@dataclass
class ReportQualityResult:
    report_nonempty: bool = False
    section_completeness: float = 0.0
    no_future_explicit_dates: bool = True
    numeric_grounding_rate: float = 1.0
    evidence_aligned: bool = True
    pass_rate: float = 0.0


@dataclass
class EvaluationReport:
    case_id: str
    single_step: SingleStepResult
    trajectory: TrajectoryScores
    report_quality: ReportQualityResult
    overall_pass: bool
    notes: list[str] = field(default_factory=list)

    @property
    def final_conclusion(self) -> ReportQualityResult:
        return self.report_quality

    def to_dict(self) -> dict:
        return asdict(self)


def _forbidden_tools(case: EvalCase) -> set[str]:
    tools: set[str] = set()
    for step in case.expected_trajectory:
        tools.update(step.forbidden_tools)
    for constraint in case.forbidden_constraints:
        if constraint.startswith("FORBID_TOOL:"):
            tools.add(constraint.split(":", 1)[1])
    return tools


class SingleStepEvaluator:
    """Check tool allow/deny rules and explicit date parameters."""

    def __init__(self, case: EvalCase):
        self.case = case

    def evaluate(self, record: TrajectoryRecord) -> SingleStepResult:
        cutoff = date.fromisoformat(self.case.trade_date[:10])
        forbidden = _forbidden_tools(self.case)
        result = SingleStepResult()

        for node in record.nodes:
            for call in node.tool_calls:
                result.total_tool_calls += 1
                if call.tool_name in forbidden:
                    result.invalid_tool_calls += 1
                else:
                    result.correct_tool_selection += 1

                for key in ("as_of_date", "curr_date", "trade_date", "end_date"):
                    raw = call.arguments.get(key)
                    if not raw:
                        continue
                    try:
                        parsed = date.fromisoformat(str(raw)[:10])
                    except ValueError:
                        continue
                    result.checked_date_params += 1
                    if parsed <= cutoff:
                        result.valid_date_params += 1

        if result.total_tool_calls:
            result.accuracy = (
                result.correct_tool_selection / result.total_tool_calls
            )
        if result.checked_date_params:
            result.parameter_accuracy = (
                result.valid_date_params / result.checked_date_params
            )
        return result


class TrajectoryEvaluator:
    def __init__(self, case: EvalCase):
        self.case = case

    def evaluate(self, record: TrajectoryRecord) -> TrajectoryScores:
        return score_trajectory(
            record,
            self.case.expected_trajectory,
            trade_date=self.case.trade_date,
        )


def _explicit_dates(text: str) -> list[date]:
    values: list[date] = []
    for year, month, day in _DATE_RE.findall(text):
        try:
            values.append(date(int(year), int(month), int(day)))
        except ValueError:
            continue
    return values


def _numbers(text: str) -> list[str]:
    return [match.group(0).replace("+", "") for match in _NUMBER_RE.finditer(text)]


class ReportQualityEvaluator:
    """Evaluate report completeness, PIT dates and numeric evidence alignment."""

    def __init__(self, case: EvalCase):
        self.case = case

    def evaluate(
        self,
        state: dict[str, Any],
        reports: dict[str, str],
    ) -> ReportQualityResult:
        expected_sections = self.case.expected_analyst_reports or [
            "market_report",
            "news_report",
            "fundamentals_report",
        ]
        source_sections: list[str] = []
        for key in expected_sections:
            value = reports.get(key) or state.get(key) or ""
            if isinstance(value, str) and value.strip():
                source_sections.append(value.strip())

        final_text = str(
            reports.get("final_report")
            or reports.get("final_trade_decision")
            or state.get("final_report")
            or state.get("final_trade_decision")
            or ""
        ).strip()

        result = ReportQualityResult()
        result.report_nonempty = bool(final_text or source_sections)
        result.section_completeness = (
            len(source_sections) / len(expected_sections)
            if expected_sections
            else 1.0
        )

        cutoff = date.fromisoformat(self.case.trade_date[:10])
        all_text = "\n".join(source_sections + ([final_text] if final_text else []))
        result.no_future_explicit_dates = all(
            item <= cutoff for item in _explicit_dates(all_text)
        )

        if final_text and source_sections:
            source_numbers = set(_numbers("\n".join(source_sections)))
            final_numbers = _numbers(final_text)
            if final_numbers:
                grounded = sum(item in source_numbers for item in final_numbers)
                result.numeric_grounding_rate = grounded / len(final_numbers)

        result.evidence_aligned = result.numeric_grounding_rate >= 0.8
        checks = [
            result.report_nonempty,
            result.section_completeness >= 0.99,
            result.no_future_explicit_dates,
            result.evidence_aligned,
        ]
        result.pass_rate = sum(bool(item) for item in checks) / len(checks)
        return result


FinalConclusionEvaluator = ReportQualityEvaluator
