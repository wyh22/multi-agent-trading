from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

from tradingagents.evaluation.grounding import GroundingEvalResult, evaluate_claim_grounding
from tradingagents.evaluation.routing import RoutingEvalCase, RoutingRecord, evaluate_route
from tradingagents.orchestration.schemas import CompletionAssessment, TaskContract


_DATE_RE = re.compile(r"\b20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b")


@dataclass
class ArchitectureBenchmarkCase:
    case_id: str
    user_query: str
    ticker: str | None = None
    as_of_date: str = ""
    expected_actions: list[str] = field(default_factory=list)
    expected_targets: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    forbidden_targets: list[str] = field(default_factory=list)
    required_dimensions: list[str] = field(default_factory=list)
    required_entities: list[str] = field(default_factory=list)
    critical_requirements: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ArchitectureBenchmarkCase":
        return cls(**row)

    def contract(self) -> TaskContract:
        return TaskContract(
            objective=self.user_query,
            required_dimensions=list(self.required_dimensions),
            required_entities=(
                list(self.required_entities)
                or ([self.ticker] if self.ticker else [])
            ),
            critical_requirements=(
                list(self.critical_requirements)
                or list(self.required_dimensions)
            ),
            expected_output="research_answer",
            can_be_partial=True,
        )

    def routing_case(self) -> RoutingEvalCase:
        return RoutingEvalCase(
            case_id=self.case_id,
            user_query=self.user_query,
            ticker=self.ticker,
            as_of_date=self.as_of_date,
            expected_actions=list(self.expected_actions),
            expected_targets=list(self.expected_targets),
            forbidden_actions=list(self.forbidden_actions),
            forbidden_targets=list(self.forbidden_targets),
        )


@dataclass
class ArchitectureSystemRun:
    case_id: str
    system_name: str
    answer: str
    route_action: str
    route_target: str | None
    response_status: str = ""
    audit_status: str = ""
    system_completion_ratio: float | None = None
    latency_ms: float = 0.0
    telemetry: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArchitectureEvalResult:
    case_id: str
    system_name: str
    route_pass: bool
    action_correct: bool
    target_correct: bool
    forbidden_route_violation: bool
    unnecessary_deep_research: bool
    claim_grounding_rate: float
    unsupported_claim_rate: float
    evidence_coverage_rate: float
    completion_ratio: float
    complete: bool
    pit_parameter_violations: int
    checked_pit_parameters: int
    future_date_mentions: int
    latency_ms: float
    llm_calls: int
    tool_calls: int
    tokens_in: int
    tokens_out: int
    estimated_cost: float | None
    response_status: str
    audit_status: str
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_architecture_benchmark(
    path: str | Path,
) -> list[ArchitectureBenchmarkCase]:
    cases: list[ArchitectureBenchmarkCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                cases.append(
                    ArchitectureBenchmarkCase.from_dict(json.loads(line))
                )
    return cases


def _iter_date_arguments(
    tool_trace: list[dict[str, Any]],
):
    keys = {
        "as_of_date",
        "curr_date",
        "trade_date",
        "end_date",
        "start_date",
    }
    for event in tool_trace:
        arguments = event.get("arguments", {}) if isinstance(event, dict) else {}
        if not isinstance(arguments, dict):
            continue
        for key in keys:
            value = arguments.get(key)
            if value:
                yield key, str(value)[:10]


def pit_parameter_score(
    tool_trace: list[dict[str, Any]],
    as_of_date: str,
) -> tuple[int, int]:
    if not as_of_date:
        return 0, 0
    cutoff = date.fromisoformat(as_of_date[:10])
    checked = 0
    violations = 0
    for _key, raw in _iter_date_arguments(tool_trace):
        try:
            parsed = date.fromisoformat(raw)
        except ValueError:
            continue
        checked += 1
        if parsed > cutoff:
            violations += 1
    return violations, checked


def future_date_mentions(text: str, as_of_date: str) -> int:
    if not as_of_date:
        return 0
    cutoff = date.fromisoformat(as_of_date[:10])
    violations = 0
    for raw in _DATE_RE.findall(text or ""):
        normalized = raw.replace("/", "-").replace(".", "-")
        try:
            if date.fromisoformat(normalized) > cutoff:
                violations += 1
        except ValueError:
            continue
    return violations


def evaluate_architecture_run(
    case: ArchitectureBenchmarkCase,
    run: ArchitectureSystemRun,
    *,
    completion_assessor: Callable[
        [TaskContract, list[str]], CompletionAssessment
    ],
    estimated_cost: float | None = None,
) -> ArchitectureEvalResult:
    routing = evaluate_route(
        case.routing_case(),
        RoutingRecord(
            action=run.route_action,
            target=run.route_target,
            steps=1,
            tool_calls=int(run.telemetry.get("tool_calls", 0) or 0),
            input_tokens=int(run.telemetry.get("tokens_in", 0) or 0),
            output_tokens=int(run.telemetry.get("tokens_out", 0) or 0),
            latency_ms=run.latency_ms,
            estimated_cost=estimated_cost,
        ),
    )

    source_reports = {
        f"source_{index + 1}": text
        for index, text in enumerate(run.evidence)
        if (text or "").strip()
    }
    grounding: GroundingEvalResult = evaluate_claim_grounding(
        run.answer,
        source_reports,
    )

    observations = list(run.evidence)
    if run.answer:
        observations.append("FINAL ANSWER\n" + run.answer)
    completion = completion_assessor(case.contract(), observations)

    pit_violations, pit_checked = pit_parameter_score(
        run.tool_trace,
        case.as_of_date,
    )
    future_mentions = future_date_mentions(run.answer, case.as_of_date)

    telemetry = run.telemetry or {}
    return ArchitectureEvalResult(
        case_id=case.case_id,
        system_name=run.system_name,
        route_pass=routing.pass_case,
        action_correct=routing.action_correct,
        target_correct=routing.target_correct,
        forbidden_route_violation=routing.forbidden_violation,
        unnecessary_deep_research=routing.unnecessary_deep_research,
        claim_grounding_rate=grounding.claim_grounding_rate,
        unsupported_claim_rate=grounding.unsupported_claim_rate,
        evidence_coverage_rate=grounding.evidence_coverage_rate,
        completion_ratio=completion.completion_ratio,
        complete=completion.complete,
        pit_parameter_violations=pit_violations,
        checked_pit_parameters=pit_checked,
        future_date_mentions=future_mentions,
        latency_ms=float(run.latency_ms),
        llm_calls=int(telemetry.get("llm_calls", 0) or 0),
        tool_calls=int(telemetry.get("tool_calls", 0) or 0),
        tokens_in=int(telemetry.get("tokens_in", 0) or 0),
        tokens_out=int(telemetry.get("tokens_out", 0) or 0),
        estimated_cost=estimated_cost,
        response_status=run.response_status,
        audit_status=run.audit_status,
        error=run.error,
    )
