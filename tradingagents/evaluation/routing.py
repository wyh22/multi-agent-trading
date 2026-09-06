from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class RoutingEvalCase:
    case_id: str
    user_query: str
    ticker: str | None = None
    as_of_date: str = ""
    expected_actions: list[str] = field(default_factory=list)
    expected_targets: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    forbidden_targets: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, row: dict) -> "RoutingEvalCase":
        return cls(**row)


@dataclass
class RoutingRecord:
    action: str
    target: str | None = None
    steps: int = 1
    tool_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    estimated_cost: float | None = None


@dataclass
class RoutingEvalResult:
    case_id: str
    action_correct: bool
    target_correct: bool
    forbidden_violation: bool
    unnecessary_deep_research: bool
    record: RoutingRecord

    @property
    def pass_case(self) -> bool:
        return (
            self.action_correct
            and self.target_correct
            and not self.forbidden_violation
        )

    def to_dict(self) -> dict:
        row = asdict(self)
        row["pass_case"] = self.pass_case
        return row


def load_routing_eval(path: str | Path) -> list[RoutingEvalCase]:
    cases = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                cases.append(RoutingEvalCase.from_dict(json.loads(line)))
    return cases


def evaluate_route(
    case: RoutingEvalCase,
    record: RoutingRecord,
) -> RoutingEvalResult:
    action_ok = (
        not case.expected_actions
        or record.action in set(case.expected_actions)
    )
    target_ok = (
        not case.expected_targets
        or (record.target or "") in set(case.expected_targets)
    )
    forbidden = (
        record.action in set(case.forbidden_actions)
        or (record.target or "") in set(case.forbidden_targets)
    )
    unnecessary_deep = (
        record.action in {"run_deep_research", "run_skill"}
        and (record.target or "") == "deep_stock_research"
        and "run_deep_research" not in set(case.expected_actions)
        and "deep_stock_research" not in set(case.expected_targets)
    )
    return RoutingEvalResult(
        case_id=case.case_id,
        action_correct=action_ok,
        target_correct=target_ok,
        forbidden_violation=forbidden,
        unnecessary_deep_research=unnecessary_deep,
        record=record,
    )


def run_routing_eval(
    cases: list[RoutingEvalCase],
    route_fn: Callable[[RoutingEvalCase], RoutingRecord],
) -> list[RoutingEvalResult]:
    return [evaluate_route(case, route_fn(case)) for case in cases]


def summarize_routing(results: list[RoutingEvalResult]) -> dict:
    total = max(1, len(results))
    passed = sum(item.pass_case for item in results)
    return {
        "cases": len(results),
        "route_accuracy": passed / total,
        "action_accuracy": sum(item.action_correct for item in results) / total,
        "target_accuracy": sum(item.target_correct for item in results) / total,
        "forbidden_violation_rate": (
            sum(item.forbidden_violation for item in results) / total
        ),
        "unnecessary_deep_research_rate": (
            sum(item.unnecessary_deep_research for item in results) / total
        ),
        "mean_steps": (
            sum(item.record.steps for item in results) / total
        ),
        "mean_tool_calls": (
            sum(item.record.tool_calls for item in results) / total
        ),
    }
