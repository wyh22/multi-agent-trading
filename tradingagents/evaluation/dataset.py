"""Data model and JSONL loader for deterministic Agent evaluation cases."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExpectedTrajectoryStep:
    analyst: str
    expected_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)


@dataclass
class EvalCase:
    case_id: str
    ticker: str
    trade_date: str
    asset_type: str = "stock"
    analyst_under_test: str = "market"
    forbidden_future_date: str = ""
    expected_constraints: list[str] = field(default_factory=list)
    forbidden_constraints: list[str] = field(default_factory=list)
    expected_analyst_reports: list[str] = field(default_factory=list)
    expected_trajectory: list[ExpectedTrajectoryStep] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.forbidden_future_date:
            self.forbidden_future_date = self.trade_date

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalCase":
        payload = dict(data)
        payload["expected_trajectory"] = [
            item if isinstance(item, ExpectedTrajectoryStep) else ExpectedTrajectoryStep(**item)
            for item in payload.get("expected_trajectory", [])
        ]
        return cls(**payload)


@dataclass
class AgentEvalDataset:
    cases: list[EvalCase]
    name: str = "a_share_agent_eval"

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)


def load_eval_dataset(path: str | Path) -> AgentEvalDataset:
    source = Path(path)
    cases: list[EvalCase] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(EvalCase.from_dict(json.loads(line)))
    return AgentEvalDataset(cases=cases, name=source.stem)
