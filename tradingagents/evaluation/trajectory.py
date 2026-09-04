"""Compact trajectory records used by offline Agent evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from .dataset import ExpectedTrajectoryStep


@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryNode:
    analyst: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


@dataclass
class TrajectoryRecord:
    nodes: list[TrajectoryNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrajectoryScores:
    expected_steps: int = 0
    matched_steps: int = 0
    missing_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    future_date_violations: list[str] = field(default_factory=list)
    trajectory_pass_rate: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def _as_tool_call(item: Any) -> ToolCallRecord | None:
    if isinstance(item, ToolCallRecord):
        return item
    if not isinstance(item, dict):
        return None
    name = item.get("tool_name") or item.get("name")
    if not name:
        return None
    arguments = item.get("arguments") or item.get("args") or {}
    return ToolCallRecord(str(name), dict(arguments))


def record_from_graph_trace(trace: list[dict[str, Any]]) -> TrajectoryRecord:
    """Convert compact `analyst_trace` records into a normalized trajectory."""
    nodes: list[TrajectoryNode] = []
    for item in trace or []:
        if not isinstance(item, dict):
            continue
        analyst = str(item.get("analyst") or item.get("node") or item.get("agent") or "unknown")
        raw_calls = item.get("tool_calls") or []
        if not raw_calls and item.get("tool_name"):
            raw_calls = [item]
        calls = [call for raw in raw_calls if (call := _as_tool_call(raw)) is not None]
        nodes.append(TrajectoryNode(analyst=analyst, tool_calls=calls))
    return TrajectoryRecord(nodes=nodes)


def _date_violation(arguments: dict[str, Any], trade_date: str) -> list[str]:
    cutoff = date.fromisoformat(trade_date[:10])
    violations: list[str] = []
    for key in ("as_of_date", "curr_date", "trade_date", "end_date"):
        raw = arguments.get(key)
        if not raw:
            continue
        try:
            parsed = date.fromisoformat(str(raw)[:10])
        except ValueError:
            continue
        if parsed > cutoff:
            violations.append(f"{key}={raw}>{trade_date}")
    return violations


def score_trajectory(
    record: TrajectoryRecord,
    expected: list[ExpectedTrajectoryStep],
    *,
    trade_date: str,
) -> TrajectoryScores:
    scores = TrajectoryScores(expected_steps=len(expected))
    by_analyst = {node.analyst: node for node in record.nodes}

    for step in expected:
        node = by_analyst.get(step.analyst)
        if node is None:
            scores.missing_tools.extend(
                f"{step.analyst}:{tool}" for tool in step.expected_tools
            )
            continue

        actual = {call.tool_name for call in node.tool_calls}
        missing = [tool for tool in step.expected_tools if tool not in actual]
        forbidden = [tool for tool in step.forbidden_tools if tool in actual]
        if not missing and not forbidden:
            scores.matched_steps += 1
        scores.missing_tools.extend(f"{step.analyst}:{tool}" for tool in missing)
        scores.forbidden_tools.extend(f"{step.analyst}:{tool}" for tool in forbidden)

        for call in node.tool_calls:
            scores.future_date_violations.extend(
                f"{step.analyst}:{call.tool_name}:{item}"
                for item in _date_violation(call.arguments, trade_date)
            )

    checks = max(
        1,
        len(expected)
        + len(scores.missing_tools)
        + len(scores.forbidden_tools)
        + len(scores.future_date_violations),
    )
    passed = (
        scores.matched_steps
        if expected
        else 1
    )
    scores.trajectory_pass_rate = max(0.0, min(1.0, passed / checks))
    return scores
