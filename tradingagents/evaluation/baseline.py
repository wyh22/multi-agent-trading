from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class SystemEvalSummary:
    system_name: str
    cases: int
    routing_accuracy: float | None = None
    claim_grounding_rate: float | None = None
    unsupported_claim_rate: float | None = None
    pit_violation_rate: float | None = None
    completeness_rate: float | None = None
    mean_tool_calls: float | None = None
    mean_tokens: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    mean_estimated_cost: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def compare_systems(rows: list[SystemEvalSummary]) -> list[dict]:
    """Return side-by-side metrics without declaring a winner automatically.

    The caller is responsible for ensuring identical model/data/tool/cutoff and
    comparable token budgets before interpreting differences.
    """
    return [row.to_dict() for row in rows]
