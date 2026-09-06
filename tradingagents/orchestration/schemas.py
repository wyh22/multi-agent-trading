from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


SupervisorActionName = Literal[
    "respond",
    "call_tool",
    "delegate_agent",
    "run_skill",
    "run_deep_research",
    "rollback",
]

ResponseStatus = Literal[
    "COMPLETE",
    "PARTIAL",
    "REVIEW_REQUIRED",
    "DATA_UNAVAILABLE",
    "SYSTEM_ERROR",
]


class SupervisorAction(BaseModel):
    """One bounded action chosen by the conversation supervisor."""

    action: SupervisorActionName
    target: str | None = Field(
        default=None,
        description=(
            "Tool, specialist agent, or skill name. "
            "Empty for respond/rollback when unnecessary."
        ),
    )
    objective: str = Field(
        default="",
        description="Concrete task to accomplish in this step.",
    )
    arguments: dict[str, Any] = Field(default_factory=dict)
    requires_validation: bool = False
    answer: str | None = Field(
        default=None,
        description="Optional direct answer when action=respond.",
    )


class TaskContract(BaseModel):
    """Explicit contract used to decide whether a user request is actually complete."""

    objective: str
    required_dimensions: list[str] = Field(default_factory=list)
    required_entities: list[str] = Field(default_factory=list)
    critical_requirements: list[str] = Field(default_factory=list)
    expected_output: str = "answer"
    can_be_partial: bool = True

    def checklist(self) -> list[str]:
        items: list[str] = []
        if self.required_entities and self.required_dimensions:
            items.extend(
                f"{entity}::{dimension}"
                for entity in self.required_entities
                for dimension in self.required_dimensions
            )
        elif self.required_dimensions:
            items.extend(self.required_dimensions)
        elif self.required_entities:
            items.extend(self.required_entities)
        return items


class CompletionAssessment(BaseModel):
    """Structured completion judgement after one or more capability executions."""

    complete: bool = False
    completion_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    completed_items: list[str] = Field(default_factory=list)
    missing_items: list[str] = Field(default_factory=list)
    critical_missing: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    reason: str = ""


class ResearchResponse(BaseModel):
    """User-facing response envelope with explicit degradation semantics."""

    status: ResponseStatus
    answer: str
    completed_items: list[str] = Field(default_factory=list)
    missing_items: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    unavailable_sources: list[str] = Field(default_factory=list)
    audit_status: str = ""
    user_action_required: str = ""


class ExecutionResult(BaseModel):
    """Normalized result returned by tools, agents, and skills."""

    status: Literal[
        "SUCCESS",
        "NO_DATA",
        "RATE_LIMIT",
        "TIMEOUT",
        "INVALID_ARGUMENT",
        "UNAVAILABLE",
        "FAILED",
    ] = "SUCCESS"
    capability: str
    content: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None
    retryable: bool = False
    fallback_available: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "SUCCESS"
