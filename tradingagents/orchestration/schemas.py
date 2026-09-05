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


class SupervisorAction(BaseModel):
    """One bounded action chosen by the conversation supervisor."""

    action: SupervisorActionName
    target: str | None = Field(
        default=None,
        description="Tool, specialist agent, or skill name. Empty for respond/rollback when unnecessary.",
    )
    objective: str = Field(
        default="",
        description="Concrete task to accomplish in this step; keep it short and operational.",
    )
    arguments: dict[str, Any] = Field(default_factory=dict)
    requires_validation: bool = False
    answer: str | None = Field(
        default=None,
        description="Optional direct answer when action=respond.",
    )


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
