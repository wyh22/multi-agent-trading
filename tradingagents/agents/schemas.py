"""智能体结构化输出模型。

最终版只保留真正有必要结构化约束的两个环节：
1. 投资组合经理：负责综合分析结果并给出研究评级；
2. 决策审计智能体：负责检查事实依据、时点合规和结论一致性。

结构化模型用于约束字段格式，最终仍会渲染为 Markdown，方便 CLI、
记忆模块和报告写入器继续消费自然语言结果。
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class PortfolioRating(str, Enum):
    """五档研究评级。"""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class PortfolioDecision(BaseModel):
    """投资组合经理输出的最终研究结论。"""

    rating: PortfolioRating = Field(
        description="最终研究评级，只能取 Buy / Overweight / Hold / Underweight / Sell。",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="结论置信度，根据数据覆盖率、证据一致性和多空分歧程度判断。",
    )
    executive_summary: str = Field(
        description="两到四句话概括结论、主要依据和当前应对方式，避免重复前文。",
    )
    investment_thesis: str = Field(
        description="核心投资逻辑，只保留能够由上游分析报告支持的关键事实和推断。",
    )
    key_risks: list[str] = Field(
        default_factory=list,
        description="最重要的三到五项风险，每项尽量简短且可由现有证据追溯。",
    )
    catalysts: list[str] = Field(
        default_factory=list,
        description="可能改变当前判断的正向或负向催化因素，最多五项。",
    )
    invalidation_conditions: list[str] = Field(
        default_factory=list,
        description="一旦出现就应重新评估当前结论的失效条件，最多五项。",
    )
    position_guidance: str | None = Field(
        default=None,
        description="研究视角下的仓位或关注度建议，不要求给出精确价格目标。",
    )
    time_horizon: str | None = Field(
        default=None,
        description="建议观察或验证结论的时间周期，例如 1-3 个月。",
    )


class AuditIssue(BaseModel):
    """One actionable audit defect and the capability responsible for repairing it."""

    issue_type: Literal[
        "missing_evidence",
        "unsupported_claim",
        "numeric_conflict",
        "pit_violation",
        "reasoning_conflict",
        "source_conflict",
    ] = Field(description="审计问题类型。")
    repair_target: Literal[
        "market",
        "news",
        "fundamentals",
        "portfolio_manager",
        "rag",
    ] = Field(description="最适合修复该问题的责任能力。")
    affected_claims: list[str] = Field(
        default_factory=list,
        description="受影响的关键陈述，最多五项。",
    )
    instruction: str = Field(description="明确、可执行的重新取证或修订要求。")


class AuditResult(BaseModel):
    """决策审计智能体的结构化检查结果。"""

    verdict: Literal["PASS", "REVISE"] = Field(
        description="PASS 表示可以直接发布；REVISE 表示需要投资组合经理修订一次。",
    )
    grounding_score: float = Field(
        ge=0.0,
        le=1.0,
        description="事实与上游证据对齐程度，0 到 1。",
    )
    pit_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Point-in-Time 时点合规程度，0 到 1。",
    )
    consistency_score: float = Field(
        ge=0.0,
        le=1.0,
        description="数字、方向和逻辑的一致性程度，0 到 1。",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="无法从现有证据直接支持的陈述，最多列出五项。",
    )
    issues: list[AuditIssue] = Field(
        default_factory=list,
        description="需要修复的结构化问题；PASS 时应为空。",
    )
    revision_instructions: list[str] = Field(
        default_factory=list,
        description="若需要修订，给出最多五条明确、可执行的修改要求。",
    )
    summary: str = Field(
        description="一句到三句审计摘要。",
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    """将投资组合经理的结构化结果渲染为紧凑 Markdown。"""

    parts = [
        f"**Rating**: {decision.rating.value}",
        f"**置信度**: {decision.confidence}",
        "",
        f"**执行摘要**: {decision.executive_summary}",
        "",
        f"**核心投资逻辑**: {decision.investment_thesis}",
    ]
    if decision.key_risks:
        parts.extend(["", "**关键风险**:"])
        parts.extend(f"- {item}" for item in decision.key_risks[:5])
    if decision.catalysts:
        parts.extend(["", "**关键催化**:"])
        parts.extend(f"- {item}" for item in decision.catalysts[:5])
    if decision.invalidation_conditions:
        parts.extend(["", "**失效条件**:"])
        parts.extend(f"- {item}" for item in decision.invalidation_conditions[:5])
    if decision.position_guidance:
        parts.extend(["", f"**仓位/关注度建议**: {decision.position_guidance}"])
    if decision.time_horizon:
        parts.extend(["", f"**观察周期**: {decision.time_horizon}"])
    return "\n".join(parts)


def render_audit_result(result: AuditResult) -> str:
    """将审计结果渲染为便于保存和人工检查的 Markdown。"""

    parts = [
        f"**审计结论**: {result.verdict}",
        (
            "**评分**: "
            f"证据对齐 {result.grounding_score:.2f} / "
            f"PIT {result.pit_score:.2f} / "
            f"一致性 {result.consistency_score:.2f}"
        ),
        "",
        f"**摘要**: {result.summary}",
    ]
    if result.unsupported_claims:
        parts.extend(["", "**无充分证据的陈述**:"])
        parts.extend(f"- {item}" for item in result.unsupported_claims[:5])
    if result.issues:
        parts.extend(["", "**结构化修复路由**:"])
        for issue in result.issues[:5]:
            parts.append(
                f"- [{issue.issue_type}] -> {issue.repair_target}: {issue.instruction}"
            )
    if result.revision_instructions:
        parts.extend(["", "**修订要求**:"])
        parts.extend(f"- {item}" for item in result.revision_instructions[:5])
    return "\n".join(parts)
