from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    requires_ticker: bool = False
    requires_audit: bool = False


BUILTIN_SKILLS: dict[str, SkillSpec] = {
    "deep_stock_research": SkillSpec(
        name="deep_stock_research",
        description="完整单股研究：Market/News/Fundamentals -> Bull/Bear -> PM -> Auditor。只在复杂综合研究时使用。",
        requires_ticker=True,
        requires_audit=True,
    ),
    "sector_discovery": SkillSpec(
        name="sector_discovery",
        description="确定性行业发现与研究优先级排序；不让 LLM 直接计算行业排名。",
    ),
    "document_evidence_analysis": SkillSpec(
        name="document_evidence_analysis",
        description="围绕公司已入库 PDF/DOCX/公告/财报做 PIT-aware 证据检索与回答。",
        requires_ticker=True,
    ),
    "company_comparison": SkillSpec(
        name="company_comparison",
        description="多公司横向比较技能；由 Supervisor 分解为多个专业研究动作后综合。",
    ),
}
