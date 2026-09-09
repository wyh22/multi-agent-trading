from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SkillSpec:
    """Machine-readable description of a reusable research skill.

    The YAML manifest is the runtime source of truth. SKILL.md files document the
    same contract for reviewers and interview/demo use; they are not a second
    keyword router.
    """

    name: str
    description: str
    requires_ticker: bool = False
    requires_audit: bool = False
    allowed_agents: tuple[str, ...] = field(default_factory=tuple)
    when_to_use: tuple[str, ...] = field(default_factory=tuple)
    constraints: tuple[str, ...] = field(default_factory=tuple)
    completion: str = ""

    def routing_description(self) -> str:
        """Render concise metadata for the Supervisor capability catalog."""

        parts = [self.description.strip()]
        if self.when_to_use:
            parts.append("适用场景：" + "；".join(self.when_to_use))
        if self.completion:
            parts.append("完成条件：" + self.completion.strip())
        return " ".join(part for part in parts if part)


def _fallback_specs() -> dict[str, SkillSpec]:
    return {
        "deep_stock_research": SkillSpec(
            name="deep_stock_research",
            description="完整单股多阶段深度研究并经过独立审查。",
            requires_ticker=True,
            requires_audit=True,
            allowed_agents=("market", "news", "fundamentals"),
            when_to_use=("综合单股研究", "需要跨行情/新闻/基本面证据的完整研究"),
            constraints=("research_only", "respect_as_of_date", "evidence_required"),
            completion="最终研究结论必须经过 Auditor，REVISE 时进入定向修复回路。",
        ),
        "sector_discovery": SkillSpec(
            name="sector_discovery",
            description="确定性行业发现和代表性研究池。",
            when_to_use=("行业研究优先级", "代表性研究池构建"),
            constraints=("deterministic_ranking", "research_priority_not_buy_list"),
            completion="返回 PIT-safe 行业研究优先级或代表性研究池。",
        ),
        "document_evidence_analysis": SkillSpec(
            name="document_evidence_analysis",
            description="对已入库公司文档执行 PIT-aware 证据检索。",
            requires_ticker=True,
            allowed_agents=("news", "fundamentals"),
            when_to_use=("财报/公告原文证据分析", "上一轮研究存在文档型证据缺口"),
            constraints=("rag_required", "preserve_provenance", "respect_as_of_date"),
            completion="回答必须保留文档来源和发布日期语义，不得将检索缺失解释为事实不存在。",
        ),
        "company_comparison": SkillSpec(
            name="company_comparison",
            description="多公司横向证据比较。",
            allowed_agents=("market", "news", "fundamentals"),
            when_to_use=("至少两个明确标的的横向比较",),
            constraints=("at_least_two_tickers", "independent_evidence_per_company"),
            completion="至少两个标的，比较结论必须来自各标的独立证据。",
        ),
    }


def load_builtin_skills() -> dict[str, SkillSpec]:
    """Load declarative skill manifests, with code defaults for packaged fallback."""

    root = Path(__file__).with_name("manifests")
    specs: dict[str, SkillSpec] = {}
    if root.exists():
        for path in sorted(root.glob("*.yaml")):
            try:
                row = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                spec = SkillSpec(
                    name=str(row["name"]),
                    description=str(row.get("description", "")),
                    requires_ticker=bool(row.get("requires_ticker", False)),
                    requires_audit=bool(row.get("requires_audit", False)),
                    allowed_agents=tuple(row.get("allowed_agents", []) or []),
                    when_to_use=tuple(row.get("when_to_use", []) or []),
                    constraints=tuple(row.get("constraints", []) or []),
                    completion=str(row.get("completion", "") or ""),
                )
                specs[spec.name] = spec
            except Exception:
                continue
    return specs or _fallback_specs()


BUILTIN_SKILLS = load_builtin_skills()
