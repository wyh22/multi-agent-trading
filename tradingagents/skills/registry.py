from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    requires_ticker: bool = False
    requires_audit: bool = False
    allowed_agents: tuple[str, ...] = field(default_factory=tuple)
    completion: str = ""


def _fallback_specs() -> dict[str, SkillSpec]:
    return {
        "deep_stock_research": SkillSpec(
            name="deep_stock_research",
            description="完整单股多角色深度研究并经过 Auditor。",
            requires_ticker=True,
            requires_audit=True,
            allowed_agents=("market", "news", "fundamentals"),
        ),
        "sector_discovery": SkillSpec(
            name="sector_discovery",
            description="确定性行业发现和代表性研究池。",
        ),
        "document_evidence_analysis": SkillSpec(
            name="document_evidence_analysis",
            description="对已入库公司文档执行 PIT-aware 证据检索。",
            requires_ticker=True,
            allowed_agents=("news", "fundamentals"),
        ),
        "company_comparison": SkillSpec(
            name="company_comparison",
            description="多公司横向证据比较。",
            allowed_agents=("market", "news", "fundamentals"),
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
                    completion=str(row.get("completion", "") or ""),
                )
                specs[spec.name] = spec
            except Exception:
                continue
    return specs or _fallback_specs()


BUILTIN_SKILLS = load_builtin_skills()
