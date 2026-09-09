from pathlib import Path

from tradingagents.skills.registry import BUILTIN_SKILLS, SkillSpec, load_builtin_skills


EXPECTED_SKILLS = {
    "sector_discovery",
    "document_evidence_analysis",
    "company_comparison",
    "deep_stock_research",
}


def test_builtin_skill_manifests_load():
    skills = load_builtin_skills()
    assert EXPECTED_SKILLS.issubset(skills)
    assert all(isinstance(spec, SkillSpec) for spec in skills.values())


def test_skill_manifests_expose_routing_metadata():
    deep = BUILTIN_SKILLS["deep_stock_research"]
    doc = BUILTIN_SKILLS["document_evidence_analysis"]

    assert deep.requires_ticker is True
    assert deep.requires_audit is True
    assert deep.when_to_use
    assert "完成条件" in deep.routing_description()

    assert doc.requires_ticker is True
    assert "RAG" in doc.description or "检索" in doc.description
    assert "respect_as_of_date" in doc.constraints


def test_each_runtime_skill_has_skill_md():
    root = Path("tradingagents/skills")
    for name in EXPECTED_SKILLS:
        path = root / name / "SKILL.md"
        assert path.exists(), f"missing Skill specification: {path}"
        text = path.read_text(encoding="utf-8")
        assert f"# {name}" in text
        assert "## When to use" in text
        assert "## Constraints" in text


def test_no_duplicate_keyword_skill_registry():
    # Runtime routing is driven by ConversationSupervisor + CapabilityRegistry.
    # Keep a single source of truth under skills/manifests instead of a second
    # keyword matcher/registry.yaml that can drift from executable skill names.
    root = Path("tradingagents/skills")
    assert not (root / "registry.yaml").exists()
    assert not (root / "loader.py").exists()
