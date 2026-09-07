from pathlib import Path

from tradingagents.skills.loader import SkillRegistry


def test_skill_registry_loads():
    registry = SkillRegistry(Path("tradingagents/skills/registry.yaml"))
    names = [s["name"] for s in registry.list_skills()]
    assert "company_research" in names
    assert "deep_research" in names


def test_skill_matching():
    registry = SkillRegistry(Path("tradingagents/skills/registry.yaml"))
    result = registry.match("请分析贵州茅台基本面")
    assert any(item["name"] == "company_research" for item in result)
