"""Declarative skill registry loader.

Skills are metadata driven capabilities selected by the conversation routing layer.
"""

from pathlib import Path
import yaml


class SkillRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.skills = self._load()

    def _load(self):
        with self.path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("skills", [])

    def list_skills(self):
        return self.skills

    def match(self, query: str):
        query = query.lower()
        matched = []
        for skill in self.skills:
            triggers = skill.get("triggers", [])
            if any(str(t).lower() in query for t in triggers):
                matched.append(skill)
        return matched
