from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CapabilityKind = Literal["tool", "agent", "skill"]


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    kind: CapabilityKind
    description: str
    requires_ticker: bool = False
    expensive: bool = False


class CapabilityRegistry:
    """Small explicit registry so the LLM sees capabilities, not implementation details."""

    def __init__(self):
        self._items: dict[str, CapabilitySpec] = {}

    def register(self, spec: CapabilitySpec) -> None:
        if spec.name in self._items:
            return
        self._items[spec.name] = spec

    def get(self, name: str) -> CapabilitySpec | None:
        return self._items.get(name)

    def list(self, *, kind: CapabilityKind | None = None) -> list[CapabilitySpec]:
        values = list(self._items.values())
        return [item for item in values if kind is None or item.kind == kind]

    def prompt_catalog(self) -> str:
        lines = []
        for item in sorted(self._items.values(), key=lambda x: (x.kind, x.name)):
            flags = []
            if item.requires_ticker:
                flags.append("requires_ticker")
            if item.expensive:
                flags.append("expensive")
            suffix = f" [{' '.join(flags)}]" if flags else ""
            lines.append(f"- {item.kind}:{item.name}{suffix} — {item.description}")
        return "\n".join(lines)
