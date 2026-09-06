"""Claim-aware context compression with separate evidence/hypothesis channels.

Raw reports remain persisted for audit. Downstream agents receive two ledgers:
- Evidence Ledger: FACT / CALCULATION, optimized for grounding fidelity;
- Hypothesis Ledger: INFERENCE / CONDITIONAL, preserving reasoning diversity.

This avoids letting factual compression silently erase the hypotheses that make
deep research useful, while still preventing hypotheses from becoming source truth.
"""

from __future__ import annotations

from tradingagents.agents.utils.evidence_claims import (
    ClaimType,
    EvidenceClaim,
    extract_claims,
)


def compact_text(text: str | None, max_chars: int = 2600) -> str:
    """Character-budget truncation for non-evidence text."""

    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    head = int(max_chars * 0.72)
    tail = max_chars - head
    return (
        value[:head].rstrip()
        + "\n\n……【中间内容已按上下文预算省略，原始报告仍保存在 trace 目录】……\n\n"
        + value[-tail:].lstrip()
    )


def _select_claims(
    claims: list[EvidenceClaim],
    max_chars: int,
) -> list[EvidenceClaim]:
    if not claims or max_chars <= 0:
        return []
    ranked = sorted(
        claims,
        key=lambda item: (-item.priority_score, item.ordinal),
    )
    selected: list[EvidenceClaim] = []
    current = 0
    for claim in ranked:
        rendered = claim.render()
        extra = len(rendered) + (1 if selected else 0)
        if current + extra <= max_chars:
            selected.append(claim)
            current += extra
    if not selected:
        best = ranked[0]
        prefix = f"- [{best.claim_type.value}] "
        available = max(0, max_chars - len(prefix))
        if available:
            selected.append(
                EvidenceClaim(
                    text=best.text[:available].rstrip(),
                    claim_type=best.claim_type,
                    source_section=best.source_section,
                    ordinal=best.ordinal,
                    explicit=best.explicit,
                )
            )
    return sorted(selected, key=lambda item: item.ordinal)


def compact_evidence_text(
    text: str | None,
    *,
    source_section: str,
    max_chars: int,
) -> str:
    """Render separate grounding and hypothesis ledgers under one budget."""

    claims = extract_claims(text, source_section)
    if not claims or max_chars <= 0:
        return ""

    evidence = [
        claim
        for claim in claims
        if claim.claim_type in {ClaimType.FACT, ClaimType.CALCULATION}
    ]
    hypotheses = [
        claim
        for claim in claims
        if claim.claim_type in {ClaimType.INFERENCE, ClaimType.CONDITIONAL}
    ]

    if evidence and hypotheses:
        evidence_budget = max(1, int(max_chars * 0.62))
        hypothesis_budget = max(1, max_chars - evidence_budget - 45)
    elif evidence:
        evidence_budget = max_chars
        hypothesis_budget = 0
    else:
        evidence_budget = 0
        hypothesis_budget = max_chars

    evidence_selected = _select_claims(evidence, evidence_budget)
    hypothesis_selected = _select_claims(hypotheses, hypothesis_budget)

    parts = []
    if evidence_selected:
        parts.append(
            "### Evidence Ledger\n"
            + "\n".join(item.render() for item in evidence_selected)
        )
    if hypothesis_selected:
        parts.append(
            "### Hypothesis Ledger\n"
            + "\n".join(item.render() for item in hypothesis_selected)
        )
    return "\n\n".join(parts)


def build_analyst_context(state: dict, per_report_chars: int = 2200) -> str:
    """Build dual-ledger analyst context for Bull/Bear."""

    sections = [
        ("市场与技术面", state.get("market_report", "")),
        ("新闻、公告、宏观与情绪", state.get("news_report", "")),
        ("基本面", state.get("fundamentals_report", "")),
    ]
    parts: list[str] = []
    for title, content in sections:
        if not content:
            continue
        compressed = compact_evidence_text(
            content,
            source_section=title,
            max_chars=per_report_chars,
        )
        if compressed:
            parts.append(f"## {title}\n{compressed}")
    return "\n\n".join(parts)


def build_decision_context(state: dict) -> str:
    """Build PM/Auditor context while preserving evidence and hypotheses separately."""

    analyst = build_analyst_context(state, per_report_chars=1500)
    bull = compact_evidence_text(
        state.get("bull_thesis", ""),
        source_section="看多研究",
        max_chars=1600,
    )
    bear = compact_evidence_text(
        state.get("bear_thesis", ""),
        source_section="看空研究",
        max_chars=1600,
    )
    parts = [analyst]
    if bull:
        parts.append(f"## 看多论点\n{bull}")
    if bear:
        parts.append(f"## 看空论点\n{bear}")
    return "\n\n".join(part for part in parts if part)
