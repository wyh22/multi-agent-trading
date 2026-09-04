"""Claim-aware evidence extraction and deterministic context compression.

The module deliberately avoids an extra LLM call. Analysts are encouraged to emit
explicit evidence tags; legacy/untagged reports fall back to conservative lexical
classification. Compression then preserves the most decision-useful claims under a
character budget while keeping claim type boundaries visible to downstream agents.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum


class ClaimType(str, Enum):
    """Semantic role of a statement in the research evidence chain."""

    FACT = "FACT"
    CALCULATION = "CALCULATION"
    INFERENCE = "INFERENCE"
    CONDITIONAL = "CONDITIONAL"


_TAG_ALIASES = {
    "FACT": ClaimType.FACT,
    "事实": ClaimType.FACT,
    "CALCULATION": ClaimType.CALCULATION,
    "CALC": ClaimType.CALCULATION,
    "计算": ClaimType.CALCULATION,
    "INFERENCE": ClaimType.INFERENCE,
    "推断": ClaimType.INFERENCE,
    "推理": ClaimType.INFERENCE,
    "CONDITIONAL": ClaimType.CONDITIONAL,
    "条件": ClaimType.CONDITIONAL,
    "条件性预测": ClaimType.CONDITIONAL,
    "条件预测": ClaimType.CONDITIONAL,
}

_TAG_RE = re.compile(
    r"^\s*(?:[-*+]\s*)?\[("
    + "|".join(re.escape(key) for key in sorted(_TAG_ALIASES, key=len, reverse=True))
    + r")\]\s*(.+?)\s*$",
    re.IGNORECASE,
)

_CONDITIONAL_RE = re.compile(
    r"(如果|若(?:是|出现|发生|未来|后续)?|一旦|除非|前提是|条件下|情景下|"
    r"在.{0,20}情况下|假设|if\b|when\b|unless\b|provided that\b|scenario\b)",
    re.IGNORECASE,
)
_CALCULATION_RE = re.compile(
    r"(计算(?:得|结果|公式)?|测算(?:得|结果)?|估算(?:得|结果)?|推导(?:得|结果)?|"
    r"由.{0,30}(?:计算|测算|估算|推导).{0,20}(?:得到|得出)?|"
    r"公式|CAGR\s*=|ROE\s*=|\d+(?:\.\d+)?\s*[%％]?\s*[+\-*/×÷]\s*"
    r"\d+(?:\.\d+)?|=\s*[-+]?\d)",
    re.IGNORECASE,
)
_INFERENCE_RE = re.compile(
    r"(表明|意味着|说明|反映|暗示|可能|预计|预期|推测|判断|认为|或将|倾向于|"
    r"suggests?\b|indicates?\b|implies?\b|likely\b|expects?\b|may\b|could\b)",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?")
_DATE_RE = re.compile(r"\b20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}")


@dataclass(frozen=True)
class EvidenceClaim:
    """One typed statement extracted from an analyst/researcher report."""

    text: str
    claim_type: ClaimType
    source_section: str
    ordinal: int
    explicit: bool = False

    @property
    def priority_score(self) -> float:
        """Compression priority, not epistemic confidence.

        Facts/calculations receive more budget because they are the grounding layer.
        Conditional scenarios are preserved ahead of generic inference when possible
        because they carry explicit triggers rather than unconditional predictions.
        """

        base = {
            ClaimType.FACT: 4.0,
            ClaimType.CALCULATION: 3.8,
            ClaimType.CONDITIONAL: 3.0,
            ClaimType.INFERENCE: 2.5,
        }[self.claim_type]
        if self.explicit:
            base += 0.8
        if self.claim_type in {ClaimType.FACT, ClaimType.CALCULATION} and (
            _NUMERIC_RE.search(self.text) or _DATE_RE.search(self.text)
        ):
            base += 0.35
        if len(self.text) <= 180:
            base += 0.1
        return base

    def render(self) -> str:
        return f"- [{self.claim_type.value}] {self.text}"


@dataclass(frozen=True)
class ClaimCompressionResult:
    """Structured result retained for tests, observability and future metrics."""

    claims: tuple[EvidenceClaim, ...]
    rendered: str
    original_count: int
    dropped_count: int
    type_counts: dict[str, int]


def classify_claim(text: str) -> ClaimType:
    """Conservatively classify an untagged statement.

    Precedence matters: an explicit condition is a conditional scenario even if it
    also contains words such as "可能"; formula/derivation markers come next.
    Plain interpretations are inference. Everything else remains a fact candidate.
    """

    value = text.strip()
    if _CONDITIONAL_RE.search(value):
        return ClaimType.CONDITIONAL
    if _CALCULATION_RE.search(value):
        return ClaimType.CALCULATION
    if _INFERENCE_RE.search(value):
        return ClaimType.INFERENCE
    return ClaimType.FACT


def _clean_chunk(chunk: str) -> str:
    value = chunk.strip()
    value = re.sub(r"^#{1,6}\s*", "", value)
    value = re.sub(r"^[-*+]\s+", "", value)
    value = re.sub(r"^>\s*", "", value)
    if _TABLE_SEPARATOR_RE.match(value):
        return ""
    if value.startswith("|") and value.endswith("|"):
        value = " · ".join(part.strip() for part in value.strip("|").split("|") if part.strip())
    value = re.sub(r"\s+", " ", value).strip()
    if value.lower() in {"evidence claims", "claim summary", "证据声明", "关键证据声明"}:
        return ""
    return value


def _fallback_chunks(text: str) -> list[str]:
    chunks = re.split(r"(?<=[。！？!?；;])\s*|\n+", text)
    cleaned: list[str] = []
    for chunk in chunks:
        value = _clean_chunk(chunk)
        if len(value) < 4:
            continue
        if len(value) <= 360:
            cleaned.append(value)
            continue

        # Long legacy paragraphs are split conservatively on commas and then
        # re-grouped, producing claim-sized units without semantic rewriting.
        clauses = [item.strip() for item in re.split(r"(?<=[，,])\s*", value) if item.strip()]
        buffer = ""
        for clause in clauses:
            candidate = f"{buffer}{clause}" if not buffer else f"{buffer} {clause}"
            if buffer and len(candidate) > 260:
                cleaned.append(buffer)
                buffer = clause
            else:
                buffer = candidate
        if buffer:
            cleaned.append(buffer)
    return cleaned


def extract_claims(text: str | None, source_section: str) -> list[EvidenceClaim]:
    """Extract explicit tagged claims, with a deterministic fallback for old reports."""

    value = (text or "").strip()
    if not value:
        return []

    explicit_claims: list[EvidenceClaim] = []
    for ordinal, line in enumerate(value.splitlines()):
        match = _TAG_RE.match(line)
        if not match:
            continue
        raw_tag, raw_text = match.groups()
        tag = _TAG_ALIASES.get(raw_tag.upper()) or _TAG_ALIASES.get(raw_tag)
        if tag is None:
            continue
        cleaned = _clean_chunk(raw_text)
        if cleaned:
            explicit_claims.append(
                EvidenceClaim(
                    text=cleaned,
                    claim_type=tag,
                    source_section=source_section,
                    ordinal=ordinal,
                    explicit=True,
                )
            )

    # If an analyst emitted the requested tagged section, use it as the compact
    # semantic interface and avoid re-parsing the verbose body/table.
    if explicit_claims:
        return explicit_claims

    return [
        EvidenceClaim(
            text=chunk,
            claim_type=classify_claim(chunk),
            source_section=source_section,
            ordinal=ordinal,
            explicit=False,
        )
        for ordinal, chunk in enumerate(_fallback_chunks(value))
    ]


def _fits(current: list[EvidenceClaim], candidate: EvidenceClaim, max_chars: int) -> bool:
    lines = [claim.render() for claim in current] + [candidate.render()]
    return len("\n".join(lines)) <= max_chars


def compress_claims(
    text: str | None,
    *,
    source_section: str,
    max_chars: int,
) -> ClaimCompressionResult:
    """Compress a report by claim type and priority under a character budget."""

    claims = extract_claims(text, source_section)
    if not claims or max_chars <= 0:
        return ClaimCompressionResult(
            claims=(),
            rendered="",
            original_count=len(claims),
            dropped_count=len(claims),
            type_counts={},
        )

    full_rendered = "\n".join(claim.render() for claim in claims)
    if len(full_rendered) <= max_chars:
        counts = Counter(claim.claim_type.value for claim in claims)
        return ClaimCompressionResult(
            claims=tuple(claims),
            rendered=full_rendered,
            original_count=len(claims),
            dropped_count=0,
            type_counts=dict(counts),
        )

    selected: list[EvidenceClaim] = []
    selected_ordinals: set[int] = set()

    # Preserve claim-type diversity first, then use the remaining budget by score.
    for claim_type in (
        ClaimType.FACT,
        ClaimType.CALCULATION,
        ClaimType.CONDITIONAL,
        ClaimType.INFERENCE,
    ):
        candidates = [claim for claim in claims if claim.claim_type is claim_type]
        candidates.sort(key=lambda item: (-item.priority_score, item.ordinal))
        if not candidates:
            continue
        candidate = candidates[0]
        if _fits(selected, candidate, max_chars):
            selected.append(candidate)
            selected_ordinals.add(candidate.ordinal)

    remaining = [claim for claim in claims if claim.ordinal not in selected_ordinals]
    remaining.sort(key=lambda item: (-item.priority_score, item.ordinal))
    for claim in remaining:
        if _fits(selected, claim, max_chars):
            selected.append(claim)
            selected_ordinals.add(claim.ordinal)

    if not selected:
        best = max(claims, key=lambda item: item.priority_score)
        prefix = f"- [{best.claim_type.value}] "
        available = max(0, max_chars - len(prefix) - 1)
        shortened = best.text[:available].rstrip()
        if len(shortened) < len(best.text) and available > 1:
            shortened = shortened[:-1].rstrip() + "…"
        selected = [
            EvidenceClaim(
                text=shortened,
                claim_type=best.claim_type,
                source_section=best.source_section,
                ordinal=best.ordinal,
                explicit=best.explicit,
            )
        ]

    selected.sort(key=lambda item: item.ordinal)
    rendered = "\n".join(claim.render() for claim in selected)
    counts = Counter(claim.claim_type.value for claim in selected)
    return ClaimCompressionResult(
        claims=tuple(selected),
        rendered=rendered,
        original_count=len(claims),
        dropped_count=max(0, len(claims) - len(selected)),
        type_counts=dict(counts),
    )


def claim_boundary_instruction() -> str:
    """Stable instruction shared by analyst prompts."""

    return (
        "\n\nAt the end of the final report, append a compact `## Evidence Claims` "
        "section with 4-8 bullet points. Every bullet must begin with exactly one "
        "of `[FACT]`, `[CALCULATION]`, `[INFERENCE]`, or `[CONDITIONAL]`. "
        "[FACT] is directly supported by retrieved/tool data; [CALCULATION] is a "
        "derived numeric result and should state the inputs/formula when material; "
        "[INFERENCE] is interpretation from evidence; [CONDITIONAL] is a future "
        "scenario that must include its trigger/condition. Never upgrade an "
        "inference or conditional scenario into a fact."
    )


def claim_usage_instruction() -> str:
    """Explain typed evidence semantics to downstream reasoning agents."""

    return (
        "The compact evidence package uses four claim types: "
        "[FACT] = directly supported by tool/retrieved data; "
        "[CALCULATION] = a derived result whose inputs/formula must remain traceable; "
        "[INFERENCE] = an interpretation that must not be restated as an observed fact; "
        "[CONDITIONAL] = a forward-looking scenario valid only when its stated trigger "
        "or condition holds. Prefer FACT/CALCULATION for grounding, preserve the "
        "conditions attached to CONDITIONAL claims, and treat INFERENCE as lower-level "
        "interpretation rather than source truth."
    )
