from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass

from tradingagents.agents.utils.evidence_claims import (
    ClaimType,
    EvidenceClaim,
    extract_claims,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_.%-]+|[\u4e00-\u9fff]{2,}")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?%?")


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text or "")
        if len(token.strip()) >= 2
    }


def _numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text or ""))


def _support_score(claim: EvidenceClaim, evidence: EvidenceClaim) -> float:
    claim_tokens = _tokens(claim.text)
    evidence_tokens = _tokens(evidence.text)
    lexical = (
        len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))
    )
    claim_numbers = _numbers(claim.text)
    evidence_numbers = _numbers(evidence.text)
    number_ok = not claim_numbers or claim_numbers.issubset(evidence_numbers)

    if claim.claim_type in {ClaimType.FACT, ClaimType.CALCULATION}:
        return lexical if number_ok else 0.0
    # Inference/conditional claims are not treated as source truth; this score
    # only verifies that their stated reasoning is anchored to some evidence.
    return lexical * (1.0 if evidence_tokens else 0.0)


@dataclass
class ClaimGroundingRecord:
    claim_text: str
    claim_type: str
    supported: bool
    best_support_score: float
    source_section: str | None = None


@dataclass
class GroundingEvalResult:
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    claim_grounding_rate: float
    unsupported_claim_rate: float
    evidence_coverage_rate: float
    type_counts: dict[str, int]
    records: list[ClaimGroundingRecord]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_claim_grounding(
    final_text: str,
    source_reports: dict[str, str],
    *,
    support_threshold: float = 0.22,
) -> GroundingEvalResult:
    final_claims = extract_claims(final_text, "final")
    evidence_claims: list[EvidenceClaim] = []
    for section, text in source_reports.items():
        evidence_claims.extend(extract_claims(text, section))

    records: list[ClaimGroundingRecord] = []
    used_sources: set[str] = set()
    for claim in final_claims:
        best = 0.0
        best_section = None
        for evidence in evidence_claims:
            score = _support_score(claim, evidence)
            if score > best:
                best = score
                best_section = evidence.source_section
        supported = best >= support_threshold
        if supported and best_section:
            used_sources.add(best_section)
        records.append(
            ClaimGroundingRecord(
                claim_text=claim.text,
                claim_type=claim.claim_type.value,
                supported=supported,
                best_support_score=best,
                source_section=best_section,
            )
        )

    total = len(records)
    supported_count = sum(item.supported for item in records)
    type_counts = Counter(item.claim_type for item in records)
    source_count = len([v for v in source_reports.values() if (v or "").strip()])
    return GroundingEvalResult(
        total_claims=total,
        supported_claims=supported_count,
        unsupported_claims=max(0, total - supported_count),
        claim_grounding_rate=(
            supported_count / total if total else 1.0
        ),
        unsupported_claim_rate=(
            (total - supported_count) / total if total else 0.0
        ),
        evidence_coverage_rate=(
            len(used_sources) / source_count if source_count else 1.0
        ),
        type_counts=dict(type_counts),
        records=records,
    )
