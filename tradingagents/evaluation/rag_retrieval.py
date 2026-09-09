from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable

from tradingagents.rag.evidence_pack import source_document_key
from tradingagents.rag.models import RetrievalHit


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    ticker: str
    query: str
    as_of_date: str
    dimension: str = ""
    company_name: str = ""
    industry: str = ""
    doc_type: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class EvidenceJudgment:
    case_id: str
    relevance: int
    chunk_id: str = ""
    document_key: str = ""
    page: int | None = None
    notes: str = ""

    def __post_init__(self):
        if int(self.relevance) < 0:
            raise ValueError("relevance must be >= 0")


@dataclass
class RetrievalMetricRow:
    case_id: str
    strategy: str
    relevant_evidence: int
    relevant_documents: int
    returned: int
    latency_ms: float = 0.0
    pit_violations: int = 0
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "strategy": self.strategy,
            "relevant_evidence": self.relevant_evidence,
            "relevant_documents": self.relevant_documents,
            "returned": self.returned,
            "latency_ms": self.latency_ms,
            "pit_violations": self.pit_violations,
            **self.metrics,
        }


def load_retrieval_cases(path: str | Path) -> list[RetrievalCase]:
    rows: list[RetrievalCase] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            payload = json.loads(raw)
            try:
                rows.append(
                    RetrievalCase(
                        case_id=str(payload["case_id"]),
                        ticker=str(payload["ticker"]),
                        query=str(payload["query"]),
                        as_of_date=str(payload["as_of_date"]),
                        dimension=str(payload.get("dimension", "") or ""),
                        company_name=str(payload.get("company_name", "") or ""),
                        industry=str(payload.get("industry", "") or ""),
                        doc_type=(
                            str(payload["doc_type"])
                            if payload.get("doc_type")
                            else None
                        ),
                        notes=str(payload.get("notes", "") or ""),
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"{source}:{line_no} missing required field: {exc}"
                ) from exc
    return rows


def load_annotation_csv(
    path: str | Path,
) -> dict[str, list[EvidenceJudgment]]:
    grouped: dict[str, list[EvidenceJudgment]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "relevance"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"annotation CSV missing columns: {sorted(missing)}"
            )
        for row in reader:
            raw_relevance = str(row.get("relevance", "") or "").strip()
            if raw_relevance == "":
                continue
            relevance = int(float(raw_relevance))
            if relevance <= 0:
                continue
            raw_page = str(row.get("page", "") or "").strip()
            page = int(raw_page) if raw_page else None
            judgment = EvidenceJudgment(
                case_id=str(row.get("case_id", "") or "").strip(),
                relevance=relevance,
                chunk_id=str(row.get("chunk_id", "") or "").strip(),
                document_key=str(
                    row.get("document_key", "") or ""
                ).strip(),
                page=page,
                notes=str(row.get("notes", "") or "").strip(),
            )
            if not judgment.case_id:
                continue
            if not judgment.chunk_id and not judgment.document_key:
                raise ValueError(
                    "positive judgment requires chunk_id or document_key"
                )
            grouped.setdefault(judgment.case_id, []).append(judgment)
    return grouped


def _hit_page(hit: RetrievalHit) -> int | None:
    raw = (hit.chunk.metadata or {}).get("page")
    try:
        return int(raw) if raw not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _matches(hit: RetrievalHit, judgment: EvidenceJudgment) -> bool:
    if judgment.chunk_id:
        return hit.chunk.chunk_id == judgment.chunk_id
    if judgment.document_key:
        if source_document_key(hit.chunk) != judgment.document_key:
            return False
        if judgment.page is not None and _hit_page(hit) != judgment.page:
            return False
        return True
    return False


def _graded_matches(
    hits: list[RetrievalHit],
    judgments: list[EvidenceJudgment],
    k: int,
) -> tuple[list[int], set[int]]:
    """Greedily match each hit to at most one positive judgment.

    This prevents overlap chunks from double-counting the same annotated evidence
    item. Exact chunk_id judgments take precedence because _matches handles them
    deterministically in annotation order.
    """

    gains: list[int] = []
    used: set[int] = set()
    for hit in hits[: max(0, int(k))]:
        best_idx = None
        best_rel = -1
        for idx, judgment in enumerate(judgments):
            if idx in used or judgment.relevance <= 0:
                continue
            if _matches(hit, judgment) and judgment.relevance > best_rel:
                best_idx = idx
                best_rel = judgment.relevance
        if best_idx is None:
            gains.append(0)
        else:
            used.add(best_idx)
            gains.append(int(judgments[best_idx].relevance))
    return gains, used


def _dcg(gains: Iterable[int]) -> float:
    score = 0.0
    for rank, relevance in enumerate(gains, start=1):
        if relevance <= 0:
            continue
        score += (2 ** int(relevance) - 1) / math.log2(rank + 1)
    return score


def evaluate_retrieval_hits(
    case: RetrievalCase,
    hits: list[RetrievalHit],
    judgments: list[EvidenceJudgment],
    *,
    strategy: str,
    ks: Iterable[int] = (5, 10),
    latency_ms: float = 0.0,
) -> RetrievalMetricRow:
    positive = [item for item in judgments if item.relevance > 0]
    if not positive:
        raise ValueError(
            f"case {case.case_id} has no positive relevance judgments"
        )

    relevant_docs = {
        item.document_key
        for item in positive
        if item.document_key
    }
    cutoff = date.fromisoformat(case.as_of_date[:10])
    pit_violations = sum(
        date.fromisoformat(hit.chunk.publish_date) > cutoff for hit in hits
    )

    metrics: dict[str, float] = {}
    for raw_k in sorted({max(1, int(k)) for k in ks}):
        gains, used = _graded_matches(hits, positive, raw_k)
        matched = len(used)
        first_rank = next(
            (idx for idx, gain in enumerate(gains, start=1) if gain > 0),
            None,
        )

        metrics[f"recall@{raw_k}"] = matched / len(positive)
        metrics[f"precision@{raw_k}"] = matched / raw_k
        metrics[f"hit_rate@{raw_k}"] = 1.0 if matched else 0.0
        metrics[f"mrr@{raw_k}"] = (
            1.0 / first_rank if first_rank is not None else 0.0
        )

        ideal = sorted(
            (item.relevance for item in positive),
            reverse=True,
        )[:raw_k]
        ideal_dcg = _dcg(ideal)
        metrics[f"ndcg@{raw_k}"] = (
            _dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0
        )

        if relevant_docs:
            retrieved_docs = {
                source_document_key(hit.chunk)
                for hit in hits[:raw_k]
            }
            metrics[f"document_recall@{raw_k}"] = (
                len(retrieved_docs & relevant_docs) / len(relevant_docs)
            )

    return RetrievalMetricRow(
        case_id=case.case_id,
        strategy=strategy,
        relevant_evidence=len(positive),
        relevant_documents=len(relevant_docs),
        returned=len(hits),
        latency_ms=float(latency_ms),
        pit_violations=int(pit_violations),
        metrics=metrics,
    )


def annotation_candidate_row(
    case: RetrievalCase,
    hit: RetrievalHit,
    *,
    rank_sources: str,
    excerpt_chars: int = 500,
) -> dict:
    metadata = hit.chunk.metadata or {}
    return {
        "case_id": case.case_id,
        "ticker": case.ticker,
        "company_name": case.company_name,
        "dimension": case.dimension,
        "query": case.query,
        "as_of_date": case.as_of_date,
        "rank_sources": rank_sources,
        "chunk_id": hit.chunk.chunk_id,
        "doc_id": hit.chunk.doc_id,
        "document_key": source_document_key(hit.chunk),
        "file_name": str(metadata.get("file_name", "") or ""),
        "page": str(metadata.get("page", "") or ""),
        "publish_date": hit.chunk.publish_date,
        "source_url": str(
            metadata.get("source_url") or hit.chunk.url or ""
        ),
        "excerpt": hit.chunk.text[: max(80, int(excerpt_chars))].replace(
            "\n",
            " ",
        ),
        "relevance": "",
        "notes": "",
    }
