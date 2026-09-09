"""Prepare and run a RAG retrieval benchmark against the local Qdrant corpus.

Two-stage workflow:

1) Prepare a pooled annotation worksheet from several retrieval strategies.
2) Human-label relevance (0-3), then run Recall/MRR/nDCG ablations.

Examples:
    python scripts/run_rag_retrieval_benchmark.py prepare \
      --dataset evaluation/datasets/rag_retrieval_queries_v1.jsonl \
      --output evaluation/data/rag_retrieval_annotations_v1.csv

    python scripts/run_rag_retrieval_benchmark.py run \
      --dataset evaluation/datasets/rag_retrieval_queries_v1.jsonl \
      --annotations evaluation/data/rag_retrieval_annotations_v1.csv \
      --output results/rag_retrieval_benchmark_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.evaluation.rag_retrieval import (
    annotation_candidate_row,
    evaluate_retrieval_hits,
    load_annotation_csv,
    load_retrieval_cases,
)
from tradingagents.rag.retriever import HybridKnowledgeRetriever


STRATEGIES = {
    "dense": ("dense", False),
    "bm25": ("bm25", False),
    "hybrid": ("hybrid", False),
    "hybrid_rerank": ("hybrid", True),
}


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_ks(raw: str) -> list[int]:
    values = sorted({max(1, int(item)) for item in _parse_csv_list(raw)})
    if not values:
        raise ValueError("at least one K is required")
    return values


def _build_retriever() -> HybridKnowledgeRetriever:
    config = dict(DEFAULT_CONFIG)
    return HybridKnowledgeRetriever.from_config(config)


def _strategy_settings(name: str, retriever: HybridKnowledgeRetriever):
    if name not in STRATEGIES:
        raise ValueError(f"unknown retrieval strategy: {name}")
    strategy, use_reranker = STRATEGIES[name]
    if use_reranker and retriever.reranker is None:
        return None
    return strategy, use_reranker


def _search(
    retriever: HybridKnowledgeRetriever,
    case,
    *,
    strategy_name: str,
    top_k: int,
    candidate_k: int,
    corpus_limit: int,
    max_chunks_per_doc: int,
):
    settings = _strategy_settings(strategy_name, retriever)
    if settings is None:
        return None
    strategy, use_reranker = settings
    started = time.perf_counter()
    hits = retriever.search(
        case.query,
        ticker=case.ticker,
        as_of_date=case.as_of_date,
        top_k=top_k,
        candidate_k=candidate_k,
        corpus_limit=corpus_limit,
        doc_type=case.doc_type,
        max_chunks_per_doc=max_chunks_per_doc,
        industry=case.industry or None,
        strategy=strategy,
        use_reranker=use_reranker,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    return hits, latency_ms


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_annotations(args) -> None:
    cases = load_retrieval_cases(args.dataset)
    retriever = _build_retriever()
    requested = _parse_csv_list(args.strategies)
    available = [
        name
        for name in requested
        if _strategy_settings(name, retriever) is not None
    ]
    unavailable = [name for name in requested if name not in available]

    rows: list[dict] = []
    for index, case in enumerate(cases, start=1):
        print(f"[prepare {index}/{len(cases)}] {case.case_id}: {case.query}")
        pooled: dict[str, dict] = {}
        for strategy_name in available:
            result = _search(
                retriever,
                case,
                strategy_name=strategy_name,
                top_k=args.annotation_top_k,
                candidate_k=args.candidate_k,
                corpus_limit=args.corpus_limit,
                max_chunks_per_doc=args.max_chunks_per_doc,
            )
            if result is None:
                continue
            hits, _ = result
            for rank, hit in enumerate(hits, start=1):
                item = pooled.setdefault(
                    hit.chunk.chunk_id,
                    {
                        "hit": hit,
                        "ranks": {},
                    },
                )
                item["ranks"][strategy_name] = rank

        ordered = sorted(
            pooled.values(),
            key=lambda item: (
                min(item["ranks"].values()),
                sum(item["ranks"].values()),
                item["hit"].chunk.chunk_id,
            ),
        )
        for item in ordered:
            rank_sources = "|".join(
                f"{name}:{rank}"
                for name, rank in sorted(item["ranks"].items())
            )
            rows.append(
                annotation_candidate_row(
                    case,
                    item["hit"],
                    rank_sources=rank_sources,
                    excerpt_chars=args.excerpt_chars,
                )
            )

    _write_csv(args.output, rows)
    print(f"annotation worksheet: {len(rows)} rows -> {args.output}")
    if unavailable:
        print(
            "skipped unavailable strategies: "
            + ", ".join(unavailable)
            + " (reranker model may be unavailable)"
        )
    print(
        "Fill relevance with 0=not relevant, 1=relevant, 2=highly relevant, "
        "3=direct answer evidence. Blank rows are treated as unjudged."
    )


def _summary_rows(per_case_rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in per_case_rows:
        groups[str(row["strategy"])].append(row)

    summary: list[dict] = []
    for strategy, rows in groups.items():
        metric_keys = [
            key
            for key in rows[0]
            if (
                key.startswith("recall@")
                or key.startswith("precision@")
                or key.startswith("hit_rate@")
                or key.startswith("mrr@")
                or key.startswith("ndcg@")
                or key.startswith("document_recall@")
            )
        ]
        total_returned = sum(int(row["returned"]) for row in rows)
        total_pit = sum(int(row["pit_violations"]) for row in rows)
        item = {
            "strategy": strategy,
            "annotated_cases": len(rows),
            "avg_latency_ms": mean(float(row["latency_ms"]) for row in rows),
            "pit_violation_rate": (
                total_pit / total_returned if total_returned else 0.0
            ),
        }
        for key in metric_keys:
            item[key] = mean(float(row[key]) for row in rows)
        summary.append(item)
    return summary


def _write_report(
    path: Path,
    *,
    dataset: Path,
    annotations: Path,
    summary: list[dict],
    skipped_cases: list[str],
    unavailable: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RAG Retrieval Benchmark Report",
        "",
        f"- Dataset: {dataset}",
        f"- Annotations: {annotations}",
        f"- Evaluated strategies: {', '.join(row['strategy'] for row in summary) or 'none'}",
        f"- Cases without positive judgments: {len(skipped_cases)}",
        "",
        "## Summary",
        "",
    ]
    if summary:
        columns = list(summary[0].keys())
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in summary:
            values = []
            for column in columns:
                value = row.get(column)
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
    else:
        lines.append(
            "No metrics were produced because the annotation file contains "
            "no positive judgments."
        )

    lines += [
        "",
        "## Metric semantics",
        "",
        "- Recall@K: fraction of annotated relevant evidence units retrieved in top K.",
        "- DocumentRecall@K: fraction of relevant parent documents represented in top K.",
        "- MRR@K: reciprocal rank of the first relevant evidence hit.",
        "- nDCG@K: graded ranking quality using relevance 1/2/3.",
        "- PIT violation rate: returned chunks published after the case as_of_date.",
        "",
        "## Interpretation guardrails",
        "",
        "- Results are only meaningful for manually judged cases.",
        "- Do not report a Recall number from blank/unreviewed annotation rows.",
        "- The pooled worksheet reduces, but does not eliminate, annotation-pool bias.",
        "- Keep the same corpus, cutoff, candidate_k, corpus_limit and top-K budget "
        "when comparing Dense, BM25, RRF and Reranker.",
        "- A higher retrieval metric does not by itself prove better investment "
        "conclusions; retrieval quality and downstream answer grounding are separate.",
    ]
    if unavailable:
        lines += [
            "",
            "## Unavailable strategies",
            "",
            "- " + "\n- ".join(unavailable),
        ]
    if skipped_cases:
        lines += [
            "",
            "## Unjudged / skipped cases",
            "",
            "- " + "\n- ".join(skipped_cases),
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(args) -> None:
    cases = load_retrieval_cases(args.dataset)
    judgments = load_annotation_csv(args.annotations)
    ks = _parse_ks(args.ks)
    retriever = _build_retriever()
    requested = _parse_csv_list(args.strategies)
    available = [
        name
        for name in requested
        if _strategy_settings(name, retriever) is not None
    ]
    unavailable = [name for name in requested if name not in available]

    per_case_rows: list[dict] = []
    raw_runs: list[dict] = []
    skipped_cases: list[str] = []
    max_k = max(ks)

    for index, case in enumerate(cases, start=1):
        positive = judgments.get(case.case_id, [])
        if not positive:
            skipped_cases.append(case.case_id)
            continue

        print(f"[run {index}/{len(cases)}] {case.case_id}: {case.query}")
        for strategy_name in available:
            result = _search(
                retriever,
                case,
                strategy_name=strategy_name,
                top_k=max_k,
                candidate_k=args.candidate_k,
                corpus_limit=args.corpus_limit,
                max_chunks_per_doc=args.max_chunks_per_doc,
            )
            if result is None:
                continue
            hits, latency_ms = result
            metrics = evaluate_retrieval_hits(
                case,
                hits,
                positive,
                strategy=strategy_name,
                ks=ks,
                latency_ms=latency_ms,
            )
            per_case_rows.append(metrics.to_dict())
            raw_runs.append(
                {
                    "case_id": case.case_id,
                    "strategy": strategy_name,
                    "ticker": case.ticker,
                    "query": case.query,
                    "as_of_date": case.as_of_date,
                    "latency_ms": latency_ms,
                    "hits": [
                        {
                            "rank": rank,
                            "chunk_id": hit.chunk.chunk_id,
                            "doc_id": hit.chunk.doc_id,
                            "document_key": (
                                hit.chunk.metadata or {}
                            ).get("file_hash", hit.chunk.doc_id),
                            "page": (
                                hit.chunk.metadata or {}
                            ).get("page"),
                            "publish_date": hit.chunk.publish_date,
                            "title": hit.chunk.title,
                            "score": hit.score,
                            "dense_score": hit.dense_score,
                            "bm25_score": hit.bm25_score,
                            "rerank_score": hit.rerank_score,
                        }
                        for rank, hit in enumerate(hits, start=1)
                    ],
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "per_case.csv", per_case_rows)
    summary = _summary_rows(per_case_rows)
    _write_csv(args.output / "summary.csv", summary)

    with (args.output / "raw_runs.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in raw_runs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    _write_report(
        args.output / "RAG_RETRIEVAL_REPORT.md",
        dataset=args.dataset,
        annotations=args.annotations,
        summary=summary,
        skipped_cases=skipped_cases,
        unavailable=unavailable,
    )
    print(
        f"benchmark complete: {len(per_case_rows)} evaluated runs -> {args.output}"
    )
    if skipped_cases:
        print(
            f"skipped {len(skipped_cases)} cases without positive judgments"
        )


def _add_common(parser) -> None:
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "evaluation/datasets/rag_retrieval_queries_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--strategies",
        default="dense,bm25,hybrid,hybrid_rerank",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=int(DEFAULT_CONFIG.get("rag_candidate_k", 30)),
    )
    parser.add_argument(
        "--corpus-limit",
        type=int,
        default=int(DEFAULT_CONFIG.get("rag_bm25_corpus_limit", 1000)),
    )
    parser.add_argument(
        "--max-chunks-per-doc",
        type=int,
        default=int(DEFAULT_CONFIG.get("rag_max_chunks_per_doc", 2)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser(
        "prepare",
        help="pool retrieval candidates into a human-annotation CSV",
    )
    _add_common(prepare)
    prepare.add_argument(
        "--output",
        type=Path,
        default=Path(
            "evaluation/data/rag_retrieval_annotations_v1.csv"
        ),
    )
    prepare.add_argument("--annotation-top-k", type=int, default=12)
    prepare.add_argument("--excerpt-chars", type=int, default=500)

    run = sub.add_parser(
        "run",
        help="evaluate manually annotated retrieval cases",
    )
    _add_common(run)
    run.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "evaluation/data/rag_retrieval_annotations_v1.csv"
        ),
    )
    run.add_argument(
        "--output",
        type=Path,
        default=Path("results/rag_retrieval_benchmark_v1"),
    )
    run.add_argument("--ks", default="5,10")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare_annotations(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
