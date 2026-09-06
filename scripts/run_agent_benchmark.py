"""Run end-to-end architecture benchmark across three system variants.

Example:
    python scripts/run_agent_benchmark.py \
      --systems single-agent-all-tools,fixed-deep-research,dynamic-supervisor \
      --dataset evaluation/datasets/architecture_benchmark_v1.jsonl \
      --output results/benchmark_v1

Optional pricing JSON:
{
  "gpt-model-name": {
    "input_per_million": 1.0,
    "output_per_million": 4.0
  }
}
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.evaluation.architecture import (
    ArchitectureEvalResult,
    ArchitectureSystemRun,
    evaluate_architecture_run,
    load_architecture_benchmark,
)
from tradingagents.evaluation.systems import SYSTEM_BUILDERS
from tradingagents.evaluation.telemetry import estimate_cost
from tradingagents.llm_clients import create_llm_client
from tradingagents.orchestration.completion import CompletionGate


def _completion_assessor(config):
    client = create_llm_client(
        provider=config["llm_provider"],
        model=config["quick_think_llm"],
        base_url=config.get("backend_url"),
    )
    gate = CompletionGate(client.get_llm())

    def assess(contract, observations):
        return gate.assess(
            contract,
            observations=observations,
            used_capabilities=[],
        )

    return assess


def _load_pricing(path: Path | None):
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pricing JSON must be an object")
    return payload


def _summary_rows(results: list[ArchitectureEvalResult]):
    groups = defaultdict(list)
    for item in results:
        groups[item.system_name].append(item)

    rows = []
    for system, items in groups.items():
        n = max(1, len(items))
        checked_pit = sum(item.checked_pit_parameters for item in items)
        pit_violations = sum(item.pit_parameter_violations for item in items)
        costs = [
            item.estimated_cost
            for item in items
            if item.estimated_cost is not None
        ]
        rows.append(
            {
                "system": system,
                "cases": len(items),
                "success_rate": sum(not item.error for item in items) / n,
                "route_accuracy": sum(item.route_pass for item in items) / n,
                "claim_grounding_rate": mean(
                    item.claim_grounding_rate for item in items
                ),
                "unsupported_claim_rate": mean(
                    item.unsupported_claim_rate for item in items
                ),
                "evidence_coverage_rate": mean(
                    item.evidence_coverage_rate for item in items
                ),
                "completion_ratio": mean(
                    item.completion_ratio for item in items
                ),
                "complete_case_rate": sum(item.complete for item in items) / n,
                "pit_violation_rate": (
                    pit_violations / checked_pit if checked_pit else 0.0
                ),
                "future_date_mention_rate": (
                    sum(item.future_date_mentions > 0 for item in items) / n
                ),
                "avg_llm_calls": mean(item.llm_calls for item in items),
                "avg_tool_calls": mean(item.tool_calls for item in items),
                "avg_tokens": mean(
                    item.tokens_in + item.tokens_out for item in items
                ),
                "avg_latency_ms": mean(item.latency_ms for item in items),
                "estimated_cost_total": (
                    sum(costs) if costs else None
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_report(path: Path, summary_rows: list[dict], dataset: str):
    lines = [
        "# Architecture Benchmark Report",
        "",
        f"- Dataset: {dataset}",
        "- Fairness requirement: same model configuration, same tool catalog, "
        "same cutoff and comparable budgets.",
        "- Completion judge calls are evaluation overhead and are not included "
        "in each system telemetry.",
        "- Token counts are provider-reported lower bounds; missing provider usage "
        "metadata is never estimated silently.",
        "",
        "## Summary",
        "",
    ]
    columns = [
        "system",
        "route_accuracy",
        "claim_grounding_rate",
        "unsupported_claim_rate",
        "completion_ratio",
        "pit_violation_rate",
        "avg_llm_calls",
        "avg_tool_calls",
        "avg_tokens",
        "avg_latency_ms",
        "estimated_cost_total",
    ]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(_fmt(row.get(column)) for column in columns)
            + " |"
        )

    lines += [
        "",
        "## Interpretation Guardrails",
        "",
        "- Do not claim the Supervisor is better unless measured results support it.",
        "- Route accuracy rewards minimal capability selection; the fixed graph is "
        "expected to be inefficient on simple tasks by design.",
        "- Grounding is a deterministic claim heuristic, not a perfect semantic judge.",
        "- Style-weight performance is a separate walk-forward experiment and is "
        "not inferred from this architecture benchmark.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "evaluation/datasets/architecture_benchmark_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--systems",
        default=(
            "single-agent-all-tools,"
            "fixed-deep-research,"
            "dynamic-supervisor"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark_v1"),
    )
    parser.add_argument("--pricing-json", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    requested = [
        item.strip()
        for item in args.systems.split(",")
        if item.strip()
    ]
    unknown = [name for name in requested if name not in SYSTEM_BUILDERS]
    if unknown:
        raise ValueError(f"unknown systems: {unknown}")

    cases = load_architecture_benchmark(args.dataset)
    if args.limit is not None:
        cases = cases[: max(0, int(args.limit))]

    pricing = _load_pricing(args.pricing_json)
    config = dict(DEFAULT_CONFIG)
    completion_assessor = _completion_assessor(config)

    raw_runs: list[ArchitectureSystemRun] = []
    evaluated: list[ArchitectureEvalResult] = []

    for system_name in requested:
        system = SYSTEM_BUILDERS[system_name](config)
        for case in cases:
            print(f"[{system_name}] {case.case_id}: {case.user_query}")
            run = system.run(case)
            raw_runs.append(run)
            cost = estimate_cost(run.telemetry, pricing)
            evaluated.append(
                evaluate_architecture_run(
                    case,
                    run,
                    completion_assessor=completion_assessor,
                    estimated_cost=cost,
                )
            )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "raw_runs.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for run in raw_runs:
            handle.write(
                json.dumps(run.to_dict(), ensure_ascii=False) + "\n"
            )

    per_case_rows = [item.to_dict() for item in evaluated]
    _write_csv(args.output / "per_case.csv", per_case_rows)

    summary = _summary_rows(evaluated)
    _write_csv(args.output / "summary.csv", summary)
    _write_report(
        args.output / "BENCHMARK_REPORT.md",
        summary,
        str(args.dataset),
    )

    print(
        f"benchmark complete: {len(evaluated)} runs -> {args.output}"
    )


if __name__ == "__main__":
    main()
