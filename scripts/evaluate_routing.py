"""Run the routing benchmark without executing finance tools.

Examples:
    python scripts/evaluate_routing.py --system dynamic-supervisor
    python scripts/evaluate_routing.py --system single-agent-all-tools
    python scripts/evaluate_routing.py --system fixed-deep-research

The dynamic-supervisor mode requires the configured LLM credentials. The two
static baselines are dependency-light and useful for validating the eval set.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from tradingagents.conversation import ConversationAgent, ConversationStore
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.evaluation.routing import (
    RoutingRecord,
    load_routing_eval,
    run_routing_eval,
    summarize_routing,
)


def _dynamic_runner(agent: ConversationAgent):
    def run(case):
        action = agent.supervisor.decide(
            case.user_query,
            current_ticker=case.ticker,
            as_of_date=case.as_of_date or "2026-09-05",
            history=[],
        )
        return RoutingRecord(
            action=action.action,
            target=action.target,
            steps=1,
            tool_calls=0,
        )
    return run


def _static_runner(system: str):
    if system == "single-agent-all-tools":
        return lambda _case: RoutingRecord(
            action="call_tool",
            target="auto",
            steps=1,
            tool_calls=0,
        )
    if system == "fixed-deep-research":
        return lambda _case: RoutingRecord(
            action="run_deep_research",
            target="deep_stock_research",
            steps=1,
            tool_calls=0,
        )
    raise ValueError(system)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/datasets/routing_eval_v1.jsonl"),
    )
    parser.add_argument(
        "--system",
        choices=[
            "dynamic-supervisor",
            "single-agent-all-tools",
            "fixed-deep-research",
        ],
        default="dynamic-supervisor",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = load_routing_eval(args.dataset)
    if args.system == "dynamic-supervisor":
        with tempfile.TemporaryDirectory(prefix="tradingagents-routing-eval-") as tmp:
            config = dict(DEFAULT_CONFIG)
            config["conversation_db_path"] = str(Path(tmp) / "conversation.db")
            agent = ConversationAgent(
                config,
                ConversationStore(config["conversation_db_path"]),
            )
            results = run_routing_eval(cases, _dynamic_runner(agent))
    else:
        results = run_routing_eval(cases, _static_runner(args.system))

    payload = {
        "system": args.system,
        "dataset": str(args.dataset),
        "summary": summarize_routing(results),
        "cases": [item.to_dict() for item in results],
        "fairness_note": (
            "Only compare architecture results when model, tool catalog, cutoff, "
            "prompt context and budget are controlled. Static baselines here do "
            "not execute finance tools or measure grounding."
        ),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
