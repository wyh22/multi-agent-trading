"""候选发现协调智能体演示入口。"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.discovery.coordinator_agent import DiscoveryCoordinatorAgent
from tradingagents.llm_clients import create_llm_client


def main() -> int:
    parser = argparse.ArgumentParser(description="A股候选发现协调智能体演示")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--deep-research", type=int, default=3)
    parser.add_argument("--budget", choices=["low", "medium", "high"], default="low")
    args = parser.parse_args()

    client = create_llm_client(
        provider=DEFAULT_CONFIG["llm_provider"],
        model=DEFAULT_CONFIG["quick_think_llm"],
        base_url=DEFAULT_CONFIG.get("backend_url"),
    )
    agent = DiscoveryCoordinatorAgent(client.get_llm())
    result = agent.run(
        args.date,
        max_deep_research=args.deep_research,
        research_budget=args.budget,
    )

    print("\n=== 候选发现协调智能体 ===")
    print(result.final_message)
    print("\n工具轨迹:")
    for index, item in enumerate(result.tool_trace, start=1):
        print(f"{index}. {item['tool_name']} {item['arguments']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
