"""单股深度研究 LangGraph 拓扑，支持审计驱动的定向重新取证。"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import (
    create_bear_researcher,
    create_bull_researcher,
    create_decision_auditor,
    create_fundamentals_analyst,
    create_market_analyst,
    create_news_analyst,
    create_portfolio_manager,
)
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.graph.analyst_subgraph import build_analyst_runner

from .analyst_execution import ANALYST_NODE_SPECS, build_analyst_execution_plan
from .conditional_logic import ConditionalLogic


class GraphSetup:
    """组装单股研究图；首轮可选 Analyst，返修可按审计结果定向补证据。"""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        tool_groups: dict[str, list] | None = None,
    ):
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.tool_groups = tool_groups or {}

    def setup_graph(
        self,
        selected_analysts=("market", "news", "fundamentals"),
    ):
        """构建图但不立即编译，便于外层按需绑定 Checkpointer。"""

        plan = build_analyst_execution_plan(selected_analysts)
        analyst_factories = {
            "market": lambda: create_market_analyst(
                self.quick_thinking_llm, self.tool_groups.get("market")
            ),
            "news": lambda: create_news_analyst(
                self.quick_thinking_llm, self.tool_groups.get("news")
            ),
            "fundamentals": lambda: create_fundamentals_analyst(
                self.quick_thinking_llm, self.tool_groups.get("fundamentals")
            ),
        }

        workflow = StateGraph(AgentState)
        analyst_runners = {}

        # Build all three specialist runners once. Only selected analysts run in
        # the initial fan-out; any specialist may later be invoked as a repair
        # node if the auditor identifies a missing evidence domain.
        for key, spec in ANALYST_NODE_SPECS.items():
            runner = build_analyst_runner(
                analyst_key=spec.key,
                agent_node_name=spec.agent_node,
                tool_node_name=spec.tool_node,
                report_key=spec.report_key,
                analyst_factory=analyst_factories[key],
                tool_node=self.tool_nodes[key],
                conditional_router=getattr(
                    self.conditional_logic,
                    f"should_continue_{key}",
                ),
            )
            analyst_runners[key] = runner

        for spec in plan.specs:
            workflow.add_node(spec.agent_node, analyst_runners[spec.key])

        repair_nodes = {
            "market": "_RepairMarket",
            "news": "_RepairNews",
            "fundamentals": "_RepairFundamentals",
        }
        for key, node_name in repair_nodes.items():
            workflow.add_node(node_name, analyst_runners[key])

        workflow.add_node(
            "Bull Researcher",
            create_bull_researcher(self.quick_thinking_llm),
        )
        workflow.add_node(
            "Bear Researcher",
            create_bear_researcher(self.quick_thinking_llm),
        )
        workflow.add_node(
            "Portfolio Manager",
            create_portfolio_manager(self.deep_thinking_llm),
        )
        workflow.add_node(
            "Decision Auditor",
            create_decision_auditor(self.quick_thinking_llm),
        )
        workflow.add_node("_AnalystFanIn", lambda state: {})
        workflow.add_node("_ResearchFanIn", lambda state: {})

        # Initial specialist research: parallel fan-out/fan-in.
        for spec in plan.specs:
            workflow.add_edge(START, spec.agent_node)
        workflow.add_edge(
            [spec.agent_node for spec in plan.specs],
            "_AnalystFanIn",
        )

        # Bull/Bear are still parallel and bounded; no unconditional debate loop.
        workflow.add_edge("_AnalystFanIn", "Bull Researcher")
        workflow.add_edge("_AnalystFanIn", "Bear Researcher")
        workflow.add_edge(
            ["Bull Researcher", "Bear Researcher"],
            "_ResearchFanIn",
        )

        workflow.add_edge("_ResearchFanIn", "Portfolio Manager")
        workflow.add_edge("Portfolio Manager", "Decision Auditor")

        # A specialist repair refreshes only the evidence domain that failed,
        # then returns to PM for re-synthesis and another audit.
        for node_name in repair_nodes.values():
            workflow.add_edge(node_name, "Portfolio Manager")

        workflow.add_conditional_edges(
            "Decision Auditor",
            self.conditional_logic.route_after_audit,
            {
                "修订_market": repair_nodes["market"],
                "修订_news": repair_nodes["news"],
                "修订_fundamentals": repair_nodes["fundamentals"],
                "修订_pm": "Portfolio Manager",
                "结束": END,
            },
        )
        return workflow
