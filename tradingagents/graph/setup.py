"""最终版单股深度研究 LangGraph 拓扑。"""

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

from .analyst_execution import build_analyst_execution_plan
from .conditional_logic import ConditionalLogic


class GraphSetup:
    """负责组装精简后的单股多智能体工作流。"""

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

        # 三类分析师使用私有子图并行执行，只把报告和工具轨迹写回父状态。
        for spec in plan.specs:
            runner = build_analyst_runner(
                analyst_key=spec.key,
                agent_node_name=spec.agent_node,
                tool_node_name=spec.tool_node,
                report_key=spec.report_key,
                analyst_factory=analyst_factories[spec.key],
                tool_node=self.tool_nodes[spec.key],
                conditional_router=getattr(
                    self.conditional_logic,
                    f"should_continue_{spec.key}",
                ),
            )
            workflow.add_node(spec.agent_node, runner)

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

        # 第一阶段：分析师并行 Fan-out / Fan-in。
        for spec in plan.specs:
            workflow.add_edge(START, spec.agent_node)
        workflow.add_edge(
            [spec.agent_node for spec in plan.specs],
            "_AnalystFanIn",
        )

        # 第二阶段：Bull/Bear 并行生成互补假设，不再多轮互相复述。
        workflow.add_edge("_AnalystFanIn", "Bull Researcher")
        workflow.add_edge("_AnalystFanIn", "Bear Researcher")
        workflow.add_edge(
            ["Bull Researcher", "Bear Researcher"],
            "_ResearchFanIn",
        )

        # 第三阶段：投资组合经理收敛结论，再由审计智能体检查。
        workflow.add_edge("_ResearchFanIn", "Portfolio Manager")
        workflow.add_edge("Portfolio Manager", "Decision Auditor")
        workflow.add_conditional_edges(
            "Decision Auditor",
            self.conditional_logic.route_after_audit,
            {
                "修订": "Portfolio Manager",
                "结束": END,
            },
        )
        return workflow
