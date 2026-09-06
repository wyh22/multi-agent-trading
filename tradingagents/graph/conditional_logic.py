"""LangGraph 条件路由逻辑。"""

from tradingagents.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """集中管理分析师工具循环和最终审计/修复回路。"""

    def __init__(self, max_audit_rounds: int = 2):
        self.max_audit_rounds = max(1, int(max_audit_rounds))

    @staticmethod
    def _route_tool_loop(state: AgentState, tool_node: str) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        if getattr(last_message, "tool_calls", None):
            return tool_node
        return "完成"

    def should_continue_market(self, state: AgentState) -> str:
        return self._route_tool_loop(state, "tools_market")

    def should_continue_news(self, state: AgentState) -> str:
        return self._route_tool_loop(state, "tools_news")

    def should_continue_fundamentals(self, state: AgentState) -> str:
        return self._route_tool_loop(state, "tools_fundamentals")

    def route_after_audit(self, state: AgentState) -> str:
        """PASS 结束；REVISE 定向到责任 Agent 或 PM，且受轮次上限约束。"""

        status = str(state.get("audit_status", "PASS")).upper()
        rounds = int(state.get("audit_round", 0) or 0)
        if status != "REVISE" or rounds >= self.max_audit_rounds:
            return "结束"

        target = str(
            state.get("audit_repair_target", "portfolio_manager")
            or "portfolio_manager"
        ).lower()
        mapping = {
            "market": "修订_market",
            "news": "修订_news",
            "fundamentals": "修订_fundamentals",
            "portfolio_manager": "修订_pm",
            "rag": "修订_news",
        }
        return mapping.get(target, "修订_pm")
