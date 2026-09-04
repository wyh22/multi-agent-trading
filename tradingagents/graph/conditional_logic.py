"""LangGraph 条件路由逻辑。"""

from tradingagents.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """集中管理分析师工具循环和最终审计回路。"""

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
        """市场分析师有工具调用时继续工具循环。"""
        return self._route_tool_loop(state, "tools_market")

    def should_continue_news(self, state: AgentState) -> str:
        """新闻与情绪分析师有工具调用时继续工具循环。"""
        return self._route_tool_loop(state, "tools_news")

    def should_continue_fundamentals(self, state: AgentState) -> str:
        """基本面分析师有工具调用时继续工具循环。"""
        return self._route_tool_loop(state, "tools_fundamentals")

    def route_after_audit(self, state: AgentState) -> str:
        """审计通过则结束；首次失败允许投资组合经理修订一次。"""
        status = str(state.get("audit_status", "PASS")).upper()
        rounds = int(state.get("audit_round", 0) or 0)
        if status == "REVISE" and rounds < self.max_audit_rounds:
            return "修订"
        return "结束"
