from __future__ import annotations

from typing import Any

from langgraph.prebuilt import ToolNode

from tradingagents.agents import (
    create_fundamentals_analyst,
    create_market_analyst,
    create_news_analyst,
)
from tradingagents.agents.utils.agent_utils import build_instrument_context, resolve_instrument_identity
from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS
from tradingagents.graph.analyst_subgraph import build_analyst_runner
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.orchestration.schemas import ExecutionResult


_FACTORIES = {
    "market": create_market_analyst,
    "news": create_news_analyst,
    "fundamentals": create_fundamentals_analyst,
}


class SpecialistAgentExecutor:
    """Execute an existing analyst private subgraph without running the full research graph."""

    def __init__(self, llm: Any, tool_groups: dict[str, list], *, max_recur_limit: int = 60):
        self.llm = llm
        self.tool_groups = tool_groups
        self.max_recur_limit = max(4, int(max_recur_limit))
        self.logic = ConditionalLogic(max_audit_rounds=2)
        self._runners: dict[str, Any] = {}

    def _runner(self, key: str):
        if key in self._runners:
            return self._runners[key]
        if key not in _FACTORIES:
            raise ValueError(f"unknown specialist agent: {key}")
        spec = ANALYST_NODE_SPECS[key]
        tools = self.tool_groups.get(key, [])
        tool_node = ToolNode(tools, handle_tool_errors=True)
        factory = _FACTORIES[key]
        runner = build_analyst_runner(
            analyst_key=spec.key,
            agent_node_name=spec.agent_node,
            tool_node_name=spec.tool_node,
            report_key=spec.report_key,
            analyst_factory=lambda: factory(self.llm, tools),
            tool_node=tool_node,
            conditional_router=getattr(self.logic, f"should_continue_{key}"),
        )
        self._runners[key] = runner
        return runner

    def run(
        self,
        key: str,
        *,
        ticker: str,
        as_of_date: str,
        objective: str = "",
        candidate_context: str = "",
        past_context: str = "",
    ) -> ExecutionResult:
        try:
            identity = resolve_instrument_identity(ticker)
            instrument_context = build_instrument_context(ticker, "stock", identity)
            state = {
                "company_of_interest": ticker,
                "asset_type": "stock",
                "instrument_context": instrument_context,
                "candidate_context": candidate_context,
                "trade_date": as_of_date,
                "past_context": past_context,
                "audit_feedback": objective,
            }
            result = self._runner(key)(
                state,
                config={"recursion_limit": self.max_recur_limit},
            )
            report_key = ANALYST_NODE_SPECS[key].report_key
            content = str(result.get(report_key, "") or "").strip()
            if not content:
                return ExecutionResult(
                    status="NO_DATA",
                    capability=f"{key}_agent",
                    content="专业 Agent 未形成可用报告。",
                    data={"trace": result.get("analyst_trace", [])},
                )
            return ExecutionResult(
                capability=f"{key}_agent",
                content=content,
                data={"trace": result.get("analyst_trace", [])},
            )
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                status="FAILED",
                capability=f"{key}_agent",
                content=f"{type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
                retryable=True,
            )
