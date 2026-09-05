"""并行分析师私有子图。

每个分析师拥有独立 ``messages`` 通道，父图只接收最终报告和紧凑工具轨迹，
从而避免并行执行时不同分析师的 ToolCall 消息相互污染。
"""

from dataclasses import dataclass
from typing import Annotated, Any, Callable

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


class AnalystSubState(MessagesState):
    company_of_interest: Annotated[str, "标的代码"]
    asset_type: Annotated[str, "资产类型"]
    instrument_context: Annotated[str, "标的身份信息"]
    candidate_context: Annotated[str, "行业发现/代表股研究来源；selection prior only"]
    trade_date: Annotated[str, "研究截止日"]
    past_context: Annotated[str, "历史复盘摘要"]
    sender: Annotated[str, "发送方"]
    market_report: Annotated[str, "市场报告"]
    news_report: Annotated[str, "新闻与情绪报告"]
    fundamentals_report: Annotated[str, "基本面报告"]
    direct_tool_trace: list[dict[str, Any]]


@dataclass(frozen=True)
class AnalystSubgraphResult:
    report_key: str
    report_value: str
    tool_trace: list[dict[str, Any]]


def build_analyst_subgraph(
    *,
    analyst_key: str,
    agent_node_name: str,
    tool_node_name: str,
    report_key: str,
    analyst_factory: Callable[[], Any],
    tool_node: ToolNode,
    conditional_router: Callable[[AnalystSubState], str],
):
    """构建并编译一个分析师的私有 Agent-Tool 循环。"""

    workflow = StateGraph(AnalystSubState)
    workflow.add_node(agent_node_name, analyst_factory())
    workflow.add_node(tool_node_name, tool_node)
    workflow.add_edge(START, agent_node_name)

    def _route(state: AnalystSubState):
        destination = conditional_router(state)
        return tool_node_name if destination == tool_node_name else END

    workflow.add_conditional_edges(agent_node_name, _route, [tool_node_name, END])
    workflow.add_edge(tool_node_name, agent_node_name)
    return workflow.compile()


def _extract_tool_calls(update: dict[str, Any], analyst_key: str) -> list[dict[str, Any]]:
    """从子图增量中提取紧凑工具调用轨迹。"""

    events: list[dict[str, Any]] = []
    for message in update.get("messages", []) or []:
        for call in getattr(message, "tool_calls", None) or []:
            events.append(
                {
                    "analyst": analyst_key,
                    "tool_name": call.get("name", ""),
                    "arguments": call.get("args", {}) or {},
                }
            )

    for call in update.get("direct_tool_trace", []) or []:
        if not isinstance(call, dict):
            continue
        events.append(
            {
                "analyst": analyst_key,
                "tool_name": str(call.get("tool_name", "")),
                "arguments": dict(call.get("arguments", {}) or {}),
            }
        )
    return events


def build_analyst_runner(
    *,
    analyst_key: str,
    agent_node_name: str,
    tool_node_name: str,
    report_key: str,
    analyst_factory: Callable[[], Any],
    tool_node: ToolNode,
    conditional_router: Callable[[AnalystSubState], str],
):
    """返回可挂到父图上的分析师包装节点。"""

    child = build_analyst_subgraph(
        analyst_key=analyst_key,
        agent_node_name=agent_node_name,
        tool_node_name=tool_node_name,
        report_key=report_key,
        analyst_factory=analyst_factory,
        tool_node=tool_node,
        conditional_router=conditional_router,
    )

    def run(
        parent_state: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        ticker = str(parent_state["company_of_interest"])
        trade_date = str(parent_state["trade_date"])
        child_state: dict[str, Any] = {
            "messages": [HumanMessage(content=f"分析 {ticker}，截止日期 {trade_date}。")],
            "company_of_interest": ticker,
            "asset_type": parent_state.get("asset_type", "stock"),
            "instrument_context": parent_state.get("instrument_context", ""),
            "candidate_context": parent_state.get("candidate_context", ""),
            "trade_date": trade_date,
            "past_context": parent_state.get("past_context", ""),
            "sender": "",
            "market_report": "",
            "news_report": "",
            "fundamentals_report": "",
            "direct_tool_trace": [],
        }

        report_value = ""
        tool_trace: list[dict[str, Any]] = []
        for chunk in child.stream(child_state, config=config, stream_mode="updates"):
            if not isinstance(chunk, dict):
                continue
            for update in chunk.values():
                if not isinstance(update, dict):
                    continue
                if update.get(report_key):
                    report_value = str(update[report_key])
                tool_trace.extend(_extract_tool_calls(update, analyst_key))

        return {report_key: report_value, "analyst_trace": tool_trace}

    run.__name__ = f"run_{analyst_key}_subgraph"
    return run
