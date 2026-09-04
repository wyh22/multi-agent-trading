"""候选发现协调智能体。

确定性的市场、行业和候选计算仍由 Python 工具负责；协调智能体只负责根据
研究预算决定调用哪些工具以及把哪些候选交给后续深度研究。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from .market import analyze_market_regime
from .pipeline import run_discovery
from .sectors import analyze_sectors


def _records(df, columns: list[str], limit: int) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    existing = [column for column in columns if column in df.columns]
    return df[existing].head(limit).where(df[existing].notna(), None).to_dict("records")


@tool
def analyze_market_tool(as_of_date: str) -> str:
    """分析 A 股核心指数并返回市场环境。"""
    result = analyze_market_regime(as_of_date)
    payload = {
        "as_of_date": as_of_date,
        "regime": result.regime,
        "score": round(result.score, 2),
        "summary": result.summary,
        "indices": _records(
            result.indices,
            ["index_name", "ret_20d", "ret_60d", "vol_20d", "index_score"],
            5,
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def rank_sector_tool(as_of_date: str, top_n: int = 6) -> str:
    """按截止日分析申万一级行业并返回前 N 个行业。"""
    market = analyze_market_regime(as_of_date)
    result = analyze_sectors(as_of_date, market_regime=market.regime, top_n=top_n)
    payload = {
        "market_regime": market.regime,
        "top_sectors": _records(
            result.sectors,
            [
                "sector_code",
                "sector_name",
                "ret_20d",
                "ret_60d",
                "trend_quality",
                "sector_score",
            ],
            top_n,
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def discover_candidates_tool(
    as_of_date: str,
    sector_count: int = 4,
    per_sector: int = 35,
    top_n: int = 10,
) -> str:
    """执行确定性的市场->行业->量化->Quality 二筛并返回研究候选。"""
    result = run_discovery(
        as_of_date,
        sector_count=sector_count,
        per_sector=per_sector,
        top_n=top_n,
    )
    payload = {
        "market_regime": result.market.regime,
        "market_score": round(result.market.score, 2),
        "sector_distribution": result.stocks.sector_quotas,
        "candidates": _records(
            result.stocks.candidates,
            [
                "ticker",
                "name",
                "sector_name",
                "final_score",
                "quant_score",
                "quality_score",
                "quality_flag",
                "research_label",
            ],
            top_n,
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


@dataclass
class DiscoveryCoordinatorResult:
    final_message: str
    tool_trace: list[dict[str, Any]] = field(default_factory=list)


class DiscoveryCoordinatorAgent:
    """通过 LLM Tool Calling 编排候选发现工具。"""

    def __init__(self, llm, max_steps: int = 5):
        self.llm = llm
        self.max_steps = max(2, int(max_steps))
        self.tools = [
            analyze_market_tool,
            rank_sector_tool,
            discover_candidates_tool,
        ]
        self.tool_map = {item.name: item for item in self.tools}
        self.bound_llm = llm.bind_tools(self.tools)

    def run(
        self,
        as_of_date: str,
        *,
        max_deep_research: int = 3,
        research_budget: str = "low",
    ) -> DiscoveryCoordinatorResult:
        system_text = f"""
你是 A 股候选发现协调智能体。当前日期截止为 {as_of_date}。
所有市场、行业和候选结论都必须来自工具结果。

目标：
1. 先理解市场环境；
2. 必要时查看行业排名；
3. 根据研究预算调用候选发现工具；
4. 最终只选择最多 {max_deep_research} 个候选进入后续深度研究。

研究预算：{research_budget}。
预算低时优先减少候选数量，不要重复调用同一工具。
最终回答说明候选代码、入选依据以及下游需要继续核验的风险。
""".strip()

        messages = [HumanMessage(content=system_text)]
        trace: list[dict[str, Any]] = []

        for _ in range(self.max_steps):
            response = self.bound_llm.invoke(messages)
            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                return DiscoveryCoordinatorResult(
                    final_message=str(response.content).strip(),
                    tool_trace=trace,
                )

            for call in tool_calls:
                name = call.get("name", "")
                arguments = call.get("args", {}) or {}
                tool_obj = self.tool_map.get(name)
                result_text = f"未知工具：{name}" if tool_obj is None else str(tool_obj.invoke(arguments))
                trace.append({"tool_name": name, "arguments": arguments})
                messages.append(
                    ToolMessage(
                        content=result_text,
                        tool_call_id=call.get("id", name),
                    )
                )

        final = self.llm.invoke(
            messages + [HumanMessage(content="工具调用步数已达到上限，请基于现有结果直接给出最终候选。")]
        )
        return DiscoveryCoordinatorResult(
            final_message=str(final.content).strip(),
            tool_trace=trace,
        )
