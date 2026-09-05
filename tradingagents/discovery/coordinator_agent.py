"""行业发现协调智能体。

市场与行业排序仍由确定性 Python / 可选量化 Ranker 负责；LLM 只负责决定
调用哪些只读工具，并把行业研究优先级解释成后续深度研究计划。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from .market import analyze_market_regime
from .pipeline import run_discovery, run_research_pool
from .sectors import analyze_sectors


def _records(df, columns: list[str], limit: int) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    existing = [column for column in columns if column in df.columns]
    return (
        df[existing]
        .head(limit)
        .where(df[existing].notna(), None)
        .to_dict("records")
    )


@tool
def analyze_market_tool(as_of_date: str) -> str:
    """分析 A 股核心指数并返回 Market Regime。"""

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
def rank_sector_tool(as_of_date: str, top_n: int = 8) -> str:
    """使用确定性 Style Score 对申万一级行业做横截面排名。"""

    market = analyze_market_regime(as_of_date)
    result = analyze_sectors(
        as_of_date,
        market_regime=market.regime,
        top_n=top_n,
    )
    payload = {
        "market_regime": market.regime,
        "top_sectors": _records(
            result.sectors,
            [
                "sector_code",
                "sector_name",
                "ret_20d",
                "ret_60d",
                "momentum_score",
                "valuation_score",
                "dividend_score",
                "liquidity_score",
                "primary_style",
                "trend_quality",
                "sector_score",
            ],
            top_n,
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def discover_sectors_tool(as_of_date: str, top_n: int = 6) -> str:
    """执行 Sector-first Discovery 并返回行业研究 Shortlist。"""

    result = run_discovery(as_of_date, top_n=top_n)
    payload = {
        "market_regime": result.market.regime,
        "market_score": round(result.market.score, 2),
        "style_weights": result.metadata.get("style_weights", {}),
        "rank_source": result.metadata.get("rank_source", "rule"),
        "sectors": _records(
            result.sectors.sectors,
            [
                "sector_code",
                "sector_name",
                "primary_style",
                "style_profile",
                "momentum_score",
                "valuation_score",
                "dividend_score",
                "liquidity_score",
                "rule_score",
                "ml_score",
                "sector_score",
            ],
            top_n,
        ),
    }
    return json.dumps(payload, ensure_ascii=False)




@tool
def build_research_pool_tool(
    as_of_date: str,
    sector_top_n: int = 4,
    representatives_per_sector: int = 2,
) -> str:
    """为 Top 行业选择代表性股票作为后续 7-Agent 研究入口。"""

    result = run_research_pool(
        as_of_date,
        sector_top_n=sector_top_n,
        representatives_per_sector=representatives_per_sector,
        strict_pit=True,
    )
    payload = {
        "market_regime": result.discovery.market.regime,
        "sectors": _records(
            result.discovery.sectors.sectors,
            [
                "sector_code",
                "sector_name",
                "primary_style",
                "style_profile",
                "sector_score",
            ],
            sector_top_n,
        ),
        "representatives": _records(
            result.representatives.representatives,
            [
                "ticker",
                "name",
                "sector_code",
                "sector_name",
                "representative_score",
                "selection_reason",
                "research_label",
            ],
            sector_top_n * representatives_per_sector,
        ),
        "warning": (
            "Representative Research Entry 仅用于分配研究预算，不是股票投资评级。"
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


@dataclass
class DiscoveryCoordinatorResult:
    final_message: str
    tool_trace: list[dict[str, Any]] = field(default_factory=list)


class DiscoveryCoordinatorAgent:
    """通过 LLM Tool Calling 编排行业发现工具。"""

    def __init__(self, llm, max_steps: int = 5):
        self.llm = llm
        self.max_steps = max(2, int(max_steps))
        self.tools = [
            analyze_market_tool,
            rank_sector_tool,
            discover_sectors_tool,
            build_research_pool_tool,
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
你是 A 股行业发现协调智能体。当前研究截止日期为 {as_of_date}。
所有市场环境、行业得分和 Style 判断都必须来自工具结果。

目标：
1. 先理解 Market Regime；
2. 查看申万一级行业横截面排名；
3. 根据研究预算选择最多 {max_deep_research} 个行业作为后续研究优先级；
4. 如果后续需要具体研究入口，可调用 build_research_pool_tool 为 Top 行业选择代表股；
5. 不直接声称某只股票是最佳投资标的，也不要把代表性评分当作投资评级。

研究预算：{research_budget}。
预算低时减少后续研究行业数，不要重复调用同一工具。
最终回答需要说明：
- 推荐优先研究的行业；
- 每个行业主要由 Momentum / Value / Dividend / Liquidity 中哪些 Style 驱动；
- 当前排名可能失效的条件；
- 如果已经调用代表股工具，说明哪些 ticker 只是后续 7-Agent 的 Research Entry；
- 代表性评分只解释研究路由，不得升级为买入结论。
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
                result_text = (
                    f"未知工具：{name}"
                    if tool_obj is None
                    else str(tool_obj.invoke(arguments))
                )
                trace.append({"tool_name": name, "arguments": arguments})
                messages.append(
                    ToolMessage(
                        content=result_text,
                        tool_call_id=call.get("id", name),
                    )
                )

        final = self.llm.invoke(
            messages
            + [
                HumanMessage(
                    content=(
                        "工具调用步数已达到上限。请基于已有结果直接给出"
                        "行业研究优先级，不要补充新的数字。"
                    )
                )
            ]
        )
        return DiscoveryCoordinatorResult(
            final_message=str(final.content).strip(),
            tool_trace=trace,
        )
