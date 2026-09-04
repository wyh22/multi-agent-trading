"""投资组合经理：一次综合多空观点，直接生成最终研究决策。"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.context_compaction import build_decision_context
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)


def create_portfolio_manager(llm):
    """创建投资组合经理节点。"""

    structured_llm = bind_structured(llm, PortfolioDecision, "投资组合经理")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        evidence_context = build_decision_context(state)
        past_context = state.get("past_context", "")
        audit_feedback = state.get("audit_feedback", "").strip()
        audit_round = int(state.get("audit_round", 0) or 0)

        revision_block = ""
        if audit_feedback:
            revision_block = f"""
## 审计反馈
这是一次修订调用。请只针对下列问题修正原结论，不要无意义扩写：
{audit_feedback}
""".strip()

        prompt = f"""
你是投资组合经理，负责把三类分析师证据和 Bull/Bear 两个互补假设收敛为最终研究结论。
当前系统定位是“研究与候选发现”，不是自动交易，因此不要为了显得可执行而编造目标价、
精确收益率或不存在的仓位依据。

{instrument_context}

## 紧凑证据包
{evidence_context}

## 历史复盘摘要
{past_context or '无'}

{revision_block}

评级只能使用：Buy / Overweight / Hold / Underweight / Sell。
要求：
1. 结论必须能追溯到当前证据；没有证据时明确写“不确定/待验证”。
2. 同时吸收 Bull 与 Bear 的有效部分，不重复抄写其全文。
3. 最多列出 5 项关键风险、5 项催化和 5 项失效条件。
4. 不得给出无法由当前数据支持的精确价格目标。
5. 若多空证据接近，允许 Hold；否则必须根据更强证据明确倾向。
6. 输出要紧凑，目标约 800~1200 个中文字符。
7. 如果结构化输出不可用，普通文本第一行必须写 `Rating: <五档评级>`。
8. 当前审计轮次为 {audit_round}。

{NO_EXTERNAL_TOOLS}
{get_language_instruction()}
""".strip()

        final_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "投资组合经理",
        )

        return {
            "final_trade_decision": final_decision.strip(),
            "audit_status": "PENDING",
        }

    return portfolio_manager_node
