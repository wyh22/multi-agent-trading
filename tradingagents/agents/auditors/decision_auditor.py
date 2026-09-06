"""决策审计智能体。

该节点不生成新的投资观点，只检查最终研究结论，并把实质性问题
结构化路由到最适合重新取证/修订的责任能力。
"""

from __future__ import annotations

from tradingagents.agents.schemas import AuditResult, render_audit_result
from tradingagents.agents.utils.agent_utils import get_candidate_context_from_state
from tradingagents.agents.utils.context_compaction import (
    build_decision_context,
    compact_text,
)
from tradingagents.agents.utils.evidence_claims import claim_usage_instruction
from tradingagents.agents.utils.structured import NO_EXTERNAL_TOOLS, bind_structured


def _fallback_audit(llm, prompt: str) -> tuple[str, str, str, list[dict], str]:
    """结构化输出不可用时，保守退回 PM 文本修订。"""

    response = llm.invoke(
        prompt
        + "\n\n结构化输出不可用。请仅输出 PASS 或 REVISE 开头的简短审计结论。"
    )
    text = response.content.strip()
    status = "REVISE" if text.upper().startswith("REVISE") else "PASS"
    feedback = text if status == "REVISE" else ""
    return text, status, feedback, [], "portfolio_manager"


def _primary_target(result: AuditResult) -> str:
    if not result.issues:
        return "portfolio_manager"
    target = result.issues[0].repair_target
    # RAG is an evidence tool rather than a graph node. Route the actual repair
    # to the evidence-owning analyst; the analyst may call shared RAG itself.
    if target == "rag":
        return "news"
    return target


def create_decision_auditor(llm):
    """创建最终决策审计节点。"""

    structured_llm = bind_structured(llm, AuditResult, "决策审计智能体")

    def audit_node(state) -> dict:
        trade_date = state["trade_date"]
        final_decision = compact_text(state.get("final_trade_decision", ""), 2600)
        evidence_context = build_decision_context(state)
        candidate_context = get_candidate_context_from_state(state)
        current_round = int(state.get("audit_round", 0) or 0) + 1

        prompt = f"""
你是独立的决策审计智能体。你不负责提出新的投资观点，只负责检查最终结论。
研究截止日为 `{trade_date}`。

{candidate_context}

## 上游证据
{evidence_context}

## Claim 类型规则
{claim_usage_instruction()}

## 待审计最终结论
{final_decision}

检查项：
1. 事实与数字是否能在上游证据中找到依据；
2. 是否出现截止日之后的信息或把报告期误当披露日；
3. 是否把推断写成事实，或给出无证据的因果关系、价格目标、行业地位等；
4. 最终评级是否与核心证据方向明显矛盾；
5. 是否存在同一指标前后数值冲突；
6. 是否把行业发现得分、Style 标签或代表股选择原因误写成公司投资事实或评级依据。

修复路由规则：
- 行情/指标/价格证据缺失 -> market；
- 公告/新闻/政策/公司文档证据缺失 -> news；
- 财务报表/估值/现金流证据缺失 -> fundamentals；
- 只是最终综合、措辞、评级逻辑问题 -> portfolio_manager；
- 如果最直接需要重新查知识库，可以标记 rag，系统会交给能调用共享 RAG 的专业 Agent。
- REVISE 时 issues 至少给出一个问题，并明确 instruction。
- PASS 时 issues 应为空。

判定规则：
- 轻微措辞问题可以 PASS；
- 只有会实质影响结论可信度的问题才 REVISE；
- unsupported_claims、revision_instructions、issues 各最多 5 项；
- 不得调用工具，不得加入新事实。

{NO_EXTERNAL_TOOLS}
""".strip()

        if structured_llm is None:
            report, status, feedback, issues, repair_target = _fallback_audit(
                llm, prompt
            )
        else:
            try:
                result = structured_llm.invoke(prompt)
                if result is None:
                    raise ValueError("结构化审计没有返回结果")
                report = render_audit_result(result)
                status = result.verdict
                issues = [item.model_dump() for item in result.issues[:5]]
                repair_target = (
                    _primary_target(result)
                    if status == "REVISE"
                    else "portfolio_manager"
                )
                instructions = list(result.revision_instructions[:5])
                if status == "REVISE" and result.issues:
                    instructions.extend(
                        f"[{item.repair_target}/{item.issue_type}] {item.instruction}"
                        for item in result.issues[:5]
                    )
                feedback = (
                    "\n".join(f"- {item}" for item in instructions[:8])
                    if status == "REVISE"
                    else ""
                )
            except Exception:
                report, status, feedback, issues, repair_target = _fallback_audit(
                    llm, prompt
                )

        return {
            "audit_report": report,
            "audit_status": status,
            "audit_feedback": feedback,
            "audit_issues": issues,
            "audit_repair_target": repair_target,
            "audit_round": current_round,
        }

    return audit_node
