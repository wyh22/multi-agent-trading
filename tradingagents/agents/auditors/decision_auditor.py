"""决策审计智能体。

该智能体不生成新的投资观点，只检查最终研究结论是否被现有证据支持、
是否违反 Point-in-Time 截止规则、是否存在数字或方向冲突。若发现明显问题，
通过 LangGraph 条件边要求投资组合经理最多修订一次。
"""

from __future__ import annotations

from tradingagents.agents.schemas import AuditResult, render_audit_result
from tradingagents.agents.utils.context_compaction import build_decision_context, compact_text
from tradingagents.agents.utils.evidence_claims import claim_usage_instruction
from tradingagents.agents.utils.structured import NO_EXTERNAL_TOOLS, bind_structured


def _fallback_audit(llm, prompt: str) -> tuple[str, str, str]:
    """结构化输出不可用时，使用保守的文本审计回退。"""

    response = llm.invoke(
        prompt
        + "\n\n结构化输出不可用。请仅输出 PASS 或 REVISE 开头的简短审计结论。"
    )
    text = response.content.strip()
    status = "REVISE" if text.upper().startswith("REVISE") else "PASS"
    feedback = text if status == "REVISE" else ""
    return text, status, feedback


def create_decision_auditor(llm):
    """创建最终决策审计节点。"""

    structured_llm = bind_structured(llm, AuditResult, "决策审计智能体")

    def audit_node(state) -> dict:
        trade_date = state["trade_date"]
        final_decision = compact_text(state.get("final_trade_decision", ""), 2600)
        evidence_context = build_decision_context(state)
        current_round = int(state.get("audit_round", 0) or 0) + 1

        prompt = f"""
你是独立的决策审计智能体。你不负责提出新的投资观点，只负责检查最终结论。
研究截止日为 `{trade_date}`。

## 上游证据
{evidence_context}
\n## Claim 类型规则\n{claim_usage_instruction()}\n
## 待审计最终结论
{final_decision}

检查项：
1. 事实与数字是否能在上游证据中找到依据；
2. 是否出现截止日之后的信息或把报告期误当披露日；
3. 是否把推断写成事实，或给出无证据的因果关系、价格目标、行业地位等；
4. 最终评级是否与核心证据方向明显矛盾；
5. 是否存在同一指标前后数值冲突。

判定规则：
- 轻微措辞问题可以 PASS；
- 只有会实质影响结论可信度的问题才 REVISE；
- unsupported_claims 和 revision_instructions 各最多 5 项；
- 审计摘要控制在 300~500 个中文字符；
- 不得调用工具，不得加入新事实。

{NO_EXTERNAL_TOOLS}
""".strip()

        if structured_llm is None:
            report, status, feedback = _fallback_audit(llm, prompt)
        else:
            try:
                result = structured_llm.invoke(prompt)
                if result is None:
                    raise ValueError("结构化审计没有返回结果")
                report = render_audit_result(result)
                status = result.verdict
                feedback = (
                    "\n".join(f"- {item}" for item in result.revision_instructions[:5])
                    if status == "REVISE"
                    else ""
                )
            except Exception:
                report, status, feedback = _fallback_audit(llm, prompt)

        return {
            "audit_report": report,
            "audit_status": status,
            "audit_feedback": feedback,
            "audit_round": current_round,
        }

    return audit_node
