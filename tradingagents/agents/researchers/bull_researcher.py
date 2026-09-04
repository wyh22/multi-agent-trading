"""看多研究员：基于三类分析师证据独立形成一次性看多论点。"""

from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.context_compaction import build_analyst_context


def create_bull_researcher(llm):
    """创建只读取分析证据、只写入 bull_thesis 的看多研究员。"""

    def bull_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        evidence_context = build_analyst_context(state)

        prompt = f"""
你是 A 股研究系统中的看多研究员。你的职责不是与另一名研究员多轮辩论，
而是基于已经完成的市场、新闻/公告/宏观、基本面证据，独立形成一次紧凑的看多假设。

{instrument_context}

## 上游证据
{evidence_context or '当前没有可用的上游证据。'}

要求：
1. 只使用上游证据，不调用外部工具，不补充未经当前证据支持的新事实。
2. 优先提炼能够支持上涨或基本面改善的驱动因素、竞争优势、催化与验证条件。
3. 明确指出看多逻辑中最脆弱、仍需验证的部分，避免单边乐观。
4. 不假设 Bear Researcher 已经发言，也不要编造对手观点。
5. 不给出没有证据支持的精确目标价、收益率或仓位数字。
6. 输出保持紧凑，建议约 500~900 个中文字符。

{get_language_instruction()}
""".strip()

        response = llm.invoke(prompt)
        return {"bull_thesis": str(response.content).strip()}

    return bull_node
