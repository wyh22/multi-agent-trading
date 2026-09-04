"""用于限制多智能体上下文体量的确定性压缩工具。

这里不调用额外 LLM，只做字符预算截断。核心原则是：
- 原始报告仍单独保存，便于审计；
- 下游智能体只读取有限长度的证据摘要，降低 Token 重复；
- 截断时保留文本头尾，兼顾结论与限制说明。
"""

from __future__ import annotations


def compact_text(text: str | None, max_chars: int = 2600) -> str:
    """按字符预算压缩文本，超限时保留开头和结尾。"""

    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    head = int(max_chars * 0.72)
    tail = max_chars - head
    return (
        value[:head].rstrip()
        + "

……【中间内容已按上下文预算省略，原始报告仍保存在 trace 目录】……

"
        + value[-tail:].lstrip()
    )


def build_analyst_context(state: dict, per_report_chars: int = 2200) -> str:
    """构造给 Bull/Bear 使用的三类分析师紧凑上下文。"""

    sections = [
        ("市场与技术面", state.get("market_report", "")),
        ("新闻、公告、宏观与情绪", state.get("news_report", "")),
        ("基本面", state.get("fundamentals_report", "")),
    ]
    return "

".join(
        f"## {title}
{compact_text(content, per_report_chars)}"
        for title, content in sections
        if content
    )


def build_decision_context(state: dict) -> str:
    """构造投资组合经理和审计智能体使用的紧凑证据包。"""

    analyst = build_analyst_context(state, per_report_chars=1500)
    bull = compact_text(state.get("bull_thesis", ""), 1600)
    bear = compact_text(state.get("bear_thesis", ""), 1600)
    parts = [analyst]
    if bull:
        parts.append(f"## 看多论点
{bull}")
    if bear:
        parts.append(f"## 看空论点
{bear}")
    return "

".join(part for part in parts if part)
