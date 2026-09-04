"""用于限制多智能体上下文体量的 Claim-aware 确定性压缩工具。

核心原则：
- 原始报告仍单独保存，便于审计；
- 下游优先消费显式标注的 FACT / CALCULATION / INFERENCE / CONDITIONAL；
- 若旧报告没有标签，则使用保守规则做确定性分类；
- 在字符预算内优先保留事实、计算和类型多样性，而不是机械截取文本头尾。
"""

from __future__ import annotations

from tradingagents.agents.utils.evidence_claims import compress_claims


def compact_text(text: str | None, max_chars: int = 2600) -> str:
    """保留给最终决策等非证据文本的原始字符预算截断。"""

    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    head = int(max_chars * 0.72)
    tail = max_chars - head
    return (
        value[:head].rstrip()
        + "\n\n……【中间内容已按上下文预算省略，原始报告仍保存在 trace 目录】……\n\n"
        + value[-tail:].lstrip()
    )


def compact_evidence_text(
    text: str | None,
    *,
    source_section: str,
    max_chars: int,
) -> str:
    """将一段研究报告压缩为带 Claim Type 的证据接口。"""

    result = compress_claims(
        text,
        source_section=source_section,
        max_chars=max_chars,
    )
    return result.rendered


def build_analyst_context(state: dict, per_report_chars: int = 2200) -> str:
    """构造给 Bull/Bear 使用的三类分析师 Claim-aware 紧凑上下文。"""

    sections = [
        ("市场与技术面", state.get("market_report", "")),
        ("新闻、公告、宏观与情绪", state.get("news_report", "")),
        ("基本面", state.get("fundamentals_report", "")),
    ]
    parts: list[str] = []
    for title, content in sections:
        if not content:
            continue
        compressed = compact_evidence_text(
            content,
            source_section=title,
            max_chars=per_report_chars,
        )
        if compressed:
            parts.append(f"## {title}\n{compressed}")
    return "\n\n".join(parts)


def build_decision_context(state: dict) -> str:
    """构造 PM/Auditor 使用的 Claim-aware 紧凑证据包。"""

    analyst = build_analyst_context(state, per_report_chars=1500)
    bull = compact_evidence_text(
        state.get("bull_thesis", ""),
        source_section="看多研究",
        max_chars=1600,
    )
    bear = compact_evidence_text(
        state.get("bear_thesis", ""),
        source_section="看空研究",
        max_chars=1600,
    )
    parts = [analyst]
    if bull:
        parts.append(f"## 看多论点\n{bull}")
    if bear:
        parts.append(f"## 看空论点\n{bear}")
    return "\n\n".join(part for part in parts if part)
