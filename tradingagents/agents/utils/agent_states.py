"""LangGraph 主状态定义。

最终版删除了 Research Manager、Trader 和三类 Risk Debator 的冗余状态，
将单股深度研究收敛为：三类分析师 -> Bull/Bear -> Portfolio Manager -> Auditor。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """单股深度研究主状态。"""

    company_of_interest: Annotated[str, "研究标的代码"]
    asset_type: Annotated[str, "资产类型，例如 stock 或 crypto"]
    instrument_context: Annotated[str, "启动时解析的确定性标的身份信息"]
    candidate_context: Annotated[str, "行业发现/代表股选择来源；仅作研究路由先验，不是投资证据"]
    trade_date: Annotated[str, "研究截止日期"]
    sender: Annotated[str, "最近写入消息的智能体"]

    market_report: Annotated[str, "市场与技术面分析报告"]
    news_report: Annotated[str, "新闻、公告、宏观与情绪综合报告"]
    fundamentals_report: Annotated[str, "基本面分析报告"]

    bull_thesis: Annotated[str, "看多研究员的紧凑论点"]
    bear_thesis: Annotated[str, "看空研究员的紧凑论点"]

    final_trade_decision: Annotated[str, "投资组合经理最终研究结论"]
    audit_report: Annotated[str, "决策审计报告"]
    audit_status: Annotated[str, "审计状态：PENDING/PASS/REVISE"]
    audit_feedback: Annotated[str, "提供给投资组合经理的修订要求"]
    audit_issues: Annotated[list[dict[str, Any]], "结构化审计问题"]
    audit_repair_target: Annotated[str, "本轮优先重新取证/修订的责任能力"]
    audit_round: Annotated[int, "已经执行的审计轮次"]

    past_context: Annotated[str, "同标的历史决策与复盘摘要"]
    analyst_trace: Annotated[list[dict[str, Any]], operator.add]
