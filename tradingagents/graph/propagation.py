"""图状态初始化与运行参数组装。"""

from __future__ import annotations

from typing import Any


class Propagator:
    """负责创建初始状态并生成 LangGraph 调用参数。"""

    def __init__(self, max_recur_limit: int = 60):
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        asset_type: str = "stock",
        past_context: str = "",
        instrument_context: str = "",
        candidate_context: str = "",
    ) -> dict[str, Any]:
        """创建一次单股深度研究的初始状态。"""

        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "asset_type": asset_type,
            "instrument_context": instrument_context,
            "candidate_context": candidate_context,
            "trade_date": str(trade_date),
            "sender": "",
            "past_context": past_context,
            "market_report": "",
            "news_report": "",
            "fundamentals_report": "",
            "bull_thesis": "",
            "bear_thesis": "",
            "final_trade_decision": "",
            "audit_report": "",
            "audit_status": "PENDING",
            "audit_feedback": "",
            "audit_round": 0,
            "analyst_trace": [],
        }

    def get_graph_args(
        self,
        callbacks: list | None = None,
        max_concurrency: int | None = None,
        metadata: dict | None = None,
        stream_mode: str = "values",
    ) -> dict[str, Any]:
        """生成 LangGraph invoke/stream 所需参数。"""

        config: dict[str, Any] = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        if max_concurrency:
            config["max_concurrency"] = int(max_concurrency)
        if metadata:
            config["metadata"] = dict(metadata)
        return {"stream_mode": stream_mode, "config": config}
