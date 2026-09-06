from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from tradingagents.conversation import ConversationAgent, ConversationStore
from tradingagents.evaluation.architecture import (
    ArchitectureBenchmarkCase,
    ArchitectureSystemRun,
)
from tradingagents.evaluation.telemetry import EvaluationTelemetryHandler
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _isolated_config(base_config: dict[str, Any], root: Path) -> dict[str, Any]:
    config = dict(base_config)
    config["results_dir"] = str(root / "results")
    config["data_cache_dir"] = str(root / "cache")
    config["memory_log_path"] = str(root / "memory" / "trading_memory.md")
    config["conversation_db_path"] = str(root / "conversation.db")
    config["checkpoint_enabled"] = False
    config["memory_reflection_enabled"] = False
    return config


class SingleAgentAllToolsSystem:
    """One LLM with the full tool catalog and no Supervisor/specialist routing."""

    name = "single-agent-all-tools"

    def __init__(self, base_config: dict[str, Any]):
        self.base_config = dict(base_config)

    def run(self, case: ArchitectureBenchmarkCase) -> ArchitectureSystemRun:
        started = time.perf_counter()
        telemetry = EvaluationTelemetryHandler()
        try:
            with tempfile.TemporaryDirectory(
                prefix="tradingagents-bench-single-"
            ) as tmp:
                root = Path(tmp)
                config = _isolated_config(self.base_config, root)
                store = ConversationStore(config["conversation_db_path"])
                agent = ConversationAgent(
                    config,
                    store,
                    callbacks=[telemetry],
                )
                evidence: list[str] = []
                trace: list[dict[str, Any]] = []
                answer = agent._tool_chat(
                    message=case.user_query,
                    history=[],
                    ticker=case.ticker,
                    as_of_date=case.as_of_date,
                    research_context="",
                    evidence_sink=evidence,
                    trace_sink=trace,
                )
            return ArchitectureSystemRun(
                case_id=case.case_id,
                system_name=self.name,
                answer=answer,
                route_action="call_tool",
                route_target="auto",
                response_status="RAW",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                telemetry=telemetry.snapshot(),
                evidence=evidence,
                tool_trace=trace,
            )
        except Exception as exc:  # noqa: BLE001
            return ArchitectureSystemRun(
                case_id=case.case_id,
                system_name=self.name,
                answer="",
                route_action="call_tool",
                route_target="auto",
                response_status="SYSTEM_ERROR",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                telemetry=telemetry.snapshot(),
                error=f"{type(exc).__name__}: {exc}",
            )


class FixedDeepResearchSystem:
    """Always run the complete fixed research graph, then answer the query."""

    name = "fixed-deep-research"

    def __init__(self, base_config: dict[str, Any]):
        self.base_config = dict(base_config)

    def run(self, case: ArchitectureBenchmarkCase) -> ArchitectureSystemRun:
        started = time.perf_counter()
        telemetry = EvaluationTelemetryHandler()
        if not case.ticker:
            return ArchitectureSystemRun(
                case_id=case.case_id,
                system_name=self.name,
                answer="",
                route_action="run_deep_research",
                route_target="deep_stock_research",
                response_status="REVIEW_REQUIRED",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                telemetry=telemetry.snapshot(),
                error="ticker is required for fixed deep research",
            )

        try:
            with tempfile.TemporaryDirectory(
                prefix="tradingagents-bench-fixed-"
            ) as tmp:
                root = Path(tmp)
                config = _isolated_config(self.base_config, root)
                graph = TradingAgentsGraph(
                    selected_analysts=("market", "news", "fundamentals"),
                    config=config,
                    callbacks=[telemetry],
                )
                state, _signal = graph.propagate(
                    case.ticker,
                    case.as_of_date,
                )

                evidence = [
                    str(state.get(key, "") or "")
                    for key in (
                        "market_report",
                        "news_report",
                        "fundamentals_report",
                        "bull_thesis",
                        "bear_thesis",
                    )
                    if str(state.get(key, "") or "").strip()
                ]
                final_decision = str(
                    state.get("final_trade_decision", "") or ""
                )
                evidence_text = "\n\n".join(evidence)
                prompt = f"""
你是基准评测中的最终回答器。完整固定研究图已经运行完毕。
请只基于下列研究结果回答原始用户问题，不新增任何事实或数字。
研究截止日：{case.as_of_date}
标的：{case.ticker}

用户问题：
{case.user_query}

分析师/研究员证据：
{evidence_text[:24000]}

Portfolio Manager 原始结论：
{final_decision[:8000]}

要求：
1. 直接回答用户实际问题，而不是机械复述完整研报；
2. 如果证据不足，明确说明缺口；
3. 不得使用截止日之后的信息。
""".strip()
                answer = str(
                    graph.quick_thinking_llm.invoke(
                        [HumanMessage(content=prompt)]
                    ).content
                    or ""
                )
                audit_status = str(
                    state.get("audit_status", "") or ""
                ).upper()
                response_status = (
                    "COMPLETE"
                    if audit_status == "PASS"
                    else "REVIEW_REQUIRED"
                )
                trace = list(state.get("analyst_trace", []) or [])

            return ArchitectureSystemRun(
                case_id=case.case_id,
                system_name=self.name,
                answer=answer,
                route_action="run_deep_research",
                route_target="deep_stock_research",
                response_status=response_status,
                audit_status=audit_status,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                telemetry=telemetry.snapshot(),
                evidence=evidence,
                tool_trace=trace,
            )
        except Exception as exc:  # noqa: BLE001
            return ArchitectureSystemRun(
                case_id=case.case_id,
                system_name=self.name,
                answer="",
                route_action="run_deep_research",
                route_target="deep_stock_research",
                response_status="SYSTEM_ERROR",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                telemetry=telemetry.snapshot(),
                error=f"{type(exc).__name__}: {exc}",
            )


class DynamicSupervisorSystem:
    """V1.6 Conversation Supervisor with Task Contract and Completion Gate."""

    name = "dynamic-supervisor"

    def __init__(self, base_config: dict[str, Any]):
        self.base_config = dict(base_config)

    def run(self, case: ArchitectureBenchmarkCase) -> ArchitectureSystemRun:
        started = time.perf_counter()
        telemetry = EvaluationTelemetryHandler()
        try:
            with tempfile.TemporaryDirectory(
                prefix="tradingagents-bench-supervisor-"
            ) as tmp:
                root = Path(tmp)
                config = _isolated_config(self.base_config, root)
                store = ConversationStore(config["conversation_db_path"])
                agent = ConversationAgent(
                    config,
                    store,
                    callbacks=[telemetry],
                )
                result = agent.chat(
                    case.user_query,
                    ticker=case.ticker,
                    as_of_date=case.as_of_date,
                    force_mode="auto",
                    capture_evidence=True,
                )
                trace = list(result.get("supervisor_trace", []) or [])
                first = trace[0] if trace else {}
                route_action = str(
                    first.get("action")
                    or result.get("supervisor_action")
                    or ""
                )
                route_target = (
                    first.get("target")
                    if first
                    else result.get("supervisor_target")
                )

            return ArchitectureSystemRun(
                case_id=case.case_id,
                system_name=self.name,
                answer=str(result.get("answer", "") or ""),
                route_action=route_action,
                route_target=str(route_target) if route_target else None,
                response_status=str(result.get("status", "") or ""),
                audit_status=str(result.get("audit_status", "") or ""),
                system_completion_ratio=float(
                    result.get("completion_ratio", 0.0) or 0.0
                ),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                telemetry=telemetry.snapshot(),
                evidence=[
                    str(item)
                    for item in result.get("evaluation_evidence", []) or []
                    if str(item).strip()
                ],
                tool_trace=[
                    dict(item)
                    for item in result.get("evaluation_tool_trace", []) or []
                    if isinstance(item, dict)
                ],
            )
        except Exception as exc:  # noqa: BLE001
            return ArchitectureSystemRun(
                case_id=case.case_id,
                system_name=self.name,
                answer="",
                route_action="",
                route_target=None,
                response_status="SYSTEM_ERROR",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                telemetry=telemetry.snapshot(),
                error=f"{type(exc).__name__}: {exc}",
            )


SYSTEM_BUILDERS = {
    SingleAgentAllToolsSystem.name: SingleAgentAllToolsSystem,
    FixedDeepResearchSystem.name: FixedDeepResearchSystem,
    DynamicSupervisorSystem.name: DynamicSupervisorSystem,
}
