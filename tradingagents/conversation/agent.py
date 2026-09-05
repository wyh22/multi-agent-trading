from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tradingagents.agents.utils.tool_registry import build_tool_groups
from tradingagents.capabilities.registry import CapabilityRegistry, CapabilitySpec
from tradingagents.conversation.router import extract_ticker
from tradingagents.conversation.store import ConversationStore
from tradingagents.discovery.pipeline import run_discovery, run_research_pool
from tradingagents.graph.trading_graph import TradingAgentsGraph, _coerce_max_retries
from tradingagents.llm_clients import create_llm_client
from tradingagents.orchestration.analyst_executor import SpecialistAgentExecutor
from tradingagents.orchestration.schemas import ExecutionResult, SupervisorAction
from tradingagents.orchestration.supervisor import ConversationSupervisor
from tradingagents.skills.registry import BUILTIN_SKILLS

logger = logging.getLogger(__name__)


class ConversationAgent:
    """Conversation-first supervisor over tools, specialist agents and skills."""

    def __init__(self, config: dict[str, Any], store: ConversationStore):
        self.config = config
        self.store = store
        self.llm = self._create_quick_llm()
        self.tool_groups = build_tool_groups(self.config)
        self.tools = self._build_tools()
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools) if self.tools else self.llm
        self.registry = self._build_capability_registry()
        self.supervisor = ConversationSupervisor(self.llm, self.registry)
        self.specialists = SpecialistAgentExecutor(
            self.llm,
            self.tool_groups,
            max_recur_limit=int(self.config.get("max_recur_limit", 60)),
        )

    def _provider_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        provider = str(self.config.get("llm_provider", "")).lower()
        if provider == "google" and self.config.get("google_thinking_level"):
            kwargs["thinking_level"] = self.config["google_thinking_level"]
        elif provider == "openai" and self.config.get("openai_reasoning_effort"):
            kwargs["reasoning_effort"] = self.config["openai_reasoning_effort"]
        elif provider == "anthropic" and self.config.get("anthropic_effort"):
            kwargs["effort"] = self.config["anthropic_effort"]
        temperature = self.config.get("temperature")
        if temperature not in (None, ""):
            kwargs["temperature"] = float(temperature)
        max_retries = self.config.get("llm_max_retries")
        if max_retries not in (None, ""):
            kwargs["max_retries"] = _coerce_max_retries(max_retries)
        return kwargs

    def _create_quick_llm(self):
        client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **self._provider_kwargs(),
        )
        return client.get_llm()

    def _build_tools(self):
        merged = []
        seen = set()
        for group in ("market", "news", "fundamentals", "knowledge"):
            for tool in self.tool_groups.get(group, []):
                if tool.name not in seen:
                    merged.append(tool)
                    seen.add(tool.name)

        if self.config.get("external_mcp_enabled", False):
            raw = str(
                self.config.get("external_mcp_servers_json", "") or ""
            ).strip()
            if raw:
                try:
                    servers = json.loads(raw)
                    if not isinstance(servers, dict):
                        raise ValueError("external_mcp_servers_json必须是JSON对象")
                    from tradingagents.mcp.client import load_mcp_tools_sync

                    allowlist = {
                        item.strip()
                        for item in str(
                            self.config.get("external_mcp_tool_allowlist", "") or ""
                        ).split(",")
                        if item.strip()
                    }
                    for tool in load_mcp_tools_sync(servers=servers):
                        if tool.name in allowlist and tool.name not in seen:
                            merged.append(tool)
                            seen.add(tool.name)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "外部MCP工具加载失败，将继续使用核心工具: %s",
                        exc,
                    )
        return merged

    def _build_capability_registry(self) -> CapabilityRegistry:
        registry = CapabilityRegistry()
        for tool in self.tools:
            description = str(getattr(tool, "description", "") or tool.name)
            registry.register(
                CapabilitySpec(
                    name=tool.name,
                    kind="tool",
                    description=description[:260],
                    requires_ticker=tool.name
                    not in {"get_global_news", "get_macro_indicators"},
                )
            )
        for key, description in {
            "market": "市场/技术面专业 Agent；可自主循环调用行情、指标和验证工具。",
            "news": "新闻/公告/宏观专业 Agent；可自主调用新闻、宏观和共享 RAG。",
            "fundamentals": "基本面专业 Agent；可自主调用财务报表工具和共享 RAG。",
        }.items():
            registry.register(
                CapabilitySpec(
                    name=key,
                    kind="agent",
                    description=description,
                    requires_ticker=True,
                    expensive=True,
                )
            )
        for skill in BUILTIN_SKILLS.values():
            registry.register(
                CapabilitySpec(
                    name=skill.name,
                    kind="skill",
                    description=skill.description,
                    requires_ticker=skill.requires_ticker,
                    expensive=skill.name
                    in {"deep_stock_research", "company_comparison"},
                )
            )
        return registry

    @staticmethod
    def _system_prompt(
        *,
        ticker: str | None,
        as_of_date: str,
        research_context: str,
    ) -> str:
        context = research_context.strip()
        if len(context) > 18000:
            context = context[:18000] + "\n...[历史研究上下文已截断]"
        return f"""你是A股研究对话助手。当前研究截止日期为 {as_of_date}。
当前会话标的：{ticker or '未指定'}。

规则：
1. 涉及行情、财务、公告、宏观或知识库事实时优先调用工具，不得凭空补数字。
2. 所有历史事实必须遵守研究截止日期，不得使用截止日之后的信息。
3. 如果用户只是在追问上一轮研究结论，优先复用已审计研究上下文。
4. 工具返回不可用时明确说明数据缺失；不要伪造来源。
5. 本系统只提供研究辅助，不自动执行证券交易。
6. 区分事实、程序计算结果、模型推断与条件性判断。

上一轮研究上下文：
{context or '无'}
"""

    @staticmethod
    def _research_context(state: dict[str, Any]) -> str:
        sections = []
        for key, title in (
            ("market_report", "市场分析"),
            ("news_report", "新闻与事件"),
            ("fundamentals_report", "基本面"),
            ("bull_thesis", "看多论点"),
            ("bear_thesis", "看空论点"),
            ("final_trade_decision", "最终研究结论"),
            ("audit_report", "审计结果"),
        ):
            value = str(state.get(key, "") or "").strip()
            if value:
                sections.append(f"## {title}\n{value}")
        return "\n\n".join(sections)[:30000]

    def _inject_common_args(
        self,
        name: str,
        args: dict[str, Any],
        *,
        ticker: str | None,
        as_of_date: str,
    ) -> dict[str, Any]:
        values = dict(args)
        if ticker:
            if name in {
                "get_stock_data",
                "get_indicators",
                "get_verified_market_snapshot",
            }:
                values.setdefault("symbol", ticker)
            elif name in {
                "get_fundamentals",
                "get_balance_sheet",
                "get_cashflow",
                "get_income_statement",
                "get_news",
                "get_insider_transactions",
                "search_company_knowledge",
            }:
                values.setdefault("ticker", ticker)
        if name in {
            "get_fundamentals",
            "get_insider_transactions",
            "get_global_news",
            "get_macro_indicators",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement",
        }:
            values.setdefault("curr_date", as_of_date)
        if name == "search_company_knowledge":
            values.setdefault("as_of_date", as_of_date)
        if name == "get_news":
            values.setdefault("end_date", as_of_date)
        return values

    def _invoke_atomic_tool(
        self,
        name: str,
        args: dict[str, Any],
        *,
        ticker: str | None,
        as_of_date: str,
    ) -> ExecutionResult:
        tool = self.tools_by_name.get(name)
        if tool is None:
            return ExecutionResult(
                status="UNAVAILABLE",
                capability=name,
                content=f"TOOL_UNAVAILABLE: {name}",
                fallback_available=True,
            )
        try:
            result = tool.invoke(
                self._inject_common_args(
                    name,
                    args,
                    ticker=ticker,
                    as_of_date=as_of_date,
                )
            )
            text = str(result)
            status = "SUCCESS"
            if text.startswith(("NO_DATA_AVAILABLE", "NO_RAG_EVIDENCE")):
                status = "NO_DATA"
            elif text.startswith(("DATA_UNAVAILABLE", "RAG_UNAVAILABLE")):
                status = "UNAVAILABLE"
            elif text.startswith("TOOL_ERROR"):
                status = "FAILED"
            return ExecutionResult(
                status=status,
                capability=name,
                content=text[:16000],
                fallback_available=status != "SUCCESS",
            )
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                status="FAILED",
                capability=name,
                content=f"{type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
                retryable=True,
            )

    def _tool_chat(
        self,
        *,
        message: str,
        history: list[dict[str, str]],
        ticker: str | None,
        as_of_date: str,
        research_context: str,
    ) -> str:
        messages: list[Any] = [
            SystemMessage(
                content=self._system_prompt(
                    ticker=ticker,
                    as_of_date=as_of_date,
                    research_context=research_context,
                )
            )
        ]
        for item in history[-10:]:
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=message))

        max_rounds = max(
            1,
            int(self.config.get("conversation_tool_rounds", 4)),
        )
        for _ in range(max_rounds):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                return str(response.content or "")
            for call in tool_calls:
                name = str(call.get("name") or "")
                result = self._invoke_atomic_tool(
                    name,
                    dict(call.get("args") or {}),
                    ticker=ticker,
                    as_of_date=as_of_date,
                )
                text = result.content
                if len(text) > 12000:
                    text = text[:12000] + "\n...[工具结果已截断]"
                messages.append(
                    ToolMessage(
                        content=text,
                        tool_call_id=str(call.get("id", name or "tool")),
                        name=name,
                    )
                )
        final = self.llm.invoke(
            messages
            + [
                HumanMessage(
                    content=(
                        "工具调用轮次已达到上限。基于已有结果回答，"
                        "并明确仍缺少的信息。"
                    )
                )
            ]
        )
        return str(final.content or "")

    def _synthesize(
        self,
        *,
        message: str,
        evidence: str,
        ticker: str | None,
        as_of_date: str,
        research_context: str,
    ) -> str:
        prompt = f"""{self._system_prompt(
            ticker=ticker,
            as_of_date=as_of_date,
            research_context=research_context,
        )}

用户问题：
{message}

本轮已执行能力返回的证据/报告：
{evidence[:18000]}

请直接回答用户问题。只能基于上面的工具/Agent 结果和已审计上下文；
若信息缺失就明确说明，不得补造数字或来源。
"""
        return str(self.llm.invoke(prompt).content or "")

    def _run_deep_research(
        self,
        *,
        ticker: str,
        cutoff: str,
        thread: dict[str, Any],
    ) -> tuple[str, dict[str, Any], str, str]:
        metadata = (
            thread.get("metadata", {})
            if isinstance(thread.get("metadata"), dict)
            else {}
        )
        context_map = metadata.get("representative_contexts", {})
        candidate_context = (
            str(context_map.get(ticker, "") or "")
            if isinstance(context_map, dict)
            else ""
        )
        graph = TradingAgentsGraph(config=self.config)
        state, signal = graph.propagate(
            ticker,
            cutoff,
            candidate_context=candidate_context,
        )
        research_context = self._research_context(state)
        answer = str(state.get("final_trade_decision", "") or "")
        audit = str(state.get("audit_report", "") or "")
        if audit:
            answer += f"\n\n---\n**结果审计**\n{audit}"
        return answer, state, signal, research_context

    @staticmethod
    def _wants_representatives(message: str) -> bool:
        normalized = (message or "").lower()
        return any(
            word in normalized
            for word in (
                "代表股",
                "代表性股票",
                "候选股",
                "研究池",
                "研究标的",
                "每个行业",
                "每个板块",
            )
        )

    def _run_sector_skill(self, message: str, cutoff: str) -> tuple[str, dict]:
        top_n = int(self.config.get("sector_discovery_top_n", 6))
        if self._wants_representatives(message):
            pool = run_research_pool(
                cutoff,
                sector_top_n=min(top_n, 6),
                representatives_per_sector=int(
                    self.config.get("representatives_per_sector", 2)
                ),
                component_limit=int(
                    self.config.get("representative_component_limit", 20)
                ),
                strict_pit=True,
                ml_model_path=self.config.get("sector_ml_model_path") or None,
                ml_weight=float(self.config.get("sector_ml_weight", 0.5)),
            )
            result = pool.discovery
            rows = result.sectors.sectors.to_dict(orient="records")
            reps = pool.representatives.representatives.to_dict(orient="records")
            lines = [
                f"截至{cutoff}，市场状态为{result.market.regime}（{result.market.score:.1f}/100）。",
                "行业研究优先级：",
            ]
            for idx, row in enumerate(rows, start=1):
                lines.append(
                    f"{idx}. {row.get('sector_code','')} {row.get('sector_name','')}，"
                    f"Style={row.get('style_profile') or row.get('primary_style') or 'N/A'}，"
                    f"行业分={float(row.get('sector_score', 0.0)):.2f}"
                )
                for rep in [
                    item
                    for item in reps
                    if str(item.get("sector_code"))
                    == str(row.get("sector_code"))
                ]:
                    lines.append(
                        f"   - {rep.get('ticker','')} {rep.get('name','')}："
                        f"代表性分={float(rep.get('representative_score', 0.0)):.2f}，"
                        f"{rep.get('selection_reason','')}"
                    )
            lines.append(
                "代表股只是研究入口，不是买入推荐；后续深度研究必须重新验证。"
            )
            context_map = {
                str(item.get("ticker")): str(item.get("research_context") or "")
                for item in reps
                if item.get("ticker")
            }
            return "\n".join(lines), {"representative_contexts": context_map}

        result = run_discovery(
            cutoff,
            top_n=top_n,
            ml_model_path=self.config.get("sector_ml_model_path") or None,
            ml_weight=float(self.config.get("sector_ml_weight", 0.5)),
        )
        rows = result.sectors.sectors.head(top_n).to_dict(orient="records")
        lines = [
            f"截至{cutoff}，市场状态为{result.market.regime}（{result.market.score:.1f}/100）。",
            f"行业研究优先级 Top{len(rows)}（{result.metadata.get('rank_source', 'rule')}）：",
        ]
        for idx, row in enumerate(rows, start=1):
            lines.append(
                f"{idx}. {row.get('sector_code','')} {row.get('sector_name','')}，"
                f"综合分={float(row.get('sector_score', 0.0)):.2f}，"
                f"Style={row.get('style_profile') or row.get('primary_style') or 'N/A'}"
            )
        lines.append("以上是研究优先级，不是个股买入清单。")
        return "\n".join(lines), {}

    def _run_skill(
        self,
        action: SupervisorAction,
        *,
        message: str,
        ticker: str | None,
        cutoff: str,
        thread: dict[str, Any],
    ) -> tuple[str, str, dict[str, Any] | None]:
        target = str(action.target or "")
        if target == "sector_discovery":
            answer, metadata = self._run_sector_skill(message, cutoff)
            return answer, "skill:sector_discovery", {"metadata": metadata}

        if target == "document_evidence_analysis":
            if not ticker:
                return "文档证据分析需要先指定股票代码。", "skill:document_evidence_analysis", None
            result = self._invoke_atomic_tool(
                "search_company_knowledge",
                {
                    "query": action.objective or message,
                    "top_k": int(action.arguments.get("top_k", 6) or 6),
                },
                ticker=ticker,
                as_of_date=cutoff,
            )
            answer = self._synthesize(
                message=message,
                evidence=result.content,
                ticker=ticker,
                as_of_date=cutoff,
                research_context=str(thread.get("research_context", "") or ""),
            )
            return answer, "skill:document_evidence_analysis", None

        if target == "deep_stock_research":
            if not ticker:
                return "完整研究需要先指定股票代码。", "skill:deep_stock_research", None
            answer, state, signal, research_context = self._run_deep_research(
                ticker=ticker,
                cutoff=cutoff,
                thread=thread,
            )
            payload = {
                "ticker": ticker,
                "as_of_date": cutoff,
                "research_context": research_context,
                "final_trade_decision": state.get("final_trade_decision", ""),
                "audit_report": state.get("audit_report", ""),
                "audit_status": state.get("audit_status", ""),
                "signal": signal,
            }
            return answer, "skill:deep_stock_research", payload

        if target == "company_comparison":
            tickers = [
                str(item)
                for item in action.arguments.get("tickers", [])
                if str(item).strip()
            ][:3]
            if len(tickers) < 2:
                return (
                    "公司比较 Skill 需要 arguments.tickers 至少提供两个股票代码。",
                    "skill:company_comparison",
                    None,
                )
            evidence = []
            for item in tickers:
                result = self.specialists.run(
                    "fundamentals",
                    ticker=item,
                    as_of_date=cutoff,
                    objective=action.objective or message,
                )
                evidence.append(f"## {item}\n{result.content}")
            answer = self._synthesize(
                message=message,
                evidence="\n\n".join(evidence),
                ticker=None,
                as_of_date=cutoff,
                research_context="",
            )
            return answer, "skill:company_comparison", None

        return f"未知 Skill: {target}", f"skill:{target}", None

    def _execute_action(
        self,
        action: SupervisorAction,
        *,
        message: str,
        tid: str,
        ticker: str | None,
        cutoff: str,
        history: list[dict[str, str]],
        thread: dict[str, Any],
    ) -> tuple[str, str, dict[str, Any] | None]:
        research_context = str(thread.get("research_context", "") or "")

        if action.action == "rollback":
            restored = self.store.rollback_research_version(tid)
            if restored is None:
                return "当前会话没有可回滚的上一版研究结果。", "rollback", None
            payload = restored.get("payload", {})
            decision = str(payload.get("final_trade_decision", "") or "")
            answer = (
                f"已回滚到研究版本 V{restored.get('id')}。"
                + (f"\n\n{decision}" if decision else "")
            )
            return answer, "rollback", None

        if action.action == "respond":
            if action.answer:
                return str(action.answer), "respond", None
            prompt = f"""{self._system_prompt(
                ticker=ticker,
                as_of_date=cutoff,
                research_context=research_context,
            )}
用户问题：{message}
只解释已有已审计上下文，不新增任何未验证金融事实。
"""
            return str(self.llm.invoke(prompt).content or ""), "respond", None

        if action.action == "call_tool":
            if not action.target or action.target == "auto":
                return (
                    self._tool_chat(
                        message=message,
                        history=history,
                        ticker=ticker,
                        as_of_date=cutoff,
                        research_context=research_context,
                    ),
                    "tool:auto",
                    None,
                )
            result = self._invoke_atomic_tool(
                action.target,
                action.arguments,
                ticker=ticker,
                as_of_date=cutoff,
            )
            answer = self._synthesize(
                message=message,
                evidence=result.content,
                ticker=ticker,
                as_of_date=cutoff,
                research_context=research_context,
            )
            return answer, f"tool:{action.target}", None

        if action.action == "delegate_agent":
            if not ticker:
                return "专业 Agent 分析需要先指定股票代码。", "agent:missing_ticker", None
            target = str(action.target or "").removesuffix("_agent")
            result = self.specialists.run(
                target,
                ticker=ticker,
                as_of_date=cutoff,
                objective=action.objective or message,
                past_context="",
            )
            answer = self._synthesize(
                message=message,
                evidence=result.content,
                ticker=ticker,
                as_of_date=cutoff,
                research_context=research_context,
            )
            return answer, f"agent:{target}", None

        if action.action == "run_deep_research":
            action.target = "deep_stock_research"
            return self._run_skill(
                action,
                message=message,
                ticker=ticker,
                cutoff=cutoff,
                thread=thread,
            )

        if action.action == "run_skill":
            return self._run_skill(
                action,
                message=message,
                ticker=ticker,
                cutoff=cutoff,
                thread=thread,
            )

        return "无法执行当前 Supervisor 动作。", "failed", None

    def chat(
        self,
        message: str,
        *,
        thread_id: str | None = None,
        ticker: str | None = None,
        as_of_date: str | None = None,
        force_mode: str = "auto",
    ) -> dict[str, Any]:
        tid = self.store.ensure_thread(
            thread_id,
            current_ticker=ticker,
            as_of_date=as_of_date,
        )
        thread = self.store.get_thread(tid) or {}
        cutoff = as_of_date or thread.get("as_of_date") or date.today().isoformat()
        resolved_ticker = ticker or extract_ticker(message) or thread.get("current_ticker")
        self.store.update_context(
            tid,
            current_ticker=resolved_ticker,
            as_of_date=cutoff,
        )
        self.store.append_message(tid, "user", message)

        history = self.store.history(
            tid,
            limit=int(self.config.get("conversation_history_turns", 12)),
        )
        thread = self.store.get_thread(tid) or thread
        action = self.supervisor.decide(
            message,
            current_ticker=resolved_ticker,
            as_of_date=cutoff,
            history=history[:-1],
            research_context=str(thread.get("research_context", "") or ""),
            force_mode=force_mode,
        )

        answer, route, version_payload = self._execute_action(
            action,
            message=message,
            tid=tid,
            ticker=resolved_ticker,
            cutoff=cutoff,
            history=history[:-1],
            thread=thread,
        )

        metadata: dict[str, Any] = {
            "supervisor_action": action.action,
            "supervisor_target": action.target,
        }
        if version_payload:
            version_id = self.store.save_research_version(
                tid,
                version_payload,
                audit_status=str(version_payload.get("audit_status", "") or ""),
            )
            metadata["active_research_version_id"] = version_id
            self.store.update_context(
                tid,
                current_ticker=resolved_ticker,
                as_of_date=cutoff,
                last_intent=route,
                research_context=str(
                    version_payload.get("research_context", "") or ""
                ),
                metadata=metadata,
            )
        else:
            skill_metadata = None
            if route == "skill:sector_discovery":
                _, skill_metadata = self._run_sector_skill(message, cutoff)
            if skill_metadata:
                metadata.update(skill_metadata)
            self.store.update_context(
                tid,
                current_ticker=resolved_ticker,
                as_of_date=cutoff,
                last_intent=route,
                metadata=metadata,
            )

        self.store.append_message(tid, "assistant", answer)
        updated = self.store.get_thread(tid) or {}
        return {
            "thread_id": tid,
            "route": route,
            "supervisor_action": action.action,
            "supervisor_target": action.target,
            "ticker": updated.get("current_ticker") or resolved_ticker,
            "as_of_date": updated.get("as_of_date") or cutoff,
            "answer": answer,
            "active_research_version_id": (
                updated.get("metadata", {}).get("active_research_version_id")
                if isinstance(updated.get("metadata"), dict)
                else None
            ),
        }
