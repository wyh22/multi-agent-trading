from __future__ import annotations

import json
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tradingagents.agents.utils.tool_registry import build_tool_groups
from tradingagents.conversation.router import route_message
from tradingagents.conversation.store import ConversationStore
from tradingagents.discovery.pipeline import run_discovery
from tradingagents.graph.trading_graph import TradingAgentsGraph, _coerce_max_retries
from tradingagents.llm_clients import create_llm_client


class ConversationAgent:
    """Conversational application layer on top of the existing research graph."""

    def __init__(self, config: dict[str, Any], store: ConversationStore):
        self.config = config
        self.store = store
        self.llm = self._create_quick_llm()
        self.tools = self._build_tools()
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools) if self.tools else self.llm

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
        groups = build_tool_groups(self.config)
        merged = []
        seen = set()
        for group in ("market", "news", "fundamentals"):
            for tool in groups.get(group, []):
                if tool.name not in seen:
                    merged.append(tool)
                    seen.add(tool.name)


        if self.config.get("external_mcp_enabled", False):
            raw = str(self.config.get("external_mcp_servers_json", "") or "").strip()
            if raw:
                try:
                    servers = json.loads(raw)
                    if not isinstance(servers, dict):
                        raise ValueError("external_mcp_servers_json必须是JSON对象")
                    from tradingagents.mcp.client import load_mcp_tools_sync
                    allowlist = {
                        item.strip() for item in str(self.config.get("external_mcp_tool_allowlist", "") or "").split(",")
                        if item.strip()
                    }
                    for tool in load_mcp_tools_sync(servers=servers):
                        if tool.name in allowlist and tool.name not in seen:
                            merged.append(tool)
                            seen.add(tool.name)
                except Exception as exc:  # noqa: BLE001
                    import logging
                    logging.getLogger(__name__).warning("外部MCP工具加载失败，将继续使用核心工具: %s", exc)
        return merged

    @staticmethod
    def _system_prompt(*, ticker: str | None, as_of_date: str, research_context: str) -> str:
        context = research_context.strip()
        if len(context) > 18000:
            context = context[:18000] + "\n...[历史研究上下文已截断]"
        return f"""你是A股研究对话助手。当前研究截止日期为 {as_of_date}。
当前会话标的：{ticker or '未指定'}。

规则：
1. 涉及行情、财务、公告、宏观或知识库事实时优先调用工具，不得凭空补数字。
2. 所有历史事实必须遵守研究截止日期，不得使用截止日之后的信息。
3. 如果用户只是在追问上一轮研究结论，优先复用下面的已审计研究上下文，而不是重新运行完整7-Agent。
4. 工具返回不可用时明确说明数据缺失；不要伪造来源。
5. 本系统只提供研究辅助，不自动执行证券交易；不得声称已经替用户下单。
6. 回答尽量直接，并区分事实、程序计算结果、模型推断与条件性判断。

上一轮研究上下文（可能为空）：
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
        text = "\n\n".join(sections)
        return text[:30000]

    def _tool_chat(self, *, message: str, history: list[dict[str, str]], ticker: str | None, as_of_date: str, research_context: str) -> str:
        messages: list[Any] = [SystemMessage(content=self._system_prompt(ticker=ticker, as_of_date=as_of_date, research_context=research_context))]
        for item in history[-10:]:
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=message))

        max_rounds = max(1, int(self.config.get("conversation_tool_rounds", 4)))
        for _ in range(max_rounds):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                return str(response.content or "")
            for call in tool_calls:
                name = call.get("name")
                tool = self.tools_by_name.get(name)
                if tool is None:
                    result = f"TOOL_UNAVAILABLE: {name}"
                else:
                    args = dict(call.get("args") or {})
                    if ticker:
                        if name in {"get_stock_data", "get_indicators", "get_verified_market_snapshot"}:
                            args.setdefault("symbol", ticker)
                        elif name in {"get_fundamentals", "get_balance_sheet", "get_cashflow", "get_income_statement",
                                     "get_news", "get_insider_transactions", "search_company_knowledge"}:
                            args.setdefault("ticker", ticker)
                    if name in {"get_fundamentals", "get_insider_transactions", "get_global_news", "get_macro_indicators"}:
                        args.setdefault("curr_date", as_of_date)
                    if name in {"get_balance_sheet", "get_cashflow", "get_income_statement"}:
                        args.setdefault("curr_date", as_of_date)
                    if name == "search_company_knowledge":
                        args.setdefault("as_of_date", as_of_date)
                    if name == "get_news":
                        args.setdefault("end_date", as_of_date)
                    try:
                        result = tool.invoke(args)
                    except Exception as exc:  # noqa: BLE001
                        result = f"TOOL_ERROR: {type(exc).__name__}: {exc}"
                text = str(result)
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
            + [HumanMessage(content="工具调用轮次已达到上限。请基于已有工具结果给出最终回答，并明确仍缺少的信息。")]
        )
        return str(final.content or "")

    def chat(
        self,
        message: str,
        *,
        thread_id: str | None = None,
        ticker: str | None = None,
        as_of_date: str | None = None,
        force_mode: str = "auto",
    ) -> dict[str, Any]:
        tid = self.store.ensure_thread(thread_id, current_ticker=ticker, as_of_date=as_of_date)
        thread = self.store.get_thread(tid) or {}
        cutoff = as_of_date or thread.get("as_of_date") or date.today().isoformat()
        current_ticker = ticker or thread.get("current_ticker")
        self.store.update_context(tid, current_ticker=current_ticker, as_of_date=cutoff)
        route = route_message(message, current_ticker=current_ticker, force_mode=force_mode)
        resolved_ticker = route.ticker
        self.store.append_message(tid, "user", message)

        if route.intent == "research":
            if not resolved_ticker:
                answer = "要运行完整多Agent研究，请提供6位A股代码（例如600519.SH），或在请求的ticker字段中指定标的。"
            else:
                graph = TradingAgentsGraph(config=self.config)
                state, signal = graph.propagate(resolved_ticker, cutoff)
                research_context = self._research_context(state)
                answer = str(state.get("final_trade_decision", "") or "")
                audit = str(state.get("audit_report", "") or "")
                if audit:
                    answer += f"\n\n---\n**结果审计**\n{audit}"
                self.store.update_context(
                    tid,
                    current_ticker=resolved_ticker,
                    as_of_date=cutoff,
                    last_intent="research",
                    research_context=research_context,
                    metadata={"signal": signal, "audit_status": state.get("audit_status", "")},
                )
        elif route.intent == "discovery":
            top_n = int(self.config.get("sector_discovery_top_n", 6))
            result = run_discovery(
                cutoff,
                top_n=top_n,
                ml_model_path=self.config.get("sector_ml_model_path") or None,
                ml_weight=float(self.config.get("sector_ml_weight", 0.5)),
            )
            rows = result.sectors.sectors.head(top_n).to_dict(orient="records")
            lines = [
                f"截至{cutoff}，市场状态为{result.market.regime}（{result.market.score:.1f}/100）。",
                (
                    "行业研究优先级 Top"
                    f"{len(rows)}（{result.metadata.get('rank_source', 'rule')}）："
                ),
            ]
            for idx, row in enumerate(rows, start=1):
                code = str(row.get("sector_code") or "")
                name = str(row.get("sector_name") or "")
                style = str(row.get("style_profile") or row.get("primary_style") or "")
                score = row.get("sector_score")
                score_text = (
                    f"，综合分{float(score):.2f}"
                    if isinstance(score, (int, float))
                    else ""
                )
                style_text = f"，Style={style}" if style else ""
                lines.append(
                    f"{idx}. {code} {name}{score_text}{style_text}".strip()
                )
            lines.append(
                "这些是行业研究优先级，不是个股买入清单；"
                "下一步应从目标行业中选择代表性股票进入7-Agent深度研究。"
            )
            answer = "\n".join(lines)
            self.store.update_context(
                tid,
                as_of_date=cutoff,
                last_intent="discovery",
            )
        else:
            thread = self.store.get_thread(tid) or {}
            history = self.store.history(tid, limit=int(self.config.get("conversation_history_turns", 12)))
            answer = self._tool_chat(
                message=message,
                history=history[:-1],
                ticker=resolved_ticker,
                as_of_date=cutoff,
                research_context=str(thread.get("research_context", "") or ""),
            )
            self.store.update_context(
                tid,
                current_ticker=resolved_ticker,
                as_of_date=cutoff,
                last_intent="tool_chat",
            )

        self.store.append_message(tid, "assistant", answer)
        updated = self.store.get_thread(tid) or {}
        return {
            "thread_id": tid,
            "route": route.intent,
            "ticker": updated.get("current_ticker") or resolved_ticker,
            "as_of_date": updated.get("as_of_date") or cutoff,
            "answer": answer,
        }
