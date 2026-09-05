from __future__ import annotations

import logging
from typing import Any

from tradingagents.agents.utils.structured import bind_structured
from tradingagents.capabilities.registry import CapabilityRegistry
from tradingagents.conversation.router import route_message
from tradingagents.orchestration.schemas import SupervisorAction

logger = logging.getLogger(__name__)


class ConversationSupervisor:
    """LLM-first router with a deterministic fallback."""

    def __init__(self, llm: Any, registry: CapabilityRegistry):
        self.llm = llm
        self.registry = registry
        self.structured_llm = bind_structured(
            llm, SupervisorAction, "Conversation Supervisor"
        )

    def _fallback(
        self,
        message: str,
        *,
        current_ticker: str | None,
        force_mode: str = "auto",
    ) -> SupervisorAction:
        route = route_message(
            message,
            current_ticker=current_ticker,
            force_mode=force_mode,
        )
        if route.intent == "research":
            return SupervisorAction(
                action="run_deep_research",
                target="deep_stock_research",
                objective=message,
            )
        if route.intent == "discovery":
            return SupervisorAction(
                action="run_skill",
                target="sector_discovery",
                objective=message,
            )
        return SupervisorAction(
            action="call_tool",
            target="auto",
            objective=message,
        )

    def decide(
        self,
        message: str,
        *,
        current_ticker: str | None,
        as_of_date: str,
        history: list[dict[str, str]],
        research_context: str = "",
        force_mode: str = "auto",
    ) -> SupervisorAction:
        if force_mode != "auto":
            return self._fallback(
                message,
                current_ticker=current_ticker,
                force_mode=force_mode,
            )

        normalized = (message or "").strip().lower()
        if any(
            word in normalized
            for word in ("回滚", "撤销上一版", "恢复上一版", "rollback")
        ):
            return SupervisorAction(action="rollback", objective=message)

        history_text = "\n".join(
            f"{item.get('role', 'unknown')}: {str(item.get('content', ''))[:800]}"
            for item in history[-8:]
        ) or "无"
        context = (research_context or "")[:5000]
        prompt = f"""
你是 A 股投研系统的 Conversation Supervisor。你的职责是根据用户目标选择最小、最合适的执行能力，
而不是亲自编造金融事实。当前研究截止日期：{as_of_date}；当前标的：{current_ticker or '未指定'}。

可用能力：
{self.registry.prompt_catalog()}

路由原则：
1. 简单事实查询优先 call_tool；不要为一个价格、指标或单份公告运行完整多智能体研究。
2. 单一专业领域的复杂分析优先 delegate_agent 到 market/news/fundamentals。
3. 只有综合投资价值、完整研报、跨多类证据的复杂请求才 run_deep_research。
4. 行业发现、文档证据分析等可复用任务优先 run_skill。
5. 用户明确要求撤销/恢复上一版研究时 rollback。
6. 如果只是解释上一轮已经审计的研究上下文且无需新事实，可 respond。
7. 任何需要事实/数字的新回答都不要直接 respond，必须通过工具、Agent 或 Skill 获取证据。
8. 不要把行业排名、候选池分数直接当成投资事实。
9. 所有历史事实必须满足 PIT 截止日期约束。

最近对话：
{history_text}

上一轮已审计研究上下文：
{context or '无'}

用户当前请求：
{message}

只返回一个结构化动作。objective 说明要完成的任务；arguments 只放执行所需参数。
""".strip()

        if self.structured_llm is None:
            return self._fallback(message, current_ticker=current_ticker)
        try:
            action = self.structured_llm.invoke(prompt)
            if action is None:
                raise ValueError("supervisor returned no structured action")
            if action.target and action.action in {
                "call_tool",
                "delegate_agent",
                "run_skill",
            }:
                target = action.target
                if action.action == "delegate_agent":
                    target = target.removesuffix("_agent")
                    action.target = target
                if self.registry.get(target) is None and target != "auto":
                    raise ValueError(f"unknown capability target: {target}")
            return action
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Supervisor structured routing failed; using fallback: %s",
                exc,
            )
            return self._fallback(message, current_ticker=current_ticker)
