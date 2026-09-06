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

    @staticmethod
    def _normalize_target(action: SupervisorAction) -> SupervisorAction:
        """Normalize catalog-style targets such as agent:market.

        The capability catalog is rendered as kind:name for readability,
        while the execution registry stores the bare capability name. Some LLMs
        copy the catalog token verbatim into target. Treat that as a valid
        structured action instead of unnecessarily falling back.
        """

        if not action.target:
            return action

        target = str(action.target).strip()
        expected_kind = {
            "call_tool": "tool",
            "delegate_agent": "agent",
            "run_skill": "skill",
        }.get(action.action)

        if ":" in target:
            prefix, bare = target.split(":", 1)
            prefix = prefix.strip().lower()
            if prefix in {"tool", "agent", "skill"}:
                if expected_kind is not None and prefix != expected_kind:
                    raise ValueError(
                        "capability kind mismatch: "
                        f"action={action.action}, target={target}"
                    )
                target = bare.strip()

        if action.action == "delegate_agent":
            target = target.removesuffix("_agent")

        action.target = target
        return action

    def _fallback(
        self,
        message: str,
        *,
        current_ticker: str | None,
        force_mode: str = "auto",
        repair_mode: bool = False,
        used_capabilities: list[str] | None = None,
    ) -> SupervisorAction:
        route = route_message(
            message,
            current_ticker=current_ticker,
            force_mode=force_mode,
        )
        if route.intent == "discovery":
            return SupervisorAction(
                action="run_skill",
                target="sector_discovery",
                objective=message,
            )

        normalized = (message or "").strip().lower()
        full_research_words = (
            "深度分析",
            "深度研究",
            "完整分析",
            "完整研究",
            "投研报告",
            "研报",
            "全面分析",
            "重新分析",
            "重新研究",
        )
        if any(word in normalized for word in full_research_words):
            return SupervisorAction(
                action="run_deep_research",
                target="deep_stock_research",
                objective=message,
            )

        ticker = route.ticker or current_ticker
        if ticker:
            domain_words = {
                "market": (
                    "技术趋势",
                    "技术面",
                    "动量",
                    "均线",
                    "macd",
                    "rsi",
                    "支撑位",
                    "压力位",
                    "成交量",
                    "波动率",
                    "技术风险",
                ),
                "fundamentals": (
                    "基本面",
                    "现金流",
                    "盈利质量",
                    "盈利能力",
                    "营收",
                    "利润",
                    "毛利率",
                    "净利率",
                    "roe",
                    "资产负债",
                    "偿债",
                    "估值",
                    "市盈率",
                    "市净率",
                    "主营业务",
                    "业务与经营",
                    "装机容量",
                    "发电量",
                    "利用小时",
                    "运营规模",
                ),
                "news": (
                    "公告",
                    "新闻",
                    "政策",
                    "舆情",
                    "监管",
                    "事件风险",
                    "宏观",
                    "弃风",
                    "限电",
                    "补贴",
                    "国补",
                    "规划",
                ),
            }
            matched = [
                name
                for name, words in domain_words.items()
                if any(word in normalized for word in words)
            ]
            if repair_mode and matched:
                used = set(used_capabilities or [])
                for name in matched:
                    if (
                        self.registry.get(name) is not None
                        and f"delegate_agent:{name}" not in used
                    ):
                        return SupervisorAction(
                            action="delegate_agent",
                            target=name,
                            objective=message,
                        )
            if len(matched) == 1 and self.registry.get(matched[0]) is not None:
                return SupervisorAction(
                    action="delegate_agent",
                    target=matched[0],
                    objective=message,
                )
            if len(matched) >= 2:
                return SupervisorAction(
                    action="run_deep_research",
                    target="deep_stock_research",
                    objective=message,
                )

        if route.intent == "research":
            return SupervisorAction(
                action="run_deep_research",
                target="deep_stock_research",
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
        observations: list[str] | None = None,
        used_capabilities: list[str] | None = None,
        repair_mode: bool = False,
    ) -> SupervisorAction:
        if force_mode != "auto":
            return self._fallback(
                message,
                current_ticker=current_ticker,
                force_mode=force_mode,
                repair_mode=repair_mode,
                used_capabilities=used_capabilities,
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
        observation_text = "\n\n".join((observations or [])[-3:])[:12000] or "无"
        used_text = ", ".join(used_capabilities or []) or "无"
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
10. 如果“本轮已获得结果”已经足够回答，就 respond；否则可以选择一个尚未使用的互补 Tool/Agent。
11. 不要重复调用同一个 capability，除非上一次明确返回 retryable 错误。
12. 如果这是上一轮 PARTIAL/REVIEW_REQUIRED 的补查（repair_mode=true），禁止重新运行完整 deep_stock_research；
    应优先选择尚未使用的 specialist Agent 或可用的文档证据能力，只补缺口。

repair_mode：{"true" if repair_mode else "false"}

本轮已使用 capability：
{used_text}

本轮已获得结果：
{observation_text}

最近对话：
{history_text}

上一轮已审计研究上下文：
{context or '无'}

用户当前请求：
{message}

只返回一个结构化动作。objective 说明要完成的任务；arguments 只放执行所需参数。\ntarget 必须只填写 capability 的裸名称，不要带 tool:/agent:/skill: 前缀。\n例如 target 应写 get_verified_market_snapshot、market、sector_discovery。
""".strip()

        if self.structured_llm is None:
            return self._fallback(
                message,
                current_ticker=current_ticker,
                repair_mode=repair_mode,
                used_capabilities=used_capabilities,
            )
        try:
            action = self.structured_llm.invoke(prompt)
            if action is None:
                raise ValueError("supervisor returned no structured action")
            action = self._normalize_target(action)
            if repair_mode and action.action == "run_deep_research":
                raise ValueError(
                    "repair mode must not rerun deep_stock_research"
                )
            if action.target and action.action in {
                "call_tool",
                "delegate_agent",
                "run_skill",
            }:
                target = action.target
                if self.registry.get(target) is None and target != "auto":
                    raise ValueError(f"unknown capability target: {target}")
            return action
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Supervisor structured routing failed; using fallback: %s",
                exc,
            )
            return self._fallback(message, current_ticker=current_ticker)
