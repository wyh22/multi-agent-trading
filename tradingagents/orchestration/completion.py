from __future__ import annotations

import logging
import re
from typing import Any

from tradingagents.agents.utils.structured import bind_structured
from tradingagents.orchestration.schemas import CompletionAssessment, TaskContract

logger = logging.getLogger(__name__)


_DIMENSION_HINTS = {
    "估值": "valuation",
    "现金流": "cash_flow",
    "增长": "growth",
    "成长": "growth",
    "技术": "technical",
    "走势": "technical",
    "政策": "policy",
    "风险": "risk",
    "公告": "announcement",
    "新闻": "news",
    "基本面": "fundamentals",
    "财务": "fundamentals",
    "盈利": "profitability",
}


def _fallback_contract(message: str, ticker: str | None) -> TaskContract:
    dimensions = [
        label
        for key, label in _DIMENSION_HINTS.items()
        if key in (message or "")
    ]
    dimensions = list(dict.fromkeys(dimensions))
    entities = []
    for item in re.findall(r"\b\d{6}(?:\.(?:SH|SZ|BJ))?\b", message or "", flags=re.I):
        entities.append(item.upper())
    if ticker and ticker not in entities:
        entities.insert(0, ticker)
    return TaskContract(
        objective=(message or "").strip(),
        required_dimensions=dimensions,
        required_entities=list(dict.fromkeys(entities)),
        critical_requirements=list(dimensions[:2]),
        expected_output="research_answer",
        can_be_partial=True,
    )


class TaskContractBuilder:
    """Convert a free-form user request into an explicit completion contract."""

    def __init__(self, llm: Any):
        self.structured_llm = bind_structured(
            llm,
            TaskContract,
            "Task Contract Builder",
        )

    def build(
        self,
        message: str,
        *,
        ticker: str | None,
        as_of_date: str,
    ) -> TaskContract:
        if self.structured_llm is None:
            return _fallback_contract(message, ticker)

        prompt = f"""
你负责把用户投研请求转换成可验证的 Task Contract，而不是回答问题。
研究截止日：{as_of_date}
当前标的：{ticker or '未指定'}

用户请求：
{message}

要求：
1. required_dimensions 只列用户明确要求或完成请求不可缺少的研究维度；
2. required_entities 列需要分别覆盖的公司/指数/实体；不要凭空扩展；
3. critical_requirements 是缺失后不能声称“完整回答”的项目；
4. 简单单点事实问题可以只有一个 dimension；
5. 不要把工具名、Agent 名写进 contract；
6. expected_output 用简短英文标签；
7. can_be_partial 通常为 true，除非用户明确要求必须完整后才回答。
""".strip()
        try:
            result = self.structured_llm.invoke(prompt)
            if result is None:
                raise ValueError("empty task contract")
            if not result.objective:
                result.objective = message
            return self._sanitize(contract, result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Task contract generation failed; using fallback: %s", exc)
            return _fallback_contract(message, ticker)


class CompletionGate:
    """Explicitly decide whether gathered evidence satisfies the Task Contract."""

    def __init__(self, llm: Any):
        self.structured_llm = bind_structured(
            llm,
            CompletionAssessment,
            "Task Completion Gate",
        )

    @staticmethod
    def _fallback(
        contract: TaskContract,
        observations: list[str],
    ) -> CompletionAssessment:
        checklist = contract.checklist()
        if not checklist:
            complete = bool(observations)
            return CompletionAssessment(
                complete=complete,
                completion_ratio=1.0 if complete else 0.0,
                completed_items=[],
                missing_items=[] if complete else ["evidence"],
                critical_missing=[] if complete else list(contract.critical_requirements),
                reason="deterministic fallback",
            )

        text = "\n".join(observations).lower()
        completed = []
        missing = []
        for item in checklist:
            normalized = item.split("::")[-1].replace("_", " ").lower()
            if normalized and normalized in text:
                completed.append(item)
            else:
                missing.append(item)
        ratio = len(completed) / max(1, len(checklist))
        critical_missing = [
            item for item in contract.critical_requirements
            if not any(item.lower() in done.lower() for done in completed)
        ]
        return CompletionAssessment(
            complete=not missing and not critical_missing,
            completion_ratio=ratio,
            completed_items=completed,
            missing_items=missing,
            critical_missing=critical_missing,
            reason="deterministic fallback",
        )

    @staticmethod
    def _sanitize(
        contract: TaskContract,
        result: CompletionAssessment,
    ) -> CompletionAssessment:
        """Constrain completion bookkeeping to the immutable Task Contract."""

        checklist = contract.checklist()
        if not checklist:
            return result

        def map_items(values: list[str]) -> tuple[list[str], list[str]]:
            mapped: list[str] = []
            unmatched: list[str] = []
            for raw in values:
                text = str(raw or "").strip()
                lower = text.lower()
                match = None
                for item in checklist:
                    if lower == item.lower():
                        match = item
                        break
                    if "::" in item:
                        entity, dimension = item.split("::", 1)
                        dim_tokens = {
                            dimension.lower(),
                            dimension.replace("_", " ").lower(),
                        }
                        if any(token and token in lower for token in dim_tokens):
                            if entity.lower() in lower or entity not in text:
                                match = item
                                break
                    elif item.lower() in lower:
                        match = item
                        break
                if match is None:
                    unmatched.append(text)
                elif match not in mapped:
                    mapped.append(match)
            return mapped, unmatched

        completed, unmatched_completed = map_items(result.completed_items)
        missing, unmatched_missing = map_items(result.missing_items)
        missing = [item for item in missing if item not in completed]
        for item in checklist:
            if item not in completed and item not in missing:
                missing.append(item)

        critical_missing = [
            requirement
            for requirement in contract.critical_requirements
            if not any(
                requirement.lower() in item.lower()
                for item in completed
            )
        ]
        evidence_gaps = list(
            dict.fromkeys(
                [
                    *result.evidence_gaps,
                    *unmatched_missing,
                    *unmatched_completed,
                ]
            )
        )
        ratio = len(completed) / max(1, len(checklist))
        return CompletionAssessment(
            complete=not missing and not critical_missing,
            completion_ratio=ratio,
            completed_items=completed,
            missing_items=missing,
            critical_missing=critical_missing,
            evidence_gaps=evidence_gaps,
            reason=result.reason,
        )

    def assess(
        self,
        contract: TaskContract,
        *,
        observations: list[str],
        used_capabilities: list[str],
    ) -> CompletionAssessment:
        if self.structured_llm is None:
            return self._fallback(contract, observations)

        prompt = f"""
你是投研任务完成度检查器。只判断“用户要求是否已经被证据覆盖”，不要新增事实。

Task Contract:
{contract.model_dump_json(indent=2)}

已使用 capability:
{used_capabilities or ['无']}

已获得结果:
{chr(10).join(observations[-4:])[:18000] or '无'}

规则：
1. complete=true 只有在所有 critical_requirements 都已覆盖时；
2. 不要因为模型“提到了一个主题”就视为完成，要有实际结果/证据；
3. completed_items / missing_items 只能填写 Task Contract checklist 中的原始条目，不得新增维度；
4. 更具体的“缺什么原文/来源/核验”只能写入 evidence_gaps；
5. completion_ratio 按 contract 覆盖度估算，不能因篇幅长就提高；
6. 如果数据源明确不可用，应保留为 missing/evidence gap，而不是假装完成。
""".strip()
        try:
            result = self.structured_llm.invoke(prompt)
            if result is None:
                raise ValueError("empty completion assessment")
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Completion assessment failed; using fallback: %s", exc)
            return self._fallback(contract, observations)
