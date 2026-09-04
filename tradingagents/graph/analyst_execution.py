"""并行分析师执行计划与耗时统计。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True)
class AnalystNodeSpec:
    """单个分析师在父图中的静态配置。"""

    key: str
    agent_node: str
    tool_node: str
    report_key: str


@dataclass(frozen=True)
class AnalystExecutionPlan:
    """分析师并行执行计划。"""

    specs: list[AnalystNodeSpec]
    parallel: bool = True


ANALYST_NODE_SPECS: dict[str, AnalystNodeSpec] = {
    "market": AnalystNodeSpec(
        key="market",
        agent_node="Market Analyst",
        tool_node="tools_market",
        report_key="market_report",
    ),
    "news": AnalystNodeSpec(
        key="news",
        agent_node="News & Sentiment Analyst",
        tool_node="tools_news",
        report_key="news_report",
    ),
    "fundamentals": AnalystNodeSpec(
        key="fundamentals",
        agent_node="Fundamentals Analyst",
        tool_node="tools_fundamentals",
        report_key="fundamentals_report",
    ),
}


def _normalize_analyst_key(value) -> str:
    """兼容 Enum、字符串以及旧版 social 配置。"""

    key = getattr(value, "value", value)
    key = str(key).lower()
    return "news" if key == "social" else key


def build_analyst_execution_plan(
    selected_analysts: Iterable[str],
) -> AnalystExecutionPlan:
    """根据用户选择生成去重后的分析师执行计划。"""

    specs: list[AnalystNodeSpec] = []
    seen: set[str] = set()
    for raw_key in selected_analysts:
        analyst_key = _normalize_analyst_key(raw_key)
        if analyst_key in seen:
            continue
        spec = ANALYST_NODE_SPECS.get(analyst_key)
        if spec is None:
            raise ValueError(f"未知分析师类型: {analyst_key}")
        seen.add(analyst_key)
        specs.append(spec)

    if not specs:
        raise ValueError("至少需要选择一名分析师")
    return AnalystExecutionPlan(specs=specs)


def get_initial_analyst_node(plan: AnalystExecutionPlan) -> str:
    """返回 CLI 启动阶段用于展示的首个分析师节点。

    父图实际会并行 fan-out 到全部已选分析师；这个兼容辅助函数只负责
    CLI 在第一批 LangGraph 增量到达前的进度展示，不改变执行拓扑。
    """

    if not plan.specs:
        raise ValueError("分析师执行计划不能为空")
    return plan.specs[0].agent_node


class AnalystWallTimeTracker:
    """记录并行分析师的墙钟耗时。"""

    def __init__(self, plan: AnalystExecutionPlan):
        self.plan = plan
        self._started_at: dict[str, float] = {}
        self._wall_times: dict[str, float] = {}

    def mark_started(self, analyst_key: str, started_at: float | None = None) -> None:
        if analyst_key not in ANALYST_NODE_SPECS:
            raise ValueError(f"未知分析师类型: {analyst_key}")
        self._started_at.setdefault(
            analyst_key,
            monotonic() if started_at is None else started_at,
        )

    def mark_completed(
        self,
        analyst_key: str,
        completed_at: float | None = None,
    ) -> None:
        if analyst_key not in ANALYST_NODE_SPECS:
            raise ValueError(f"未知分析师类型: {analyst_key}")
        if analyst_key in self._wall_times:
            return
        started_at = self._started_at.get(analyst_key)
        if started_at is None:
            return
        finished_at = monotonic() if completed_at is None else completed_at
        self._wall_times[analyst_key] = max(0.0, finished_at - started_at)

    def get_wall_times(self) -> dict[str, float]:
        return dict(self._wall_times)

    def format_summary(self) -> str:
        parts = []
        for spec in self.plan.specs:
            duration = self._wall_times.get(spec.key)
            if duration is not None:
                label = spec.agent_node.replace(" Analyst", "")
                parts.append(f"{label} {duration:.2f}s")
        if not parts:
            return "分析师耗时：等待中"
        return "分析师耗时：" + " | ".join(parts)


def sync_analyst_tracker_from_chunk(
    tracker: AnalystWallTimeTracker,
    chunk: dict[str, str],
    now: float | None = None,
) -> None:
    """根据 LangGraph 流式增量同步分析师开始/完成时间。"""

    current_time = monotonic() if now is None else now
    first_update = not tracker._wall_times and not tracker._started_at

    for spec in tracker.plan.specs:
        if tracker.plan.parallel and first_update and spec.key not in tracker._started_at:
            tracker.mark_started(spec.key, started_at=current_time)

        if chunk.get(spec.report_key) and spec.key not in tracker._wall_times:
            if spec.key not in tracker._started_at:
                tracker.mark_started(spec.key, started_at=current_time)
            tracker.mark_completed(spec.key, completed_at=current_time)
