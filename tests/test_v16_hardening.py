from datetime import date
from pathlib import Path

import pandas as pd

from tradingagents.agents.utils.context_compaction import compact_evidence_text
from tradingagents.capabilities.registry import CapabilityRegistry, CapabilitySpec
from tradingagents.conversation.agent import ConversationAgent
from tradingagents.discovery.representatives import select_representative_stocks
from tradingagents.discovery.style_adapter import load_adaptive_style_weights
from tradingagents.evaluation.grounding import evaluate_claim_grounding
from tradingagents.evaluation.routing import (
    RoutingEvalCase,
    RoutingRecord,
    evaluate_route,
    summarize_routing,
)
from tradingagents.orchestration.completion import CompletionGate, TaskContractBuilder
from tradingagents.orchestration.schemas import CompletionAssessment, TaskContract
from tradingagents.orchestration.supervisor import ConversationSupervisor
from tradingagents.rag.models import KnowledgeChunk
from tradingagents.rag.retriever import (
    HybridKnowledgeRetriever,
    InMemoryKnowledgeStore,
)


class NoStructuredLLM:
    def with_structured_output(self, _schema):
        raise NotImplementedError


def test_task_contract_and_completion_gate_have_deterministic_fallback():
    builder = TaskContractBuilder(NoStructuredLLM())
    contract = builder.build(
        "比较600519.SH的估值和现金流",
        ticker="600519.SH",
        as_of_date="2026-09-05",
    )
    assert "valuation" in contract.required_dimensions
    assert "cash_flow" in contract.required_dimensions
    assert "600519.SH" in contract.required_entities

    gate = CompletionGate(NoStructuredLLM())
    assessment = gate.assess(
        contract,
        observations=["600519.SH valuation 已覆盖；cash flow 已覆盖。"],
        used_capabilities=["delegate_agent:fundamentals"],
    )
    assert 0.0 <= assessment.completion_ratio <= 1.0


def test_dual_ledger_preserves_hypothesis_budget():
    text = """
- [FACT] 营业收入同比增长20%。
- [CALCULATION] 由经营现金流和净利润计算得到现金实现比。
- [INFERENCE] 收入增长可能部分来自渠道库存变化。
- [CONDITIONAL] 如果渠道库存继续上升，则未来收入质量可能承压。
"""
    compact = compact_evidence_text(
        text,
        source_section="fundamentals",
        max_chars=500,
    )
    assert "### Evidence Ledger" in compact
    assert "### Hypothesis Ledger" in compact
    assert "[FACT]" in compact
    assert "[INFERENCE]" in compact


def test_unverified_upload_is_excluded_from_historical_rag():
    unverified = KnowledgeChunk(
        chunk_id="u::0",
        doc_id="u",
        ticker="600519.SH",
        title="uploaded",
        text="渠道库存风险",
        publish_date="2026-04-01",
        metadata={"publish_date_verified": False},
    )
    verified = KnowledgeChunk(
        chunk_id="v::0",
        doc_id="v",
        ticker="600519.SH",
        title="verified",
        text="渠道库存风险",
        publish_date="2026-04-01",
        metadata={"publish_date_verified": True},
    )
    retriever = HybridKnowledgeRetriever(
        InMemoryKnowledgeStore([unverified, verified])
    )
    hits = retriever.search(
        "渠道库存",
        ticker="600519.SH",
        as_of_date="2026-05-01",
        top_k=5,
    )
    assert [hit.chunk.chunk_id for hit in hits] == ["v::0"]

    current_hits = retriever.search(
        "渠道库存",
        ticker="600519.SH",
        as_of_date=date.today().isoformat(),
        top_k=5,
    )
    assert {hit.chunk.chunk_id for hit in current_hits} == {"u::0", "v::0"}


def test_component_source_partial_response_fails_closed():
    sectors = pd.DataFrame(
        [
            {"sector_code": "A", "sector_name": "行业A", "sector_score": 80.0},
            {"sector_code": "B", "sector_name": "行业B", "sector_score": 70.0},
        ]
    )

    def component_fetcher(symbol: str):
        if symbol == "B":
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "证券代码": ["600000"],
                "证券名称": ["样例"],
                "最新权重": [10.0],
                "计入日期": ["2020-01-01"],
            }
        )

    result = select_representative_stocks(
        sectors,
        "2026-09-05",
        representatives_per_sector=1,
        component_limit=2,
        component_fetcher=component_fetcher,
        history_loader=lambda *_args, **_kwargs: {},
    )
    assert result.representatives.empty
    assert any(
        "COMPONENT_DATA_UNAVAILABLE" in warning
        for warning in result.warnings
    )


def test_routing_eval_reports_unnecessary_deep_research():
    case = RoutingEvalCase(
        case_id="simple_price",
        user_query="收盘价多少",
        expected_actions=["call_tool"],
        expected_targets=["get_stock_data"],
        forbidden_actions=["run_deep_research"],
    )
    bad = evaluate_route(
        case,
        RoutingRecord(
            action="run_deep_research",
            target="deep_stock_research",
        ),
    )
    assert bad.pass_case is False
    assert bad.unnecessary_deep_research is True
    summary = summarize_routing([bad])
    assert summary["route_accuracy"] == 0.0
    assert summary["unnecessary_deep_research_rate"] == 1.0


def test_claim_level_grounding_flags_unsupported_inference():
    sources = {
        "fundamentals": (
            "## Evidence Claims\n"
            "- [FACT] 营业收入同比增长20%。\n"
            "- [FACT] 经营现金流为正。"
        )
    }
    final = (
        "## Evidence Claims\n"
        "- [FACT] 营业收入同比增长20%。\n"
        "- [INFERENCE] 公司已经形成不可逆的全球垄断壁垒。"
    )
    result = evaluate_claim_grounding(final, sources)
    assert result.total_claims == 2
    assert result.supported_claims >= 1
    assert result.unsupported_claims >= 1


def test_service_exposes_v16_status_and_temporal_provenance():
    source = (
        Path(__file__).resolve().parents[1] / "service" / "app.py"
    ).read_text(encoding="utf-8")
    assert 'version="1.7"' in source
    assert 'publish_date_verified=False' in source

    conversation = (
        Path(__file__).resolve().parents[1]
        / "tradingagents"
        / "conversation"
        / "agent.py"
    ).read_text(encoding="utf-8")
    for status in (
        "COMPLETE",
        "PARTIAL",
        "REVIEW_REQUIRED",
        "DATA_UNAVAILABLE",
        "SYSTEM_ERROR",
    ):
        assert status in conversation
    assert "task_contract" in conversation
    assert "completion_ratio" in conversation


def test_adaptive_style_weights_are_pit_safe_and_shrunk_to_rule_prior(tmp_path):
    history = tmp_path / "style_ic.csv"
    rows = []
    for idx in range(20):
        signal_date = pd.Timestamp("2026-06-01") + pd.Timedelta(days=idx)
        available_date = signal_date + pd.Timedelta(days=20)
        rows.append(
            {
                "date": signal_date.date().isoformat(),
                "available_date": available_date.date().isoformat(),
                "momentum_ic": 0.01,
                "valuation_ic": 0.02,
                "dividend_ic": 0.20,
                "liquidity_ic": 0.03,
            }
        )
    # Signal is old enough, but its forward label matures after cutoff.
    rows.append(
        {
            "date": "2026-08-01",
            "available_date": "2026-10-01",
            "momentum_ic": 9.0,
            "valuation_ic": 0.0,
            "dividend_ic": 0.0,
            "liquidity_ic": 0.0,
        }
    )
    pd.DataFrame(rows).to_csv(history, index=False)

    base = {
        "momentum": 0.40,
        "valuation": 0.20,
        "dividend": 0.15,
        "liquidity": 0.25,
    }
    adapted = load_adaptive_style_weights(
        "2026-09-05",
        base_weights=base,
        history_path=history,
        strength=0.5,
        min_observations=12,
    )
    assert adapted.used is True
    assert adapted.observations == 20
    assert adapted.weights["dividend"] > base["dividend"]
    assert adapted.weights["momentum"] < 0.7
    assert abs(sum(adapted.weights.values()) - 1.0) < 1e-9


def test_adaptive_style_weights_fail_closed_without_label_availability(tmp_path):
    history = tmp_path / "legacy_style_ic.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-07-01",
                "momentum_ic": 0.1,
                "valuation_ic": 0.1,
                "dividend_ic": 0.1,
                "liquidity_ic": 0.1,
            }
        ]
    ).to_csv(history, index=False)
    base = {
        "momentum": 0.40,
        "valuation": 0.20,
        "dividend": 0.15,
        "liquidity": 0.25,
    }
    adapted = load_adaptive_style_weights(
        "2026-09-05",
        base_weights=base,
        history_path=history,
    )
    assert adapted.used is False
    assert adapted.weights == base
    assert "available_date" in adapted.warning


def test_web_ui_exposes_human_in_the_loop_controls():
    source = (
        Path(__file__).resolve().parents[1]
        / "service"
        / "static"
        / "index.html"
    ).read_text(encoding="utf-8")
    assert "继续补查缺失项" in source
    assert "回滚上一版" in source
    assert "REVIEW_REQUIRED" in source
    assert "DATA_UNAVAILABLE" in source
    assert "renderActions" in source


def test_completion_gate_constrains_missing_items_to_task_contract():
    contract = TaskContract(
        objective="梳理公告、新闻和政策风险",
        required_dimensions=["公告", "新闻", "政策"],
        required_entities=["601016.SH"],
        critical_requirements=["公告", "政策"],
    )
    raw = CompletionAssessment(
        complete=False,
        completion_ratio=0.6,
        completed_items=[
            "601016.SH × 公告：已有媒体转载",
            "601016.SH × 新闻：已有报道",
        ],
        missing_items=[
            "601016.SH × 政策：缺少专项政策原文",
            "601016.SH × 财务：缺少应收账款附注",
        ],
        evidence_gaps=["公告原文未核验"],
    )
    result = CompletionGate._sanitize(contract, raw)
    assert result.completed_items == [
        "601016.SH::公告",
        "601016.SH::新闻",
    ]
    assert result.missing_items == ["601016.SH::政策"]
    assert result.completion_ratio == 2 / 3
    assert any("财务" in item for item in result.evidence_gaps)


def test_respond_step_does_not_reassess_evidence_completion():
    source = (
        Path(__file__).resolve().parents[1]
        / "tradingagents"
        / "conversation"
        / "agent.py"
    ).read_text(encoding="utf-8")
    assert 'if action.action == "respond" and step_index > 0:' in source
    assert "must not change evidence-completion bookkeeping" in source


class StructuredContractLLM:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _prompt):
        return TaskContract(
            objective="",
            required_dimensions=["business_operations", "policy_risk"],
            required_entities=[],
            critical_requirements=["business_operations"],
            expected_output="research_answer",
            can_be_partial=True,
        )


def test_task_contract_builder_returns_structured_result_instead_of_fallback():
    builder = TaskContractBuilder(StructuredContractLLM())
    contract = builder.build(
        "完整分析节能风电业务经营和政策风险",
        ticker="601016.SH",
        as_of_date="2026-09-06",
    )
    assert contract.objective
    assert contract.required_dimensions == [
        "business_operations",
        "policy_risk",
    ]
    assert contract.required_entities == ["601016.SH"]


def test_continuation_request_preserves_previous_contract_semantics():
    assert ConversationAgent._is_continuation_request(
        "请继续补查上一轮尚未覆盖或缺少证据的项目"
    )
    objective = ConversationAgent._continuation_objective(
        "继续补查缺失项",
        {
            "missing_items": [
                "601016.SH::业务与经营分析",
                "601016.SH::风险维度-政策风险",
            ],
            "evidence_gaps": ["弃风限电数据缺失"],
        },
    )
    assert "业务与经营分析" in objective
    assert "政策风险" in objective
    assert "弃风限电" in objective
    assert "不要重跑完整研究" in objective


def test_document_evidence_skill_is_hidden_when_rag_disabled():
    agent = ConversationAgent.__new__(ConversationAgent)
    agent.config = {"rag_enabled": False}
    agent.tools = []
    registry = agent._build_capability_registry()
    assert registry.get("document_evidence_analysis") is None
    assert registry.get("deep_stock_research") is not None


def test_document_evidence_skill_is_available_when_rag_enabled():
    agent = ConversationAgent.__new__(ConversationAgent)
    agent.config = {"rag_enabled": True}
    agent.tools = []
    registry = agent._build_capability_registry()
    assert registry.get("document_evidence_analysis") is not None


def test_repair_fallback_uses_complementary_specialists_without_full_rerun():
    registry = CapabilityRegistry()
    for name in ("market", "fundamentals", "news"):
        registry.register(
            CapabilitySpec(
                name=name,
                kind="agent",
                description=name,
                requires_ticker=True,
            )
        )
    registry.register(
        CapabilitySpec(
            name="deep_stock_research",
            kind="skill",
            description="deep research",
            requires_ticker=True,
        )
    )
    supervisor = ConversationSupervisor(NoStructuredLLM(), registry)
    repair_query = (
        "继续补查：主营业务、装机容量、发电量、政策风险、弃风限电、补贴"
    )
    first = supervisor.decide(
        repair_query,
        current_ticker="601016.SH",
        as_of_date="2026-09-06",
        history=[],
        repair_mode=True,
        used_capabilities=[],
    )
    assert first.action == "delegate_agent"
    assert first.target == "fundamentals"

    second = supervisor.decide(
        repair_query,
        current_ticker="601016.SH",
        as_of_date="2026-09-06",
        history=[],
        repair_mode=True,
        used_capabilities=["delegate_agent:fundamentals"],
    )
    assert second.action == "delegate_agent"
    assert second.target == "news"


class InconsistentCompletionLLM:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _prompt):
        return CompletionAssessment(
            complete=True,
            completion_ratio=0.9,
            completed_items=["601016.SH::technical"],
            missing_items=["601016.SH::business_operations"],
            critical_missing=[],
            evidence_gaps=[],
            reason="model marked complete despite a missing checklist item",
        )


def test_structured_completion_is_sanitized_before_status_mapping():
    contract = TaskContract(
        objective="完整分析节能风电",
        required_dimensions=["technical", "business_operations"],
        required_entities=["601016.SH"],
        critical_requirements=["technical"],
        expected_output="research_answer",
        can_be_partial=True,
    )
    gate = CompletionGate(InconsistentCompletionLLM())
    result = gate.assess(
        contract,
        observations=["technical 已覆盖"],
        used_capabilities=["run_skill:deep_stock_research"],
    )
    assert result.complete is False
    assert result.completion_ratio == 0.5
    assert result.missing_items == ["601016.SH::business_operations"]
