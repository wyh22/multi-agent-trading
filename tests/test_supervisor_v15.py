from langchain_core.messages import HumanMessage
from pathlib import Path

import pytest
import tradingagents.dataflows.interface as data_interface

from tradingagents.agents.schemas import AuditIssue, AuditResult
from tradingagents.agents.analysts.news_analyst import _needs_insider_transactions
from tradingagents.agents.utils.tool_registry import build_local_tool_groups
from tradingagents.capabilities.registry import CapabilityRegistry, CapabilitySpec
from tradingagents.conversation.store import ConversationStore
from tradingagents.orchestration.schemas import SupervisorAction
from tradingagents.orchestration.supervisor import ConversationSupervisor
from tradingagents.rag.loaders import load_documents
from tradingagents.skills.registry import BUILTIN_SKILLS


class NoStructuredLLM:
    def with_structured_output(self, _schema):
        raise NotImplementedError


def test_supervisor_falls_back_to_existing_router_when_structured_output_unavailable():
    registry = CapabilityRegistry()
    registry.register(
        CapabilitySpec(
            name="deep_stock_research",
            kind="skill",
            description="deep research",
            requires_ticker=True,
        )
    )
    supervisor = ConversationSupervisor(NoStructuredLLM(), registry)
    action = supervisor.decide(
        "请深度分析600519.SH",
        current_ticker="600519.SH",
        as_of_date="2026-09-05",
        history=[],
    )
    assert action.action == "run_deep_research"
    assert action.target == "deep_stock_research"


def test_shared_rag_tool_is_available_to_news_fundamentals_and_supervisor():
    groups = build_local_tool_groups({"rag_enabled": True})
    for group in ("news", "fundamentals", "knowledge"):
        names = [tool.name for tool in groups[group]]
        assert "search_company_knowledge" in names


def test_skill_registry_loads_declarative_manifests():
    assert {
        "deep_stock_research",
        "sector_discovery",
        "document_evidence_analysis",
        "company_comparison",
    }.issubset(BUILTIN_SKILLS)
    assert BUILTIN_SKILLS["deep_stock_research"].requires_audit is True
    assert "fundamentals" in BUILTIN_SKILLS["document_evidence_analysis"].allowed_agents


def test_research_versions_are_immutable_and_can_rollback(tmp_path):
    store = ConversationStore(tmp_path / "conversation.db")
    tid = store.ensure_thread(
        current_ticker="600519.SH",
        as_of_date="2026-09-05",
    )
    v1 = store.save_research_version(
        tid,
        {
            "ticker": "600519.SH",
            "as_of_date": "2026-09-05",
            "research_context": "context-v1",
            "final_trade_decision": "decision-v1",
        },
        audit_status="PASS",
    )
    v2 = store.save_research_version(
        tid,
        {
            "ticker": "600519.SH",
            "as_of_date": "2026-09-05",
            "research_context": "context-v2",
            "final_trade_decision": "decision-v2",
        },
        audit_status="PASS",
    )
    assert v2 > v1
    assert store.get_active_research_version(tid)["id"] == v2

    restored = store.rollback_research_version(tid)
    assert restored is not None
    assert restored["id"] == v1
    assert store.get_active_research_version(tid)["id"] == v1
    assert store.get_thread(tid)["research_context"] == "context-v1"
    versions = store.list_research_versions(tid)
    assert len(versions) == 2
    assert store.reset(tid) is True
    assert store.list_research_versions(tid) == []


def test_audit_issue_can_target_specialist_repair():
    result = AuditResult(
        verdict="REVISE",
        grounding_score=0.5,
        pit_score=1.0,
        consistency_score=0.8,
        unsupported_claims=["现金流改善缺少证据"],
        issues=[
            AuditIssue(
                issue_type="missing_evidence",
                repair_target="fundamentals",
                affected_claims=["现金流改善"],
                instruction="重新核验现金流量表与对应财报原文。",
            )
        ],
        revision_instructions=["补齐现金流证据"],
        summary="需要重新取证。",
    )
    assert result.issues[0].repair_target == "fundamentals"


def test_document_loaders_support_txt_pdf_and_docx(tmp_path):
    txt = tmp_path / "note.txt"
    txt.write_text("公司公告证据。", encoding="utf-8")
    txt_docs = load_documents(
        txt,
        ticker="600519.SH",
        publish_date="2026-09-01",
        source_name="uploaded-note.txt",
    )
    assert txt_docs[0].metadata["file_hash"]
    assert txt_docs[0].url == "upload://uploaded-note.txt"
    assert "公司公告" in txt_docs[0].text

    import fitz

    pdf_path = tmp_path / "annual.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Annual report risk disclosure")
    pdf.save(pdf_path)
    pdf.close()
    pdf_docs = load_documents(
        pdf_path,
        ticker="600519.SH",
        publish_date="2026-09-01",
        doc_type="annual_report",
    )
    assert pdf_docs
    assert pdf_docs[0].metadata["page"] == 1

    from docx import Document

    docx_path = tmp_path / "report.docx"
    document = Document()
    document.add_heading("风险因素", level=1)
    document.add_paragraph("渠道库存变化需要持续观察。")
    document.save(docx_path)
    docx_docs = load_documents(
        docx_path,
        ticker="600519.SH",
        publish_date="2026-09-01",
        doc_type="annual_report",
    )
    assert "渠道库存" in docx_docs[0].text


def test_service_exposes_knowledge_and_version_routes():
    source = (
        Path(__file__).resolve().parents[1] / "service" / "app.py"
    ).read_text(encoding="utf-8")
    assert '@app.post("/knowledge/upload")' in source
    assert '@app.get("/chat/{thread_id}/versions")' in source
    assert '@app.post("/chat/{thread_id}/rollback")' in source


def test_unapproved_research_does_not_enter_long_term_memory():
    source = (
        Path(__file__).resolve().parents[1]
        / "tradingagents"
        / "graph"
        / "trading_graph.py"
    ).read_text(encoding="utf-8")
    assert 'if audit_status == "PASS":' in source
    assert 'else "REVIEW"' in source
    assert "self.memory_log.store_decision" in source


def test_conversation_supervisor_loop_is_bounded_and_observation_driven():
    source = (
        Path(__file__).resolve().parents[1]
        / "tradingagents"
        / "conversation"
        / "agent.py"
    ).read_text(encoding="utf-8")
    assert 'conversation_supervisor_steps' in source
    assert 'observations=observations' in source
    assert 'used_capabilities=used_capabilities' in source
    assert '"supervisor:repeat_guard"' in source
    assert '"supervisor:step_limit"' in source
    assert '"supervisor_trace": supervisor_trace' in source


def _specialist_registry():
    registry = CapabilityRegistry()
    for name in ("market", "news", "fundamentals"):
        registry.register(
            CapabilitySpec(
                name=name,
                kind="agent",
                description=f"{name} specialist",
                requires_ticker=True,
            )
        )
    registry.register(
        CapabilitySpec(
            name="get_verified_market_snapshot",
            kind="tool",
            description="verified market snapshot",
            requires_ticker=True,
        )
    )
    registry.register(
        CapabilitySpec(
            name="sector_discovery",
            kind="skill",
            description="sector discovery",
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
    return registry


def test_supervisor_normalizes_catalog_prefixed_targets():
    tool_action = ConversationSupervisor._normalize_target(
        SupervisorAction(
            action="call_tool",
            target="tool:get_verified_market_snapshot",
        )
    )
    assert tool_action.target == "get_verified_market_snapshot"

    agent_action = ConversationSupervisor._normalize_target(
        SupervisorAction(
            action="delegate_agent",
            target="agent:market",
        )
    )
    assert agent_action.target == "market"

    skill_action = ConversationSupervisor._normalize_target(
        SupervisorAction(
            action="run_skill",
            target="skill:sector_discovery",
        )
    )
    assert skill_action.target == "sector_discovery"


def test_supervisor_fallback_keeps_single_domain_market_query_lightweight():
    supervisor = ConversationSupervisor(
        NoStructuredLLM(),
        _specialist_registry(),
    )
    action = supervisor.decide(
        "分析当前的技术趋势、动量和主要技术风险。",
        current_ticker="601016.SH",
        as_of_date="2026-09-06",
        history=[],
    )
    assert action.action == "delegate_agent"
    assert action.target == "market"


def test_supervisor_fallback_keeps_single_domain_fundamentals_query_lightweight():
    supervisor = ConversationSupervisor(
        NoStructuredLLM(),
        _specialist_registry(),
    )
    action = supervisor.decide(
        "分析当前现金流质量和盈利能力。",
        current_ticker="601016.SH",
        as_of_date="2026-09-06",
        history=[],
    )
    assert action.action == "delegate_agent"
    assert action.target == "fundamentals"


def test_supervisor_fallback_escalates_cross_domain_query_to_deep_research():
    supervisor = ConversationSupervisor(
        NoStructuredLLM(),
        _specialist_registry(),
    )
    action = supervisor.decide(
        "结合技术趋势、现金流和近期政策风险做完整分析。",
        current_ticker="601016.SH",
        as_of_date="2026-09-06",
        history=[],
    )
    assert action.action == "run_deep_research"
    assert action.target == "deep_stock_research"


def test_news_agent_hides_expensive_insider_tool_for_general_news_request():
    messages = [
        HumanMessage(
            content="梳理节能风电近期最重要的公告、新闻和政策风险。"
        )
    ]
    assert _needs_insider_transactions(messages) is False


def test_news_agent_allows_insider_tool_for_explicit_management_holding_request():
    messages = [
        HumanMessage(
            content="梳理节能风电近期董监高和高管增减持情况。"
        )
    ]
    assert _needs_insider_transactions(messages) is True


def test_company_news_vendor_failure_degrades_instead_of_raising(monkeypatch):
    def boom(*_args, **_kwargs):
        raise ValueError("upstream returned non-JSON")

    original = data_interface.VENDOR_METHODS["get_news"]
    monkeypatch.setattr(
        data_interface,
        "get_vendor",
        lambda _category, _method=None: "cninfo",
    )
    monkeypatch.setitem(
        data_interface.VENDOR_METHODS,
        "get_news",
        {"cninfo": boom},
    )

    result = data_interface.route_to_vendor(
        "get_news",
        "601016.SH",
        "2026-06-01",
        "2026-09-06",
    )
    assert result.startswith("DATA_UNAVAILABLE:")
    assert "upstream returned non-JSON" in result

    data_interface.VENDOR_METHODS["get_news"] = original


def test_project_requires_cninfo_fixed_akshare_floor():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "akshare>=1.18.67" in pyproject
    assert "akshare>=1.18.67" in requirements


def test_vendor_circuit_breaker_skips_repeated_failed_vendor(monkeypatch):
    data_interface._reset_vendor_circuit_breakers()
    calls = {"cninfo": 0, "akshare": 0}

    def broken_cninfo(*_args, **_kwargs):
        calls["cninfo"] += 1
        raise ValueError("non-json upstream response")

    def healthy_akshare(*_args, **_kwargs):
        calls["akshare"] += 1
        return "fallback-news"

    monkeypatch.setattr(
        data_interface,
        "get_vendor",
        lambda _category, _method=None: "cninfo,akshare",
    )
    monkeypatch.setitem(
        data_interface.VENDOR_METHODS,
        "get_news",
        {
            "cninfo": broken_cninfo,
            "akshare": healthy_akshare,
        },
    )

    assert data_interface.route_to_vendor(
        "get_news",
        "601016.SH",
        "2026-06-01",
        "2026-09-06",
    ) == "fallback-news"
    assert data_interface.route_to_vendor(
        "get_news",
        "601016.SH",
        "2026-06-01",
        "2026-09-06",
    ) == "fallback-news"
    assert calls == {"cninfo": 1, "akshare": 2}
    data_interface._reset_vendor_circuit_breakers()


def test_repair_rejects_single_ticker_company_comparison():
    with pytest.raises(ValueError, match="at least two distinct tickers"):
        ConversationSupervisor._validate_action_feasibility(
            SupervisorAction(
                action="run_skill",
                target="company_comparison",
                arguments={"tickers": ["601016.SH"]},
            ),
            repair_mode=True,
        )


def test_repair_fallback_stops_after_relevant_specialists_are_used():
    supervisor = ConversationSupervisor(
        NoStructuredLLM(),
        _specialist_registry(),
    )
    action = supervisor.decide(
        "继续补查主营业务、估值、政策风险和补贴缺口",
        current_ticker="601016.SH",
        as_of_date="2026-09-06",
        history=[],
        repair_mode=True,
        used_capabilities=[
            "delegate_agent:fundamentals",
            "delegate_agent:news",
        ],
    )
    assert action.action == "respond"
    assert action.target is None
