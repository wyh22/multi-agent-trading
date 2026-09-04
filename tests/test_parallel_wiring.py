"""最终版并行拓扑的静态回归测试。"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")

def test_three_analysts_use_private_subgraphs_and_fanin_barrier():
    source = _read("tradingagents/graph/setup.py")
    assert "build_analyst_runner(" in source
    assert "workflow.add_edge(START, spec.agent_node)" in source
    assert '"_AnalystFanIn"' in source
    assert "[spec.agent_node for spec in plan.specs]" in source

def test_private_subgraph_only_returns_report_and_trace():
    source = _read("tradingagents/graph/analyst_subgraph.py")
    assert 'return {report_key: report_value, "analyst_trace": tool_trace}' in source
    assert '"messages": [HumanMessage' in source

def test_bull_and_bear_run_in_parallel_then_fanin():
    source = _read("tradingagents/graph/setup.py")
    assert 'workflow.add_edge("_AnalystFanIn", "Bull Researcher")' in source
    assert 'workflow.add_edge("_AnalystFanIn", "Bear Researcher")' in source
    assert '"_ResearchFanIn"' in source
    assert '["Bull Researcher", "Bear Researcher"]' in source

def test_redundant_original_agents_are_removed():
    setup_source = _read("tradingagents/graph/setup.py")
    for name in ["Research Manager","Aggressive Analyst","Conservative Analyst","Neutral Analyst"]:
        assert name not in setup_source
    assert not (ROOT / "tradingagents/agents/analysts/sentiment_analyst.py").exists()
    assert not (ROOT / "tradingagents/agents/trader").exists()
    assert not (ROOT / "tradingagents/agents/risk_mgmt").exists()
    assert not (ROOT / "tradingagents/agents/managers/research_manager.py").exists()

def test_decision_auditor_can_route_back_once_for_revision():
    setup_source = _read("tradingagents/graph/setup.py")
    logic_source = _read("tradingagents/graph/conditional_logic.py")
    assert 'workflow.add_edge("Portfolio Manager", "Decision Auditor")' in setup_source
    assert "route_after_audit" in setup_source
    assert '"修订": "Portfolio Manager"' in setup_source
    assert 'status == "REVISE"' in logic_source

def test_bull_and_bear_use_current_state_contract():
    from types import SimpleNamespace

    from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
    from tradingagents.agents.researchers.bull_researcher import create_bull_researcher

    class FakeLLM:
        def invoke(self, _prompt):
            return SimpleNamespace(content="evidence-based thesis")

    state = {
        "company_of_interest": "600519.SH",
        "asset_type": "stock",
        "instrument_context": "A-share test instrument",
        "market_report": "market evidence",
        "news_report": "news evidence",
        "fundamentals_report": "fundamental evidence",
    }

    bull = create_bull_researcher(FakeLLM())(state)
    bear = create_bear_researcher(FakeLLM())(state)

    assert bull == {"bull_thesis": "evidence-based thesis"}
    assert bear == {"bear_thesis": "evidence-based thesis"}
    assert "investment_debate_state" not in bull
    assert "investment_debate_state" not in bear


def test_checkpoint_signature_has_no_removed_debate_or_risk_keys():
    source = _read("tradingagents/graph/trading_graph.py")
    assert "max_debate_rounds" not in source
    assert "max_risk_discuss_rounds" not in source
    assert "max_audit_rounds" in source

def test_cli_tracks_only_current_seven_agent_state():
    source = _read("cli/main.py")
    for removed in [
        "investment_debate_state",
        "risk_debate_state",
        "trader_investment_plan",
        "Research Manager",
        "Aggressive Analyst",
        "Neutral Analyst",
        "Conservative Analyst",
        "max_debate_rounds",
        "max_risk_discuss_rounds",
    ]:
        assert removed not in source
    assert '"bull_thesis"' in source
    assert '"bear_thesis"' in source
    assert '"Decision Auditor"' in source
