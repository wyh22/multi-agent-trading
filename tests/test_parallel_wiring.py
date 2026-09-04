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
