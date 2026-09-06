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

def test_decision_auditor_routes_repairs_to_responsible_capability():
    setup_source = _read("tradingagents/graph/setup.py")
    logic_source = _read("tradingagents/graph/conditional_logic.py")
    assert 'workflow.add_edge("Portfolio Manager", "Decision Auditor")' in setup_source
    assert "route_after_audit" in setup_source
    assert '"修订_market": repair_nodes["market"]' in setup_source
    assert '"修订_news": repair_nodes["news"]' in setup_source
    assert '"修订_fundamentals": repair_nodes["fundamentals"]' in setup_source
    assert '"修订_pm": "Portfolio Manager"' in setup_source
    assert 'state.get("audit_repair_target"' in logic_source

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

def test_report_writer_uses_current_state_fields():
    source = _read("tradingagents/reporting.py")
    for current in ["bull_thesis", "bear_thesis", "final_trade_decision", "audit_report"]:
        assert current in source
    for removed in ["investment_debate_state", "risk_debate_state", "trader_investment_plan"]:
        assert removed not in source


def test_analyst_factories_accept_injected_tool_groups():
    setup_source = _read("tradingagents/graph/setup.py")
    specs = {
        "market": "tradingagents/agents/analysts/market_analyst.py",
        "news": "tradingagents/agents/analysts/news_analyst.py",
        "fundamentals": "tradingagents/agents/analysts/fundamentals_analyst.py",
    }
    for key, path in specs.items():
        source = _read(path)
        factory = {
            "market": "create_market_analyst",
            "news": "create_news_analyst",
            "fundamentals": "create_fundamentals_analyst",
        }[key]
        assert f"def {factory}(llm, tools=None):" in source
        assert "active_tools = tools or [" in source
        assert "llm.bind_tools(active_tools)" in source
        assert f'self.tool_groups.get("{key}")' in setup_source


def test_graph_setup_builds_with_injected_local_tool_groups():
    from langgraph.prebuilt import ToolNode

    from tradingagents.agents.utils.tool_registry import build_local_tool_groups
    from tradingagents.graph.conditional_logic import ConditionalLogic
    from tradingagents.graph.setup import GraphSetup

    class FakeLLM:
        # Portfolio Manager / Auditor detect this as unsupported structured output
        # during graph construction and fall back without making any LLM call.
        def with_structured_output(self, _schema):
            raise NotImplementedError

    groups = build_local_tool_groups({"rag_enabled": False})
    tool_nodes = {key: ToolNode(value) for key, value in groups.items()}
    workflow = GraphSetup(
        FakeLLM(),
        FakeLLM(),
        tool_nodes,
        ConditionalLogic(max_audit_rounds=2),
        tool_groups=groups,
    ).setup_graph(("market", "news", "fundamentals"))

    # This is intentionally a real graph-construction smoke test.  It catches
    # factory signature drift that compile/import-only CI cannot detect.
    compiled = workflow.compile()
    assert compiled is not None
