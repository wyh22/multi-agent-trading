from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def _read(path): return (ROOT/path).read_text(encoding="utf-8")

def test_mcp_server_exposes_finance_and_rag_tools():
    source=_read("tradingagents/mcp/server.py")
    assert "FastMCP" in source
    assert "def get_stock_data(" in source
    assert "def get_fundamentals(" in source
    assert "def search_company_knowledge(" in source
    assert 'transport="streamable-http"' in source

def test_langgraph_can_load_mcp_tools_with_local_fallback_and_shared_rag():
    source=_read("tradingagents/agents/utils/tool_registry.py")
    assert "load_mcp_tools_sync" in source
    assert "mcp_fallback_to_local" in source
    assert 'groups["knowledge"] = [search_company_knowledge]' in source
    assert '_append_unique(groups["news"], search_company_knowledge)' in source
    assert '_append_unique(groups["fundamentals"], search_company_knowledge)' in source

def test_docker_compose_contains_agent_mcp_qdrant_services():
    compose=_read("docker-compose.yml")
    assert "agent-api:" in compose
    assert "finance-mcp:" in compose
    assert "qdrant:" in compose
    assert "TRADINGAGENTS_MCP_ENABLED" in compose
    assert "TRADINGAGENTS_RAG_ENABLED" in compose


def test_news_analyst_prompt_knows_optional_injected_rag_tools():
    source=_read("tradingagents/agents/analysts/news_analyst.py")
    assert "active_tool_names" in source
    assert '"search_company_knowledge" in active_tool_names' in source
    assert '"get_insider_transactions" in active_tool_names' in source
    assert "llm.bind_tools(active_tools)" in source


def test_fundamentals_analyst_prompt_knows_shared_rag_tools():
    source=_read("tradingagents/agents/analysts/fundamentals_analyst.py")
    assert "active_tool_names" in source
    assert '"search_company_knowledge" in active_tool_names' in source
    assert "annual reports" in source
