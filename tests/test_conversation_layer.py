from pathlib import Path
from tradingagents.conversation.router import extract_ticker, route_message
from tradingagents.conversation.store import ConversationStore

def test_router_extracts_a_share_code_and_research_intent():
    route=route_message("请深度分析601330.SSE")
    assert route.intent=="research"
    assert route.ticker=="601330.SH"
    assert extract_ticker("看看000001")=="000001.SZ"

def test_router_keeps_thread_ticker_for_followup():
    route=route_message("刚才结论最大的风险是什么？",current_ticker="600519.SH")
    assert route.intent=="tool_chat"
    assert route.ticker=="600519.SH"

def test_conversation_store_persists_thread_history_and_context(tmp_path):
    store=ConversationStore(tmp_path/"conversation.db")
    tid=store.ensure_thread(current_ticker="600519.SH",as_of_date="2026-08-20")
    store.append_message(tid,"user","深度分析"); store.append_message(tid,"assistant","研究结论")
    store.update_context(tid,research_context="审计后的研究上下文",last_intent="research")
    thread=store.get_thread(tid)
    assert thread["current_ticker"]=="600519.SH"
    assert thread["research_context"]=="审计后的研究上下文"
    assert [item["role"] for item in store.history(tid)]==["user","assistant"]
    assert store.reset(tid) is True
    assert store.get_thread(tid) is None

def test_fastapi_exposes_multiturn_chat_ui_and_history_routes():
    source=(Path(__file__).resolve().parents[1]/"service"/"app.py").read_text(encoding="utf-8")
    assert '@app.post("/chat")' in source
    assert '@app.get("/chat/{thread_id}")' in source
    assert '@app.delete("/chat/{thread_id}")' in source
    assert 'app.mount("/ui"' in source

def test_external_mcp_tools_require_allowlist_in_conversation_layer():
    source=(Path(__file__).resolve().parents[1]/"tradingagents"/"conversation"/"agent.py").read_text(encoding="utf-8")
    assert "external_mcp_tool_allowlist" in source
    assert "tool.name in allowlist" in source


def test_router_recognizes_sector_and_representative_pool_requests():
    assert route_message("帮我看看值得关注的行业").intent=="discovery"
    assert route_message("给我每个行业的代表股").intent=="discovery"

def test_fastapi_exposes_research_pool_and_candidate_context_handoff():
    source=(Path(__file__).resolve().parents[1]/"service"/"app.py").read_text(encoding="utf-8")
    assert '@app.post("/research-pool")' in source
    assert "candidate_context" in source
