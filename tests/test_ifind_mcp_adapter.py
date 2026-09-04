from pathlib import Path
from tradingagents.mcp.ifind import _normalize_codes, _split_indicator_params

def test_ifind_code_and_indicator_payload_normalization():
    assert _normalize_codes("601330.SSE,000001.SZ")=="601330.SH,000001.SZ"
    payload=_split_indicator_params("ths_close_price_stock;ths_total_shares_stock","20260820,100,20260820;20260820")
    assert payload[0]["indicator"]=="ths_close_price_stock"
    assert payload[0]["indiparams"]==["20260820","100","20260820"]
    assert payload[1]["indiparams"]==["20260820"]

def test_finance_mcp_exposes_optional_ifind_tools_without_embedding_credentials():
    root=Path(__file__).resolve().parents[1]
    server=(root/"tradingagents"/"mcp"/"server.py").read_text(encoding="utf-8")
    env=(root/".env.example").read_text(encoding="utf-8")
    assert "def ifind_snapshot(" in server
    assert "def ifind_basic_data(" in server
    assert "def ifind_date_sequence(" in server
    assert "TRADINGAGENTS_IFIND_REFRESH_TOKEN=" in env

def test_ifind_tools_are_exposed_only_to_conversation_layer_when_enabled():
    root=Path(__file__).resolve().parents[1]
    source=(root/"tradingagents"/"conversation"/"agent.py").read_text(encoding="utf-8")
    assert 'self.config.get("ifind_enabled", False)' in source
    assert 'tool.name.startswith("ifind_")' in source
    core=(root/"tradingagents"/"agents"/"utils"/"tool_registry.py").read_text(encoding="utf-8")
    assert "ifind_snapshot" not in core
