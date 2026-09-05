import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

_ENV_OVERRIDES = {
    "TRADINGAGENTS_LLM_PROVIDER": "llm_provider",
    "TRADINGAGENTS_DEEP_THINK_LLM": "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM": "quick_think_llm",
    "TRADINGAGENTS_LLM_BACKEND_URL": "backend_url",
    "TRADINGAGENTS_OUTPUT_LANGUAGE": "output_language",
    "TRADINGAGENTS_MAX_AUDIT_ROUNDS": "max_audit_rounds",
    "TRADINGAGENTS_CHECKPOINT_ENABLED": "checkpoint_enabled",
    "TRADINGAGENTS_BENCHMARK_TICKER": "benchmark_ticker",
    "TRADINGAGENTS_MEMORY_REFLECTION_ENABLED": "memory_reflection_enabled",
    "TRADINGAGENTS_TEMPERATURE": "temperature",
    "TRADINGAGENTS_LLM_MAX_RETRIES": "llm_max_retries",
    "TRADINGAGENTS_GOOGLE_THINKING_LEVEL": "google_thinking_level",
    "TRADINGAGENTS_OPENAI_REASONING_EFFORT": "openai_reasoning_effort",
    "TRADINGAGENTS_ANTHROPIC_EFFORT": "anthropic_effort",
    "TRADINGAGENTS_STRICT_ASOF": "strict_asof",
    "TRADINGAGENTS_ANALYST_CONCURRENCY": "analyst_max_concurrency",
    "TRADINGAGENTS_SECTOR_DISCOVERY_TOP_N": "sector_discovery_top_n",
    "TRADINGAGENTS_SECTOR_ML_MODEL_PATH": "sector_ml_model_path",
    "TRADINGAGENTS_SECTOR_ML_WEIGHT": "sector_ml_weight",
    "TRADINGAGENTS_REPRESENTATIVES_PER_SECTOR": "representatives_per_sector",
    "TRADINGAGENTS_REPRESENTATIVE_COMPONENT_LIMIT": "representative_component_limit",
    "TRADINGAGENTS_MCP_ENABLED": "mcp_enabled",
    "TRADINGAGENTS_MCP_URL": "mcp_url",
    "TRADINGAGENTS_MCP_FALLBACK_TO_LOCAL": "mcp_fallback_to_local",
    "TRADINGAGENTS_EXTERNAL_MCP_ENABLED": "external_mcp_enabled",
    "TRADINGAGENTS_EXTERNAL_MCP_SERVERS_JSON": "external_mcp_servers_json",
    "TRADINGAGENTS_EXTERNAL_MCP_TOOL_ALLOWLIST": "external_mcp_tool_allowlist",
    "TRADINGAGENTS_RAG_ENABLED": "rag_enabled",
    "TRADINGAGENTS_QDRANT_URL": "qdrant_url",
    "TRADINGAGENTS_QDRANT_COLLECTION": "qdrant_collection",
    "TRADINGAGENTS_QDRANT_API_KEY": "qdrant_api_key",
    "TRADINGAGENTS_RAG_EMBEDDING_BACKEND": "rag_embedding_backend",
    "TRADINGAGENTS_RAG_EMBEDDING_MODEL": "rag_embedding_model",
    "TRADINGAGENTS_RAG_RERANKER_ENABLED": "rag_reranker_enabled",
    "TRADINGAGENTS_RAG_RERANKER_MODEL": "rag_reranker_model",
    "TRADINGAGENTS_RAG_CANDIDATE_K": "rag_candidate_k",
    "TRADINGAGENTS_RAG_BM25_CORPUS_LIMIT": "rag_bm25_corpus_limit",
    "TRADINGAGENTS_CONVERSATION_DB_PATH": "conversation_db_path",
    "TRADINGAGENTS_CONVERSATION_HISTORY_TURNS": "conversation_history_turns",
    "TRADINGAGENTS_CONVERSATION_TOOL_ROUNDS": "conversation_tool_rounds",
    "TRADINGAGENTS_CONVERSATION_SUPERVISOR_STEPS": "conversation_supervisor_steps",
    "TRADINGAGENTS_KNOWLEDGE_UPLOAD_MAX_MB": "knowledge_upload_max_mb",
}

_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")


def _coerce(value: str, reference):
    if isinstance(reference, bool):
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE:
            return True
        if normalized in _BOOL_FALSE:
            return False
        raise ValueError(f"需要布尔值，实际得到 {value!r}")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        try:
            config[key] = _coerce(raw, config.get(key))
        except ValueError as exc:
            raise ValueError(f"环境变量 {env_var} 的值无效：{exc}") from exc
    return config


DEFAULT_CONFIG = _apply_env_overrides({
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    "memory_log_max_entries": None,
    "memory_reflection_enabled": False,
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.5",
    "quick_think_llm": "gpt-5.4-mini",
    "backend_url": None,
    "google_thinking_level": None,
    "openai_reasoning_effort": None,
    "anthropic_effort": None,
    "temperature": None,
    "llm_max_retries": None,
    "checkpoint_enabled": False,
    "output_language": "Chinese",
    "max_audit_rounds": 2,
    "max_recur_limit": 60,
    "strict_asof": True,
    "analyst_max_concurrency": 3,
    "sector_discovery_top_n": 6,
    "sector_ml_model_path": "",
    "sector_ml_weight": 0.5,
    "representatives_per_sector": 2,
    "representative_component_limit": 20,
    "mcp_enabled": False,
    "mcp_url": "http://localhost:8001/mcp",
    "mcp_fallback_to_local": True,
    "external_mcp_enabled": False,
    "external_mcp_servers_json": "",
    "external_mcp_tool_allowlist": "",
    "rag_enabled": False,
    "qdrant_url": "http://localhost:6333",
    "qdrant_collection": "a_share_knowledge",
    "qdrant_api_key": "",
    "rag_embedding_backend": "fastembed",
    "rag_embedding_model": "BAAI/bge-small-zh-v1.5",
    "rag_reranker_enabled": True,
    "rag_reranker_model": "BAAI/bge-reranker-base",
    "rag_candidate_k": 30,
    "rag_bm25_corpus_limit": 1000,
    "rag_excerpt_chars": 650,
    "rag_hash_dimension": 256,
    "conversation_db_path": os.getenv(
        "TRADINGAGENTS_CONVERSATION_DB_PATH",
        os.path.join(_TRADINGAGENTS_HOME, "conversation", "conversation.db"),
    ),
    "conversation_history_turns": 12,
    "conversation_tool_rounds": 4,
    "conversation_supervisor_steps": 3,
    "knowledge_upload_max_mb": 25,
    "news_article_limit": 20,
    "global_news_article_limit": 10,
    "global_news_lookback_days": 7,
    "global_news_queries": [
        "中国人民银行 货币政策 LPR 降准 降息",
        "中国 CPI PPI PMI GDP 宏观经济",
        "A股 监管政策 证监会 上交所 深交所",
        "人民币 汇率 外贸 地缘风险",
        "原油 大宗商品 供应链 中国经济",
    ],
    "data_vendors": {
        "core_stock_apis": "baostock,akshare",
        "technical_indicators": "baostock",
        "fundamental_data": "akshare",
        "news_data": "cninfo,akshare",
        "macro_data": "akshare",
        "a_share_data": "baostock,akshare",
    },
    "tool_vendors": {
        "get_fundamentals": "baostock,akshare",
        "get_balance_sheet": "akshare",
        "get_income_statement": "akshare",
        "get_cashflow": "akshare",
        "get_news": "cninfo,akshare",
        "get_macro_indicators": "akshare",
        "get_index_data": "baostock",
    },
    "benchmark_ticker": "000300.SH",
    "benchmark_map": {
        ".SH": "000300.SH",
        ".SZ": "000300.SH",
        ".BJ": "000300.SH",
        "": "000300.SH",
    },
})
