from __future__ import annotations

import functools
from pathlib import Path
from datetime import date
import tempfile
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tradingagents.conversation import ConversationAgent, ConversationStore
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows.symbol_utils import normalize_a_share_symbol
from tradingagents.discovery.pipeline import run_discovery, run_research_pool
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.rag.ingestion import ingest_path
from tradingagents.rag.retriever import HybridKnowledgeRetriever
from tradingagents.rag.scope import normalize_scope
from tradingagents.rag.store import QdrantKnowledgeStore

app = FastAPI(title="TradingAgents A-share Agent API", version="1.7")
app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="chat-ui")


class AnalyzeRequest(BaseModel):
    ticker: str = Field(examples=["600519.SH"])
    trade_date: str = Field(default_factory=lambda: date.today().isoformat())
    analysts: list[Literal["market", "news", "fundamentals"]] = Field(
        default_factory=lambda: ["market", "news", "fundamentals"]
    )
    candidate_context: str | None = Field(
        default=None,
        max_length=3000,
        description=(
            "可选的行业发现/代表股研究来源。仅作为研究路由先验，"
            "不会被 Agent 当作投资事实。"
        ),
    )


class DiscoveryRequest(BaseModel):
    as_of_date: str = Field(default_factory=lambda: date.today().isoformat())
    top_n: int = Field(default=6, ge=1, le=31)


class ResearchPoolRequest(BaseModel):
    as_of_date: str = Field(default_factory=lambda: date.today().isoformat())
    sector_top_n: int = Field(default=4, ge=1, le=31)
    representatives_per_sector: int = Field(default=2, ge=1, le=5)
    component_limit: int = Field(default=20, ge=2, le=100)
    strict_pit: bool = True


class KnowledgeSearchRequest(BaseModel):
    ticker: str = Field(examples=["600519.SH"])
    query: str = Field(min_length=1)
    as_of_date: str = Field(default_factory=lambda: date.today().isoformat())
    top_k: int = Field(default=6, ge=1, le=10)
    doc_type: str | None = None
    industry: str | None = Field(
        default=None,
        description=(
            "可选行业名称/代码；为空时会尝试从已入库公司文档自动解析。"
        ),
    )


class RollbackRequest(BaseModel):
    version_id: int | None = Field(
        default=None,
        description="为空时回滚到当前版本的直接父版本。",
    )


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, examples=["深度分析600519.SH，然后告诉我最大的风险是什么"])
    thread_id: str | None = Field(default=None, description="为空时服务端创建新会话")
    ticker: str | None = Field(default=None, description="可选；显式指定当前会话标的")
    as_of_date: str | None = Field(default=None, description="为空时沿用会话截止日；新会话默认今天")
    mode: Literal["auto", "research", "discovery", "tool_chat"] = "auto"


@functools.lru_cache(maxsize=1)
def _conversation_store() -> ConversationStore:
    return ConversationStore(DEFAULT_CONFIG["conversation_db_path"])


@functools.lru_cache(maxsize=1)
def _conversation_agent() -> ConversationAgent:
    return ConversationAgent(DEFAULT_CONFIG, _conversation_store())


@functools.lru_cache(maxsize=1)
def _knowledge_store() -> QdrantKnowledgeStore:
    return QdrantKnowledgeStore(
        url=str(DEFAULT_CONFIG.get("qdrant_url", "http://localhost:6333")),
        collection=str(
            DEFAULT_CONFIG.get("qdrant_collection", "a_share_knowledge")
        ),
        embedder=None,
        api_key=DEFAULT_CONFIG.get("qdrant_api_key") or None,
    )


@functools.lru_cache(maxsize=1)
def _knowledge_retriever() -> HybridKnowledgeRetriever:
    return HybridKnowledgeRetriever.from_config(DEFAULT_CONFIG)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "1.7",
        "conversation_enabled": True,
        "mcp_enabled": bool(DEFAULT_CONFIG.get("mcp_enabled", False)),
        "rag_enabled": bool(DEFAULT_CONFIG.get("rag_enabled", False)),
        "qdrant_url": DEFAULT_CONFIG.get("qdrant_url"),
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        graph = TradingAgentsGraph(selected_analysts=tuple(req.analysts), config=DEFAULT_CONFIG)
        state, signal = graph.propagate(
            req.ticker,
            req.trade_date,
            candidate_context=req.candidate_context or "",
        )
        audit_status = str(state.get("audit_status", "") or "").upper()
        return {
            "status": "COMPLETE" if audit_status == "PASS" else "REVIEW_REQUIRED",
            "ticker": req.ticker,
            "trade_date": req.trade_date,
            "signal": signal,
            "final_trade_decision": state.get("final_trade_decision", ""),
            "audit_status": state.get("audit_status", ""),
            "audit_report": state.get("audit_report", ""),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/discover")
def discover(req: DiscoveryRequest):
    try:
        result = run_discovery(
            req.as_of_date,
            top_n=req.top_n,
            ml_model_path=DEFAULT_CONFIG.get("sector_ml_model_path") or None,
            ml_weight=float(DEFAULT_CONFIG.get("sector_ml_weight", 0.5)),
            style_ic_history_path=(
                DEFAULT_CONFIG.get("style_ic_history_path") or None
            ),
            adaptive_style_strength=float(
                DEFAULT_CONFIG.get("adaptive_style_strength", 0.5)
            ),
        )
        return {
            "as_of_date": result.as_of_date,
            "market_regime": result.market.regime,
            "market_score": result.market.score,
            "rank_source": result.metadata.get("rank_source", "rule"),
            "style_weights": result.metadata.get("style_weights", {}),
            "sectors": result.sectors.sectors.to_dict(orient="records"),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/research-pool")
def research_pool(req: ResearchPoolRequest):
    try:
        result = run_research_pool(
            req.as_of_date,
            sector_top_n=req.sector_top_n,
            representatives_per_sector=req.representatives_per_sector,
            component_limit=req.component_limit,
            strict_pit=req.strict_pit,
            ml_model_path=DEFAULT_CONFIG.get("sector_ml_model_path") or None,
            ml_weight=float(DEFAULT_CONFIG.get("sector_ml_weight", 0.5)),
            style_ic_history_path=(
                DEFAULT_CONFIG.get("style_ic_history_path") or None
            ),
            adaptive_style_strength=float(
                DEFAULT_CONFIG.get("adaptive_style_strength", 0.5)
            ),
        )
        warnings = list(result.representatives.warnings or [])
        component_unavailable = any(
            "COMPONENT_DATA_UNAVAILABLE" in warning
            for warning in warnings
        )
        return {
            "status": (
                "DATA_UNAVAILABLE" if component_unavailable else "COMPLETE"
            ),
            "as_of_date": result.as_of_date,
            "market_regime": result.discovery.market.regime,
            "market_score": result.discovery.market.score,
            "rank_source": result.discovery.metadata.get("rank_source", "rule"),
            "sectors": result.discovery.sectors.sectors.to_dict(orient="records"),
            "representatives": (
                result.representatives.representatives.to_dict(orient="records")
            ),
            "warnings": warnings,
            "user_action_required": (
                "行业发现仍可使用；请等待/更换成分数据源后再生成 Representative Pool。"
                if component_unavailable
                else ""
            ),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        return _conversation_agent().chat(
            req.message,
            thread_id=req.thread_id,
            ticker=req.ticker,
            as_of_date=req.as_of_date,
            force_mode=req.mode,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.get("/chats/recent")
def recent_chats(limit: int = 20):
    return {
        "threads": _conversation_store().list_threads(
            limit=max(1, min(int(limit), 100))
        )
    }


@app.get("/chat/{thread_id}")
def chat_history(thread_id: str, limit: int = 20):
    store = _conversation_store()
    thread = store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="thread not found")
    return {"thread": thread, "messages": store.history(thread_id, limit=max(1, min(int(limit), 100)))}


@app.delete("/chat/{thread_id}")
def reset_chat(thread_id: str):
    deleted = _conversation_store().reset(thread_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="thread not found")
    return {"status": "deleted", "thread_id": thread_id}


@app.get("/chat/{thread_id}/versions")
def research_versions(thread_id: str, limit: int = 20):
    store = _conversation_store()
    if store.get_thread(thread_id) is None:
        raise HTTPException(status_code=404, detail="thread not found")
    return {
        "thread_id": thread_id,
        "versions": store.list_research_versions(
            thread_id,
            limit=max(1, min(int(limit), 100)),
        ),
    }


@app.post("/chat/{thread_id}/rollback")
def rollback_research(thread_id: str, req: RollbackRequest):
    store = _conversation_store()
    if store.get_thread(thread_id) is None:
        raise HTTPException(status_code=404, detail="thread not found")
    restored = store.rollback_research_version(
        thread_id,
        version_id=req.version_id,
    )
    if restored is None:
        raise HTTPException(
            status_code=409,
            detail="no matching previous research version to restore",
        )
    return {
        "thread_id": thread_id,
        "active_version": restored,
    }


@app.get("/knowledge/status")
def knowledge_status():
    if not DEFAULT_CONFIG.get("rag_enabled", False):
        return {
            "status": "disabled",
            "rag_enabled": False,
            "collection": DEFAULT_CONFIG.get("qdrant_collection"),
        }
    try:
        info = _knowledge_store().collection_status()
        return {
            "status": "ready" if info.get("exists") else "empty",
            "rag_enabled": True,
            **info,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "unavailable",
            "rag_enabled": True,
            "error": f"{type(exc).__name__}: {exc}",
        }


@app.get("/knowledge/documents")
def knowledge_documents(
    ticker: str | None = None,
    scope_type: str | None = None,
    scope_key: str | None = None,
    limit: int = 100,
):
    if not DEFAULT_CONFIG.get("rag_enabled", False):
        raise HTTPException(status_code=409, detail="RAG is disabled")
    try:
        canonical = normalize_a_share_symbol(ticker) if ticker else None
        rows = _knowledge_store().list_documents(
            ticker=canonical,
            scope_type=scope_type,
            scope_key=scope_key,
            limit=max(1, min(int(limit), 500)),
        )
        return {
            "ticker": canonical,
            "scope_type": scope_type,
            "scope_key": scope_key,
            "documents": rows,
            "count": len(rows),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


@app.post("/knowledge/search")
def knowledge_search(req: KnowledgeSearchRequest):
    if not DEFAULT_CONFIG.get("rag_enabled", False):
        raise HTTPException(status_code=409, detail="RAG is disabled")
    try:
        canonical = normalize_a_share_symbol(req.ticker)
        hits = _knowledge_retriever().search(
            req.query,
            ticker=canonical,
            as_of_date=req.as_of_date,
            top_k=req.top_k,
            candidate_k=int(DEFAULT_CONFIG.get("rag_candidate_k", 30)),
            corpus_limit=int(
                DEFAULT_CONFIG.get("rag_bm25_corpus_limit", 1000)
            ),
            doc_type=req.doc_type,
            max_chunks_per_doc=int(
                DEFAULT_CONFIG.get("rag_max_chunks_per_doc", 2)
            ),
            industry=req.industry,
        )
        results = []
        for hit in hits:
            chunk = hit.chunk
            metadata = chunk.metadata or {}
            results.append(
                {
                    "evidence_id": (
                        f"RAG:{chunk.doc_id}#chunk-{chunk.chunk_index}"
                    ),
                    "ticker": chunk.ticker,
                    "scope_type": chunk.scope_type,
                    "scope_key": chunk.scope_key,
                    "scope_id": f"{chunk.scope_type}:{chunk.scope_key}",
                    "industry": chunk.industry,
                    "title": chunk.title,
                    "publish_date": chunk.publish_date,
                    "doc_type": chunk.doc_type,
                    "source": str(
                        metadata.get("source_authority") or chunk.source
                    ),
                    "url": str(metadata.get("source_url") or chunk.url),
                    "publish_date_verified": metadata.get(
                        "publish_date_verified"
                    ),
                    "text": chunk.text,
                    "score": hit.score,
                    "dense_score": hit.dense_score,
                    "bm25_score": hit.bm25_score,
                    "rerank_score": hit.rerank_score,
                }
            )
        return {
            "ticker": canonical,
            "as_of_date": req.as_of_date,
            "query": req.query,
            "results": results,
            "count": len(results),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


@app.post("/knowledge/upload")
async def upload_knowledge(
    file: UploadFile = File(...),
    publish_date: str = Form(...),
    ticker: str | None = Form(None),
    scope_type: str = Form("company"),
    scope_key: str | None = Form(None),
    industry: str | None = Form(None),
    doc_type: str = Form("user_document"),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".md", ".markdown"}:
        raise HTTPException(
            status_code=400,
            detail="supported formats: pdf, docx, txt, md",
        )

    max_bytes = int(DEFAULT_CONFIG.get("knowledge_upload_max_mb", 25)) * 1024 * 1024
    payload = await file.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds {max_bytes // (1024 * 1024)} MB limit",
        )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix="tradingagents_knowledge_",
            delete=False,
        ) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        scope = normalize_scope(
            scope_type=scope_type,
            scope_key=scope_key,
            ticker=ticker,
            industry=industry,
        )
        result = ingest_path(
            temp_path,
            ticker=scope.ticker if scope.scope_type == "company" else None,
            publish_date=publish_date,
            config=DEFAULT_CONFIG,
            scope_type=scope.scope_type,
            scope_key=scope.scope_key,
            industry=scope.industry,
            doc_type=doc_type,
            source_name=file.filename or temp_path.name,
            publish_date_source="USER",
            publish_date_confidence=0.5,
            publish_date_verified=False,
        )
        return {
            "status": "indexed",
            "filename": file.filename,
            "pit_notice": (
                "publish_date is user-supplied and unverified; this document "
                "is excluded from historical PIT retrieval until verified"
            ),
            **result,
        }
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
