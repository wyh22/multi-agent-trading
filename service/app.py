from __future__ import annotations

import functools
from pathlib import Path
from datetime import date
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tradingagents.conversation import ConversationAgent, ConversationStore
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.discovery.pipeline import run_discovery, run_research_pool
from tradingagents.graph.trading_graph import TradingAgentsGraph

app = FastAPI(title="TradingAgents A-share Agent API", version="1.4")
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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "1.4",
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
        return {
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
        )
        return {
            "as_of_date": result.as_of_date,
            "market_regime": result.discovery.market.regime,
            "market_score": result.discovery.market.score,
            "rank_source": result.discovery.metadata.get("rank_source", "rule"),
            "sectors": result.discovery.sectors.sectors.to_dict(orient="records"),
            "representatives": (
                result.representatives.representatives.to_dict(orient="records")
            ),
            "warnings": result.representatives.warnings,
        }
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
