# MCP + PIT-aware RAG + Qdrant + Docker 扩展说明（v1.3）

本扩展不改变原来的 7-Agent 研究拓扑，新增的是**工具协议层、知识检索层和服务部署层**：

```text
FastAPI / CLI
    │
    ▼
LangGraph 7-Agent
    │
    ├── 本地 Tool（默认兼容模式）
    │
    └── MCP Tool（启用后） ──► Finance MCP Server
                                  ├── 行情/指标
                                  ├── 财务报表
                                  ├── 新闻/宏观
                                  └── PIT-aware RAG
                                           │
                                           ▼
                                      Qdrant
                                      Dense Retrieval
                                           +
                                      BM25 Retrieval
                                           │
                                           ▼
                                         RRF
                                           │
                                           ▼
                                  BGE Cross-Encoder Rerank
```

## 1. 新增能力

- **MCP v2 / Streamable HTTP**：金融数据工具以标准 MCP Tool 方式暴露；LangGraph 通过 `langchain-mcp-adapters` 加载远程工具。
- **PIT-aware RAG**：所有文档必须带 `publish_date`，检索时强制 `publish_date <= as_of_date`，避免历史研究检索到未来公告。
- **Hybrid Retrieval**：Qdrant Dense Retrieval + 本地 BM25，使用 RRF 融合候选。
- **Reranking**：可选 `BAAI/bge-reranker-base` Cross-Encoder 二次排序。
- **Docker Compose**：一键启动 `agent-api + finance-mcp + qdrant` 三个服务。
- **FastAPI**：提供 `/analyze`、`/discover`、`/health` 服务接口。

## 2. 保持向后兼容

普通 CLI 默认仍然：

```text
TRADINGAGENTS_MCP_ENABLED=false
TRADINGAGENTS_RAG_ENABLED=false
```

因此原来的：

```bash
python -m cli.main analyze
python scripts/discover_a_share.py --mode all
```

无需 Qdrant/MCP 也可以继续运行。

启用 MCP 后，如果 MCP Server 暂时不可用，默认：

```text
TRADINGAGENTS_MCP_FALLBACK_TO_LOCAL=true
```

会回退到原来的本地 LangChain Tools，不会因为协议服务异常让整张研究图无法启动。

## 3. 本地安装扩展依赖

```bash
pip install -e '.[agent]'
cp .env.example .env
```

## 4. Docker 一键启动

先在 `.env` 中填写 LLM 与 LangSmith Key，然后：

```bash
docker compose up --build -d
```

服务：

- Agent API: `http://localhost:8000`
- Finance MCP: `http://localhost:8001/mcp`
- MCP health: `http://localhost:8001/health`
- Qdrant: `http://localhost:6333`

首次使用 FastEmbed 时会下载 BGE embedding/reranker ONNX 模型；离线部署前请预先缓存模型，或显式关闭 reranker。

## 5. 向 Qdrant 写入公告/财报知识

### JSONL 方式（推荐）

每行：

```json
{"doc_id":"600519-2025-annual","ticker":"600519.SH","title":"2025年年度报告","text":"完整解析后的报告正文……","publish_date":"2026-03-31","source":"cninfo","url":"https://...","doc_type":"annual_report"}
```

写入：

```bash
python scripts/rag_ingest.py --jsonl examples/rag_knowledge_sample.jsonl
```

Docker：

```bash
docker compose run --rm agent-api \
  python scripts/rag_ingest.py --jsonl examples/rag_knowledge_sample.jsonl
```

### 本地 Markdown/TXT 目录

```bash
python scripts/rag_ingest.py \
  --directory ./knowledge/600519 \
  --ticker 600519.SH \
  --publish-date 2026-03-31 \
  --doc-type annual_report
```

注意：目录模式使用统一 `publish_date`，适合已经按披露日整理好的单批文档。不同披露日请用 JSONL。

## 6. Agent 中启用 MCP + RAG

本地模式：

```text
TRADINGAGENTS_MCP_ENABLED=true
TRADINGAGENTS_MCP_URL=http://localhost:8001/mcp
TRADINGAGENTS_RAG_ENABLED=true
TRADINGAGENTS_QDRANT_URL=http://localhost:6333
```

新闻与情绪分析 Subgraph 会多出：

```text
search_company_knowledge(query, ticker, as_of_date, top_k)
```

该工具不会替代 `get_news`。`get_news` 负责当前窗口正式公告/新闻，RAG 负责从历史长文档知识库中检索与当前问题相关的证据片段。

## 7. RAG 检索链路

```text
query
  │
  ├── BGE-small-zh-v1.5 ─► Qdrant Dense TopK ─┐
  │                                            │
  └── BM25（PIT过滤后的同标的语料）──────────────┤
                                               ▼
                                              RRF
                                               │
                                               ▼
                                   BGE-reranker-base
                                               │
                                               ▼
                                         Evidence TopK
```

PIT 有两层保护：

1. Qdrant `DatetimeRange(lte=as_of_date)`；
2. Retriever 返回前再次检查 `publish_date <= as_of_date`。

## 8. API 示例

候选发现：

```bash
curl -X POST http://localhost:8000/discover \
  -H 'Content-Type: application/json' \
  -d '{"as_of_date":"2026-09-02","sector_count":4,"per_sector":35,"top_n":10}'
```

单股研究：

```bash
curl -X POST http://localhost:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"ticker":"600519.SH","trade_date":"2026-09-02","analysts":["market","news","fundamentals"]}'
```

## 9. 测试

基础测试不依赖 MCP/Qdrant 服务：

```bash
pytest -q
```

新增测试覆盖：

- RAG 不返回研究截止日之后的文档；
- Hybrid Retrieval 对词法相关证据的排序；
- MCP Server 暴露核心金融/RAG工具；
- MCP不可用时本地工具 fallback 架构；
- Docker Compose 包含 Agent/MCP/Qdrant 三服务。

完整在线集成测试需要先启动 Docker 服务并准备 RAG 文档。

## 10. 项目边界

- MCP 是**工具协议标准化**，并不替代 LangGraph；LangGraph 仍负责状态与工作流编排。
- RAG 是**外部知识证据检索**，Context Compression 是**已获取上下文的压缩**，两者职责不同。
- Qdrant 用于 dense vector + payload PIT filter；BM25 在经过相同 PIT 条件过滤的公司语料上计算。
- 当前未引入 K8s 与 RLHF。个人项目目前没有真实多实例弹性伸缩需求；RLHF 也不是现有 Agent 工程问题的自然解法。
