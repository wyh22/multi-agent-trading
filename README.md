# A 股自动投研 Agent

> 基于开源 TradingAgents 二次开发的 A 股多智能体智能投研与候选发现系统。  
> 项目定位是**候选发现 + 证据约束的深度研究**，不执行自动交易，不构成投资建议。

## 项目亮点

- **7-Agent LangGraph 工作流**：市场、新闻/情绪、基本面三类 Analyst 以独立 Subgraph 并行执行，Bull/Bear 并行研判，Portfolio Manager 汇总，Auditor 独立审计。
- **Tool Calling / ToolNode**：子图内部通过 LLM↔ToolNode 循环动态调用金融数据工具。
- **PIT 数据治理**：按实际披露/发布日期限制历史可见数据，避免未来数据泄漏。
- **Finance MCP Server**：以 Streamable HTTP 标准化暴露金融数据与 RAG Tool，支持 MCP Client、Local fallback 和外部 Tool allowlist。
- **PIT-aware Hybrid RAG**：Qdrant Dense Retrieval + BM25 + RRF + 可选 Reranker，并基于 publish_date≤as_of_date 过滤。
- **多轮会话**：Conversation Router + thread_id + SQLite，普通追问复用上一轮已审计研究上下文。
- **工程化服务**：FastAPI 提供 /analyze、/discover、/chat，Docker Compose 编排 agent-api、finance-mcp 与 qdrant。
- **可观测与验证**：LangSmith Trace 用于观察 Agent/LLM/Tool 调用链；v1.4 离线回归结果为 **55 passed, 1 skipped**。

## 架构概览

```text
A股候选池
  ↓
确定性候选发现
  ├─ Market Regime
  ├─ 申万行业筛选
  ├─ Quant Screening
  ├─ PIT Quality Screen
  └─ Soft Sector Cap
  ↓
Top-N Research Shortlist
  ↓
LangGraph
  ├──────────────┬──────────────┐
  ↓              ↓              ↓
Market        News           Fundamentals
Analyst       Analyst        Analyst
  └──────────────┴──────────────┘
                 ↓
              Fan-In
          ┌──────┴──────┐
          ↓             ↓
        Bull           Bear
          └──────┬──────┘
                 ↓
       Portfolio Manager
                 ↓
             Auditor
          PASS / REVISE
                 ↓
                END
```

## 二次开发边界

本项目基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) 二次开发。TradingAgents 提供原始多角色单股研究框架；本仓库重点完成 A 股适配、候选发现、LangGraph 工作流裁剪/并行化、Auditor、MCP/RAG、多轮会话、FastAPI/Docker 以及评测与回归测试等扩展。

原项目采用 Apache License 2.0，本仓库保留原许可证并在 `NOTICE` 中说明二次开发内容。

## 安全说明

公开仓库不包含 `.env`、API Key、iFinD refresh token、LangSmith Key 等敏感凭据。请从 `.env.example` 创建本地配置，并只在本地填写真实密钥。

## 快速开始

推荐 Python 3.11/3.12：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

数据源预检：

```bash
python scripts/check_data_sources.py --ticker 601016
```

候选发现：

```bash
python scripts/discover_a_share.py --mode all --date 2026-08-20 --sectors 4 --per-sector 35 --top 10
```

单股深度研究：

```bash
python -m cli.main analyze
```

FastAPI：

```bash
uvicorn service.app:app --host 0.0.0.0 --port 8000
```

Docker Compose：

```bash
docker compose up --build
```

## 验证

```bash
pytest -q
```

构建环境验证结果：

```text
55 passed, 1 skipped
```

该结果为离线回归测试，不代表真实 LLM、第三方 MCP、iFinD 或完整 Docker 在线集成已全部完成生产验证。

## 详细文档

- `FINAL_ARCHITECTURE.md`：7-Agent、Subgraph、Fan-Out/Fan-In、Auditor
- `MCP_RAG_DOCKER_GUIDE.md`：MCP、Qdrant Hybrid RAG、Docker
- `V1.4_CONVERSATION_IFIND_GUIDE.md`：多轮会话与 iFinD Adapter
- `V1.4_VALIDATION.md`：离线验证边界

## 项目边界

- 候选发现输出为 Research Shortlist，不是收益承诺。
- LLM 主要负责语义分析、工具选择、观点综合与审计；数值筛选尽量交给确定性 Python。
- 历史选股仍受到历史成分数据完整性与幸存者偏差约束。
- 项目重点是 Agent 工程、PIT 数据治理、RAG、MCP 与证据约束研究，不以预测未来价格为目标。
