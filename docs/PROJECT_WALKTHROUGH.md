# Project Walkthrough

这份文档从一次真实请求的执行路径出发，解释本仓库的工程边界、核心模块和关键设计决策。它与 `ENGINEERING_NOTES.md` 的侧重点不同：后者强调“为什么这样设计”，本文更强调“请求从哪里进入、经过哪些模块、最后如何产出可审计结果”。

## 1. 项目目标

本项目是一个面向 A 股的研究辅助系统，而不是自动交易系统。核心目标包括：

1. 使用确定性规则从全市场缩小候选范围；
2. 对少量候选运行 7-Agent 深度研究；
3. 对历史研究施加 Point-in-Time 数据约束；
4. 通过独立 Auditor 检查最终结论；
5. 使用 MCP / RAG / FastAPI 解耦数据、知识库与 Agent 工作流；
6. 支持多轮追问，并复用上一轮已审计上下文；
7. 将 Agent 工程评测与市场结果回测分开。

## 2. 与上游 TradingAgents 的边界

上游项目提供了通用多角色金融研究框架。本仓库主要二次开发集中在：

- A 股数据适配与标的规范化；
- 确定性候选发现 Pipeline；
- 7-Agent 拓扑裁剪与两阶段并行化；
- Point-in-Time 数据治理；
- 独立 Decision Auditor；
- Finance MCP 与 Local fallback；
- PIT-aware Hybrid RAG；
- 多轮 Conversation Router + SQLite；
- Agent Evaluation / Outcome Backtest；
- FastAPI、Web UI 与 Docker Compose。

原始作者、项目归属和许可证信息见根目录 `NOTICE`。

## 3. 一次候选发现请求如何执行

入口：

- CLI：`python scripts/discover_a_share.py`
- API：`POST /discover`
- Conversation：discovery intent
- Demo：`python scripts/discovery_agent_demo.py`

核心路径：`tradingagents/discovery/`

执行逻辑：

```text
A-share Universe
    ↓
Market Regime
    ↓
Sector Selection
    ↓
Multi-factor Screening
    ↓
PIT Quality Screen
    ↓
Soft Sector Cap
    ↓
Research Shortlist
```

这里刻意不让 LLM 直接从几千只股票中“挑股票”。排名、阈值、行业配额和财务质量筛选属于结构化问题，交给 Python 更可复现，也更适合回测。

## 4. 一次单股深度研究如何执行

入口：

- CLI：`python -m cli.main analyze`
- API：`POST /analyze`
- Conversation：research intent

核心类：`tradingagents.graph.trading_graph.TradingAgentsGraph`

主链路：

```text
Market Analyst ─────────┐
News & Sentiment ───────┼─> Analyst Fan-In ─┬─> Bull Researcher ─┐
Fundamentals Analyst ───┘                    └─> Bear Researcher ─┤
                                                                  ↓
                                                         Research Fan-In
                                                                  ↓
                                                        Portfolio Manager
                                                                  ↓
                                                         Decision Auditor
                                                            PASS / REVISE
```

执行过程：

1. 根据配置创建 LLM Client；
2. 构建本地工具或 MCP 工具；
3. 编译 LangGraph；
4. 三个 Analyst 在私有 Subgraph 中并行研究；
5. Analyst Fan-In 后，Bull / Bear 并行形成相反研究假设；
6. Portfolio Manager 只做一次证据收敛；
7. Decision Auditor 检查事实依据、PIT、数字一致性和方向一致性；
8. 若审计返回 `REVISE`，在有限轮次内回到 Portfolio Manager；
9. 输出最终研究结论与审计报告。

## 5. 为什么 Analyst 使用私有 Subgraph

对应路径：

- `tradingagents/graph/analyst_subgraph.py`
- `tradingagents/graph/analyst_execution.py`
- `tradingagents/graph/setup.py`

每个 Analyst 拥有独立的 `messages` 通道和 Tool Calling 循环。父图只接收：

- 最终报告；
- 紧凑工具轨迹。

这样可以避免并行执行时不同 Analyst 的 ToolMessage 相互污染，也避免把完整工具对话反复传播到后续节点。

## 6. Point-in-Time 数据如何约束

核心原则：

> 研究日期为 T 时，只允许模型看到 T 时已经公开的信息。

对应路径：

- `tradingagents/dataflows/asof.py`
- `tradingagents/dataflows/`
- `tradingagents/rag/retriever.py`
- `tradingagents/rag/store.py`

主要守卫：

1. 数据 Adapter 在查询阶段限制日期；
2. 财务数据区分报告期与实际可用时间；
3. RAG 查询要求 `publish_date <= as_of_date`；
4. 检索结果进入模型前再次做日期防御性检查；
5. Decision Auditor 检查最终报告是否引用截止日之后的信息。

## 7. Hybrid RAG 如何执行

路径：`tradingagents/rag/`

```text
Dense Retrieval
      +
     BM25
      ↓
     RRF
      ↓
Optional Reranker
      ↓
PIT Defensive Filter
      ↓
Evidence Chunks
```

Dense Retrieval 负责语义召回，BM25 对股票代码、财务术语和公告关键词等精确匹配更稳定。RRF 用于融合两个不同分数尺度的结果。

## 8. Finance MCP 在系统中的位置

路径：

- `tradingagents/mcp/server.py`
- `tradingagents/mcp/client.py`
- `tradingagents/agents/utils/tool_registry.py`

MCP 的作用不是增加 Agent 数量，而是解耦工具部署位置：

- Agent 可以通过标准协议访问金融数据与 RAG Tool；
- Docker 中 Finance MCP 可以独立运行；
- MCP 不可用时保留 Local fallback；
- 外部工具通过 allowlist 控制暴露范围。

## 9. 多轮会话如何避免重复研究

路径：

- `tradingagents/conversation/router.py`
- `tradingagents/conversation/store.py`
- `tradingagents/conversation/agent.py`

会话保存：

- `thread_id`
- 当前 ticker
- `as_of_date`
- 历史消息
- 最近一次研究结果

Router 根据用户输入决定运行 research、discovery、tool chat 或普通追问。普通追问优先复用上一轮已审计上下文，而不是重新跑完整 7-Agent。

## 10. 报告如何落盘

路径：`tradingagents/reporting.py`

当前报告目录与 7-Agent 状态契约一致：

```text
1_analysts/
  market.md
  news.md
  fundamentals.md
2_research/
  bull.md
  bear.md
3_portfolio/
  decision.md
4_audit/
  audit.md
complete_report.md
```

这使 CLI 和程序化调用可以得到一致的可审计输出。

## 11. Evaluation 与 Backtest 为什么分开

路径：

- `tradingagents/evaluation/`
- `tradingagents/backtest/`

Agent Evaluation 关注工程行为，例如：

- Tool 使用是否合理；
- 是否违反 PIT；
- 轨迹是否符合约束；
- 最终报告是否满足结构和证据要求。

Outcome Backtest 则关注研究评级与未来真实收益、基准收益之间的关系。

两者分开可以避免“收益好 = Agent 工程一定好”这种错误归因。

## 12. 服务化入口

路径：`service/app.py`

主要接口：

- `GET /health`
- `POST /analyze`
- `POST /discover`
- `POST /chat`
- `GET /chat/{thread_id}`
- `DELETE /chat/{thread_id}`
- `GET /ui/`

Docker Compose 包含：

```text
agent-api
   |
   +--> finance-mcp
   |       |
   |       +--> qdrant
   |
   +--> sqlite state
```

## 13. 测试策略

公开 CI 优先验证确定性的工程不变量，而不是把真实 LLM 输出作为测试 oracle。

当前重点包括：

- 7-Agent 并行拓扑和 Fan-In；
- Bull / Bear 当前状态契约；
- Decision Auditor 修订路由；
- PIT 数据与 RAG 过滤；
- MCP / Docker wiring；
- Conversation Router / Store；
- Agent Evaluation；
- Outcome Backtest；
- CLI import 与服务 import smoke test。

真实 LLM、外部网络和第三方数据源会引入凭据依赖与非确定性，因此属于运行环境集成验证，而不是公开离线 CI 的主要判定依据。

## 14. 面试时建议怎么讲

建议按下面的顺序，而不是从“我用了几个 Agent”开始：

1. **为什么候选发现用确定性 Python，而不是 LLM 选股**；
2. **为什么历史研究需要 PIT，而不仅是普通 RAG**；
3. **为什么 Analyst 使用私有 Subgraph 并行执行**；
4. **为什么 Bull / Bear 只做一次互补假设，不做多轮复述**；
5. **为什么 Portfolio Manager 后面还需要独立 Auditor**；
6. **为什么 MCP 和 RAG 是基础设施层，而不是业务噱头**；
7. **如何用 Evaluation + Backtest 分离工程质量和市场结果**；
8. **如何通过 Python 3.11/3.12 CI、import smoke test 和 Docker build 保证可复现性**。

如果只阅读代码 15 分钟，建议依次看：

1. `tradingagents/graph/setup.py`
2. `tradingagents/graph/analyst_subgraph.py`
3. `tradingagents/discovery/pipeline.py`
4. `tradingagents/agents/auditors/decision_auditor.py`
5. `tradingagents/rag/retriever.py`
6. `tradingagents/mcp/server.py`
7. `tradingagents/evaluation/`
8. `tradingagents/backtest/`
9. `service/app.py`
