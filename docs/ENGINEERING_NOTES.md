# Engineering Notes

本文档从工程实现角度说明本仓库相对于上游 TradingAgents 的主要改造、设计取舍和验证边界。目标是让代码评审者可以快速判断：哪些问题由 Agent 解决，哪些问题刻意交给确定性程序，以及系统如何降低投研场景中的数据穿越和无依据推断。

## 1. 二次开发边界

上游 TradingAgents 提供了多角色金融研究框架和基础数据流。本仓库重点扩展以下部分：

1. A 股数据适配与标的规范化；
2. Sector-first 行业发现 Pipeline + 可选 ML Ranker；
3. 7-Agent LangGraph 裁剪与并行化；
4. Point-in-Time 数据约束；
5. 独立 Decision Auditor；
6. Finance MCP Server / Client fallback；
7. PIT-aware Hybrid RAG；
8. 多轮 Conversation Router + SQLite；
9. FastAPI、Web UI、Docker Compose；
10. 对上述能力的回归测试。

原始作者、项目和许可证信息见根目录 `NOTICE`。

## 2. 设计原则：能确定性计算的，不交给 LLM

横截面行业排序中最容易被“Agent 化过度”的环节包括因子计算、日期、百分位排名和权重融合。这些任务并不需要自然语言推理。旧版股票筛选保留为 legacy 对照，但不再作为默认 Discovery 主链。

因此本项目的职责划分是：

| 任务 | 执行者 |
| --- | --- |
| 行业 Style 因子、横截面排序、Regime 权重 | Python / optional LightGBM |
| PIT 日期检查 | Python |
| 数据源调用与校验 | Tool / MCP |
| 新闻、基本面语义解释 | Analyst Agent |
| 多空观点生成 | Bull / Bear |
| 最终证据收敛 | Portfolio Manager |
| 事实与一致性审计 | Decision Auditor |

这样做的核心收益不是“更智能”，而是更容易测试、更容易复现。

## 3. 为什么改为 Sector-first Discovery

旧版流程先取 Top 行业，再只在这些行业中做股票多因子和财务质量筛选。实际运行后暴露出两个结构问题：

- **Hard Gate**：未进入 Top 行业的股票在个股评分前就被删除，成长股、高股息股等可能因为行业排序失去研究机会；
- **双重行业暴露**：Sector Score 既决定股票是否进入候选池，又再次进入个股 Final Score。

V2 将 Discovery 的主问题重新定义为“哪些申万一级行业值得优先研究”，所有行业先进入同一横截面，再分别计算 Momentum / Value / Dividend / Liquidity 四类 Style Score。Market Regime 只调整 Style 权重，不再决定行业资格。

默认输出：

```text
Market Regime
  -> SW L1 full cross-section
  -> Style Scores
  -> Regime-aware Rule Rank
  -> Optional LightGBM Ranker
  -> Top-K Sector Research Shortlist
```

LightGBM 只是可插拔二阶段 Ranker，Rule Score 永远保留用于解释和 fallback；仓库不内置未经 Walk-forward 验证却宣称有效的预训练模型。旧股票筛选通过 `run_stock_discovery_legacy` 保留用于 A/B 对比。

详细设计见 `docs/SECTOR_DISCOVERY.md`。

## 4. 为什么 Representative Pool 只负责研究路由

Sector Discovery 输出的是行业研究优先级，但现有 7-Agent 是单股研究图，因此中间需要一个低成本桥接层。该层不再做“哪只股票最好”的统一质量评分，而是只选择行业内适合投入深度研究预算的代表性公司：

~~~text
Representative Score
= 35% 行业指数权重
+ 30% 流动性
+ 20% 行业内相对强弱
+ 15% 数据完整性
~~~

没有 PE/PB、ROE、利润增速或 Quality Score。这样避免刚修掉 Sector hard gate 后，又在桥接层重新引入一套隐藏的跨行业选股逻辑。

每个 Representative Entry 会生成 `research_context`，作为 `candidate_context` 进入 LangGraph。该上下文只解释研究来源，并带有明确 anti-confirmation-bias guard；最终评级仍必须来自 Analyst Tool Evidence。Checkpoint signature 还包含该 context 的 hash，避免同 ticker/date 在不同研究先验下错误 resume。

## 5. 为什么重构 Agent 拓扑

原始多角色链路存在大量相近上下文的重复读取与自然语言复述。随着中间报告增多，Token 和延迟都会累积，同时错误事实也可能在角色之间继续传播。

当前单股研究主链路：

```text
Market ─┐
News ───┼─> Analyst Fan-In ─┬─> Bull ─┐
Fund. ──┘                   └─> Bear ─┼─> Portfolio Manager ─> Auditor
                                      └───────────────────────────────
```

关键点：

- 三个 Analyst 各自在私有 Subgraph 中完成 LLM ↔ ToolNode 循环；
- Analyst 阶段并行；
- Bull / Bear 阶段并行；
- Bull / Bear 不再进行多轮互相复述；
- Auditor 不创造新事实，只检查最终结论；
- Auditor 失败时只允许有限次数回到 Portfolio Manager 修订。

对应代码：

- `tradingagents/graph/setup.py`
- `tradingagents/graph/analyst_subgraph.py`
- `tradingagents/agents/auditors/decision_auditor.py`

## 6. Point-in-Time：历史研究最重要的数据约束

历史研究不能使用“今天已经知道、但当时尚未披露”的信息。

本项目把 `as_of_date / trade_date` 作为研究截止时间，在多个层级做守卫：

1. 数据 Adapter 在查询时限制日期；
2. 财务数据区分报告期与实际披露时间；
3. RAG 查询要求 `publish_date <= as_of_date`；
4. Hybrid RAG 在存储过滤后再做一次日期防御性检查；
5. Auditor 检查最终报告是否引用截止日后的信息；

这是本项目比“让多个 Agent 讨论股票”更重要的工程点之一。

## 7. Hybrid RAG 的检索路径

当前 RAG 采用：

```text
Dense Retrieval
      +
BM25 Sparse Retrieval
      ↓
RRF Fusion
      ↓
Optional Reranker
      ↓
PIT defensive filter
      ↓
Top-K evidence
```

核心实现：`tradingagents/rag/retriever.py`。

这里选择 Hybrid Retrieval 的原因是：金融文档同时包含自然语言语义和大量股票代码、指标名、公告术语。仅向量检索容易弱化精确关键词，仅 BM25 又缺乏语义召回。

## 8. MCP 的作用不是“为了用 MCP”

Finance MCP 的主要价值是把金融工具从 Agent 运行时解耦出来：

- Agent 可以通过标准协议调用工具；
- Docker 中 finance-mcp 可以独立部署；
- MCP 不可用时允许 Local fallback；
- 外部 MCP 工具默认不直接暴露，需 allowlist；

对应代码：

- `tradingagents/mcp/server.py`
- `tradingagents/mcp/client.py`
- `tradingagents/agents/utils/tool_registry.py`

## 9. 为什么增加独立 Decision Auditor

如果 Portfolio Manager 同时负责生成和验证最终结论，本质上仍然是“自己检查自己”。

Auditor 的提示词明确限制：

- 不调用工具；
- 不补充新事实；
- 只读取上游证据和最终结论；
- 检查事实依据、PIT、数字一致性、方向一致性；
- 只有实质问题才返回 REVISE。

这种设计不能消灭 LLM 错误，但能把“结论生成”和“结论校验”拆成两个职责。

## 10. 多轮会话为什么复用已审计上下文

普通追问不应该每次都重新运行完整研究图。

Conversation Layer 保存：

- `thread_id`
- 当前 ticker
- `as_of_date`
- 历史消息
- 最近一次研究结果

Router 决定用户请求属于：

- research
- discovery
- tool_chat
- 普通追问

对于普通追问，优先复用上一轮已审计结果，避免重复成本和上下文漂移。

## 11. 可复现性

仓库提供三层运行方式：

1. Python editable install；
2. FastAPI；
3. Docker Compose（Agent API + Finance MCP + Qdrant）。

CI 在 Python 3.11 / 3.12 上执行：

```bash
python -m compileall -q tradingagents service scripts
pytest -q
```

并额外执行 Docker image build，防止 Dockerfile 与仓库目录结构漂移。

## 12. 当前验证边界

离线回归测试数量会随功能增加而变化，因此不再把历史固定的 `55 passed, 1 skipped` 当作当前状态。当前状态以 GitHub Actions 的 Python 3.11/3.12、Compile、Import Smoke、Pytest 与 Docker Build 为准。

仍需要外部环境才能完整验证的部分：

- 实际 LLM Provider；
- 外部第三方 MCP；
- 实时行情接口稳定性；
- 完整 Docker 在线依赖拉取。

因此 README 中没有把单元测试结果包装成“生产可用”结论。

## 13. 代码评审时建议重点阅读

如果只阅读 15 分钟，建议按以下顺序：

1. `tradingagents/graph/setup.py`：理解 7-Agent 主拓扑；
2. `tradingagents/graph/analyst_subgraph.py`：理解私有 Analyst Subgraph；
3. `tradingagents/discovery/pipeline.py`：理解 Sector-first Discovery；
4. `tradingagents/discovery/sectors.py` / `representatives.py` / `sector_ranker.py`：理解 Style Rank 与可选 LightGBM；
5. `tradingagents/agents/auditors/decision_auditor.py`：理解独立审计；
6. `tradingagents/rag/retriever.py`：理解 Hybrid + PIT；
7. `tradingagents/mcp/server.py`：理解工具协议层；
8. `tradingagents/conversation/agent.py`：理解会话路由；
9. `service/app.py`：理解服务接口。

## 14. Claim-aware Context Compression

旧实现只按字符预算保留报告头尾，虽然能限制上下文长度，但无法区分“直接证据”和“模型解释”。当前实现增加显式 Claim 层：

```text
Analyst full report
      ↓
Evidence Claims
      ├─ FACT
      ├─ CALCULATION
      ├─ INFERENCE
      └─ CONDITIONAL
      ↓
typed budget selection
      ↓
Bull / Bear → Portfolio Manager → Auditor
```

实现路径：

- `tradingagents/agents/utils/evidence_claims.py`：Claim 类型、显式标签解析、旧报告规则分类、预算选择；
- `tradingagents/agents/utils/context_compaction.py`：构造 Analyst / Decision 紧凑证据包；
- 三类 Analyst 在最终报告末尾生成 4~8 条类型化 Evidence Claims；
- Bull/Bear、Portfolio Manager、Auditor 共享同一组 Claim 语义约束。

压缩优先级不是“置信度评分”。FACT / CALCULATION 获得更高预算优先级是因为它们构成 Grounding 层；INFERENCE 仍可保留，但不得被下游重述为事实；CONDITIONAL 必须保留原始触发条件。若旧报告没有显式标签，则使用保守的确定性规则兜底，因此该能力不会为了压缩再新增一次 LLM 调用。

## 15. 下一阶段可以继续量化的指标

后续最值得补的不是继续增加 Agent 数量，而是把工程收益量化：

- 串行拓扑 vs 并行拓扑的平均延迟；
- 单次研究的输入/输出 Token；
- Auditor 的 REVISE 触发率；
- RAG Recall@K / MRR；
- PIT 泄漏测试样例数量；
- 数据源 fallback 成功率；
- 在线调用错误率与 P95 latency。

这些指标能比“有多少个 Agent”更真实地反映系统质量。
