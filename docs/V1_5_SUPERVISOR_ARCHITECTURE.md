# V1.5 Conversation-first Supervisor Architecture

## 目标

V1.5 不再把固定 7-role LangGraph 当成所有用户请求的默认入口。用户始终通过 Conversation 层提出目标，
Supervisor 根据任务复杂度和已有上下文选择最小必要能力：

1. `respond`：只解释已有已审计上下文，不获取新事实；
2. `call_tool`：行情、财务、公告、宏观、共享 RAG 等原子事实查询；
3. `delegate_agent`：Market / News / Fundamentals 专业 Agent，继续使用各自私有 Tool Subgraph；
4. `run_skill`：Sector Discovery、Document Evidence Analysis、Company Comparison 等任务级能力；
5. `run_deep_research`：复杂综合研究才进入完整多角色研究图；
6. `rollback`：恢复上一版或指定研究版本。

高层路由由结构化 LLM 决策；ticker/date/PIT、安全边界和 fallback 保持 deterministic。对原子 Tool / Specialist Agent，Supervisor 默认最多执行 3 步有界 `Decide → Execute → Observe → Re-decide`，并记录 `supervisor_trace`；重复 capability 会触发 repeat guard。Skill、Deep Research 和 Rollback 仍是终止动作。

## 为什么不是所有请求都跑 Multi-Agent

简单价格、单项财务指标、单份公告等问题用原子 Tool 即可。单领域复杂分析委派专业 Analyst。
只有跨市场/新闻/基本面、需要多假设综合与审计的请求才使用 Deep Research Skill。

这样保留多 Agent 的角色/工具/上下文隔离收益，同时避免固定全图带来的 Token 和延迟浪费。

## Capability 层

Supervisor 看到的是 Capability Registry，而不是底层 Python 实现细节。

- Tool：原子、低成本事实/计算能力；
- Agent：带 Action-Observation-ReDecision Tool Loop 的专业执行者；
- Skill：任务级执行规范，可组合 Tool / Agent / deterministic pipeline。

Skill 使用声明式 manifest 描述用途、依赖 Agent 和完成条件。Sector Discovery Skill 内部仍使用确定性 Python 排名，
Skill 不意味着将数值计算交给 LLM。

## Shared RAG

`search_company_knowledge` 从 News 专属工具升级为共享只读 Knowledge Capability：

- Conversation Supervisor 可直接使用；
- News Agent 可检索公告、政策和事件材料；
- Fundamentals Agent 可检索年报、财务附注、会计政策、减值等非结构化证据。

检索仍采用 Dense + BM25 + RRF + optional reranker，并在 Qdrant filter 与检索后两层执行
`publish_date <= as_of_date`。

知识摄取支持：

- PDF：PyMuPDF，保留 page metadata；
- DOCX：python-docx，保留标题/表格文本；
- TXT / Markdown；
- SHA256 file hash 作为去重/版本追踪元数据。

扫描型 PDF 当前明确拒绝为空文本，不默认启用 OCR，避免静默产生低质量知识。

## Audit-driven Repair

V1.4：

```text
Portfolio Manager -> Auditor -> REVISE -> Portfolio Manager
```

V1.5：

```text
Portfolio Manager
      |
    Auditor
      |
  AuditIssue
   /  |  \
Market News Fundamentals
   \  |  /
 Portfolio Manager
      |
    Auditor
```

`AuditIssue` 明确包含 `issue_type`、`repair_target`、`affected_claims` 与 `instruction`。
专业 Agent 收到修订目标后重新调用自己的 Tool/RAG 取证，而不是由无工具的 PM 猜测缺失事实。

## Checkpoint vs Rollback

两者是不同语义：

- LangGraph Checkpoint：运行过程异常后的 crash resume；
- Research Version Rollback：已经产生研究版本后，将 active pointer 恢复到上一版/指定版本。

Research Version 采用 append-only/immutable 设计，Rollback 不删除历史版本。

## 当前边界

V1.5 有 Conversation-level dynamic routing 与有界执行后再决策，但没有宣称所有复杂任务都已经具备通用自由 DAG Planning。
复杂单股研究仍使用可审计的 Deep Research Skill；后续只有在 benchmark 证明需要时，才进一步引入动态 Task DAG / Replanning。

这一边界是刻意设计：金融研究优先可复现、PIT、证据链与故障可控，而不是追求最大自治程度。
