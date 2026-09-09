# document_evidence_analysis

## Purpose

针对已经进入共享知识库的年报、半年报、公告、投资者关系记录等长文档执行 PIT-aware 证据检索，用于回答文档型问题或补齐上一轮研究中的证据缺口。

## When to use

- 用户明确要求基于财报、公告或投资者关系材料回答。
- Market / News / Fundamentals 已执行，但仍缺少经营细节、资本开支、风险、行业政策等长文档证据。
- 上一轮结果为 PARTIAL / REVIEW_REQUIRED，且缺口适合通过 RAG 定向补查。

## Inputs

- `ticker`: 股票代码。
- `as_of_date`: 研究截止日期。
- `query` / `queries`: 检索问题；缺省时由 repair query planner 生成。
- `top_k`: 每个 query 的候选证据数，可选。

## Execution

1. 将问题扩展为有限数量的检索 query。
2. 在 company + industry + market + macro + regulation scope 中检索。
3. 对所有文档执行发布日期 PIT 过滤。
4. 使用 Dense Retrieval + BM25 + RRF + 可选 Reranker。
5. 汇总带 provenance 的证据，再由对话层生成基于证据的回答。

## Output

- 带来源、发布日期和摘录语义的 Evidence Pack。
- 如未检索到证据，显式返回缺失状态，而不是推断“事实不存在”。

## Constraints

- `publish_date <= as_of_date`。
- 历史研究对显式未验证发布日期的文档 fail closed。
- RAG 是证据层，不替代行情/财务结构化 Tool。
