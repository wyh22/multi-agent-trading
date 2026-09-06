# Project-wide RAG Knowledge Layer

RAG 在本项目中的定位是 **全项目共享的外部证据层**，不是针对某只股票写死的补丁，也不是“上传一个 PDF 然后聊天”的简单知识库。

它服务于任意 A 股研究，并和结构化数据工具分工：

- 行情/技术指标：优先结构化 Tool；
- 财务三表/基础财务指标：优先 Fundamentals Tool；
- 长文档、公司经营细节、竞争格局、行业材料、监管政策、重大事项原文：优先 RAG；
- Deep Research / Repair Loop：根据缺失证据按需调用 RAG。

## 1. 知识库分层

每份文档属于一个通用 scope，而不是写死到某个行业。

### company

公司级文档，scope key 是标准股票代码，例如：

~~~text
company:600000.SH
~~~

典型内容包括年报、半年报、季报、公司公告、投资者关系记录、重大公司行动和公司 IR 正式材料。

### industry

行业级文档：

~~~text
industry:银行
industry:半导体
industry:医药生物
~~~

用于行业供需、竞争格局、产业链、行业统计、行业标准和专项政策。公司文档可以附带 industry 元数据，检索某只股票时系统会自动把已知行业 scope 加入召回范围。

### market

A 股全市场共享材料：

~~~text
market:CN_A
~~~

### macro

宏观材料：

~~~text
macro:CN
~~~

### regulation

全国性监管/法规材料：

~~~text
regulation:CN
~~~

因此单股检索不是只查单一股票向量，而是：

~~~text
company:<ticker>
+ industry:<known industry>
+ market:CN_A
+ macro:CN
+ regulation:CN
~~~

并全部受 PIT 截止日期约束。

## 2. 通用股票研究检索维度

Repair query planner 使用通用股票分析维度，不包含任何具体股票或行业规则：

- 主营业务 / 业务模式 / 产品服务 / 产能产量 / 销量订单 / 客户供应商；
- 行业竞争 / 市场份额 / 核心竞争力 / 护城河；
- 财务质量 / 现金流 / 盈利质量 / 资产负债；
- 估值 / 可比公司；
- 增长驱动 / 资本开支 / 在建工程 / 指引；
- 行业周期 / 供需 / 政策 / 监管 / 产业链；
- 并购重组 / 回购 / 增减持 / 股权激励 / 公司治理；
- 经营、行业、政策、财务和合规风险。

Completion Gate 的原始 missing item 也会保留为检索 query，因此即使问题属于冷门行业，也不会被固定 taxonomy 覆盖掉。

## 3. 检索链

~~~text
User / Missing Items
        ↓
Generic Query Expansion
        ↓
Hierarchical Scope Resolution
company + industry + market + macro + regulation
        ↓
PIT Filter
publish_date <= as_of_date
        ↓
Dense Retrieval + BM25
        ↓
RRF
        ↓
Optional Cross-Encoder Rerank
        ↓
Parent-document Diversity
        ↓
Evidence Pack
        ↓
Fundamentals / News / Supervisor / Auditor
~~~

同一个长 PDF 默认最多占据少量 Top-K 位置，避免一个年报的多个相邻页面挤掉其他来源。

每条返回证据包含：

~~~text
evidence_id
scope_type / scope_key
publish_date
source_authority
publish_date_verified
original source URL
doc_type
excerpt
retrieval scores
~~~

## 4. PIT 数据治理

历史研究默认 fail closed：

- publish_date <= as_of_date；
- 新上传且 publish_date_verified=false 的文档不允许进入历史 cutoff；
- 只有确认了真实披露/生效日期，才应设置 publish_date_verified=true；
- 当前日期研究可以使用未核验上传文档，但会保留 unverified provenance；
- company / industry / market / macro / regulation 全部遵守同一 PIT 规则。

不要为了让 RAG 能搜到而随意把日期标成 verified。

## 5. 项目级批量入库：Manifest

项目不要求每只股票改代码。统一使用 JSONL manifest 批量描述语料。每行支持三种来源之一：

~~~text
file    本地 PDF/DOCX/TXT/MD
url     可直接下载的 HTTP/HTTPS 文档
text    已经解析好的正文
~~~

核心字段：

~~~text
scope_type
scope_key
ticker
industry
title
file / url / text
publish_date
doc_type
publish_date_source
publish_date_confidence
publish_date_verified
source_authority
source_url
~~~

其中 company scope 填 ticker；industry scope 填 scope_key/industry；market/macro/regulation 不需要 ticker。industry 对公司文档是可选但推荐字段，用于自动关联行业知识。

示例模板：

~~~bash
cp examples/rag_corpus_manifest.example.jsonl evaluation/data/my_rag_manifest.jsonl
~~~

将示例内容替换为真实文档后：

~~~bash
python scripts/rag_ingest.py --manifest evaluation/data/my_rag_manifest.jsonl
~~~

这是项目级批量入口，不需要为每家公司写 Python 逻辑。

## 6. 单文件和目录仍然支持

单只公司文档：

~~~bash
python scripts/rag_ingest.py   --file ./annual_report.pdf   --scope-type company   --ticker 600000.SH   --industry 银行   --publish-date YYYY-MM-DD   --doc-type annual_report
~~~

行业资料：

~~~bash
python scripts/rag_ingest.py   --file ./industry_report.pdf   --scope-type industry   --scope-key 银行   --publish-date YYYY-MM-DD   --doc-type industry_report
~~~

全国性监管材料：

~~~bash
python scripts/rag_ingest.py   --file ./regulation.pdf   --scope-type regulation   --scope-key CN   --publish-date YYYY-MM-DD   --doc-type regulation
~~~

这些只是 CLI 用法示例，核心代码没有任何特定股票或行业绑定。

## 7. 推荐语料层级

对进入研究 universe 的股票，公司层优先覆盖最近 2~3 年年度报告、最近半年报/季报、最近 6~12 个月的重要公告、投资者关系/业绩说明材料以及重大公司行动和风险公告。

行业层优先覆盖行业政策和标准、行业供需/产能/价格/竞争格局材料、权威行业统计、监管机构或行业协会材料。

市场/宏观/监管层只需建立一份共享库，包括 A 股市场制度与重要监管规则、宏观经济和利率/信用环境材料，以及对上市公司普遍适用的重要政策。

不建议把大量低质量自媒体、论坛或无来源转载直接塞进主证据库。

## 8. Knowledge API：先测 RAG，不必反复跑 LLM

状态：

~~~bash
curl http://127.0.0.1:8000/knowledge/status
~~~

查看全部或按 scope 查看文档：

~~~bash
curl "http://127.0.0.1:8000/knowledge/documents?scope_type=industry"
~~~

查看某公司的 corpus coverage：

~~~bash
curl "http://127.0.0.1:8000/knowledge/coverage?ticker=600000.SH"
~~~

直接检索，不调用 Deep Research：

~~~bash
curl -X POST http://127.0.0.1:8000/knowledge/search   -H 'Content-Type: application/json'   -d '{"ticker":"600000.SH","query":"主营业务 竞争格局 现金流 行业政策 主要风险","as_of_date":"2026-09-06","top_k":8}'
~~~

只有 Knowledge API 的结果稳定后，才需要做一次端到端 Agent 验证。

## 9. 与多 Agent 的集成

RAG 是共享工具：

~~~text
Fundamentals Agent ─┐
News Agent ─────────┼─> search_company_knowledge
Supervisor Skill ───┘
~~~

完整研究后如果存在文档型证据缺口：

~~~text
PARTIAL
  ↓
specialist repair
  ↓
仍有缺口
  ↓
document_evidence_analysis
  ↓
multi-query hierarchical RAG
  ↓
Completion Gate
~~~

RAG 不替代行情、财务 API，也不会为了“用了 RAG”而强制每个问题都走向量库。

## 10. Corpus Coverage

/knowledge/coverage 只用于发现语料缺口，例如是否有 annual reporting、interim reporting、recent company events、公司文档是否带行业元数据，以及 verified / unverified 文档数量。

Coverage 不等于答案质量，也不等于投资结论可信度。最终仍由 Grounding / Completion / Auditor 判断。

## 11. 你真正需要准备的东西

代码层不需要你逐股修改。要让 RAG 产生真实价值，必须有真实 corpus。

最低要求：

1. 启动 Qdrant；
2. 开启 TRADINGAGENTS_RAG_ENABLED=true；
3. 准备一份项目级 manifest，或者一个已整理的官方文档目录；
4. 对文档提供真实披露日和来源；
5. 如果能提供行业标签，检索效果会更完整；不提供也不会阻塞 company + shared scope 检索。

项目不会把演示数据伪装成真实证券证据。
