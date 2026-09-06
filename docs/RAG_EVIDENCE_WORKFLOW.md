# RAG Evidence Workflow

本项目的 RAG 定位不是“聊天知识库”，而是 **Deep Research 的长文档证据补充层**。

结构化行情/财务 API 擅长价格、财务报表和指标；RAG 负责补充：

- 主营业务构成、装机容量、发电量、利用小时、运营规模；
- 项目布局、竞争优势、行业地位；
- 年报/半年报中的风险披露；
- 风电行业政策、消纳/弃风限电、补贴与电价机制；
- 其他只能从公告、财报和政策原文获得的长文本证据。

## 1. 推荐的最小知识库

针对单只股票，先准备少量高价值官方文档即可，不需要一开始抓全量公告。

公司级（ticker=股票代码）建议：

1. 最近一份年度报告；
2. 最近一份半年报；
3. 最近 1~2 份季度报告；
4. 最近的发电量/经营数据公告；
5. 重要项目建设、投产、并网公告；
6. 与补贴应收、资产减值、重大风险有关的正式公告。

共享政策级（ticker=GLOBAL）建议：

1. 国家能源局/发改委等风电、新能源规划；
2. 新能源消纳、弃风限电相关政策/统计；
3. 可再生能源补贴、电价机制相关政策；
4. 对风电运营企业有普遍影响的行业规则。

GLOBAL 文档只入库一次，检索任意公司时会自动与该公司的文档一起召回。

## 2. 本地启动 Qdrant

只需要 RAG 时，无需先启 MCP：

~~~bash
docker run -d \
  --name tradingagents-qdrant \
  -p 6333:6333 \
  -v tradingagents_qdrant:/qdrant/storage \
  qdrant/qdrant:latest
~~~

.env：

~~~text
TRADINGAGENTS_RAG_ENABLED=true
TRADINGAGENTS_QDRANT_URL=http://localhost:6333
TRADINGAGENTS_QDRANT_COLLECTION=a_share_knowledge
TRADINGAGENTS_RAG_EMBEDDING_BACKEND=fastembed
TRADINGAGENTS_RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
TRADINGAGENTS_RAG_RERANKER_ENABLED=true
TRADINGAGENTS_RAG_MAX_CHUNKS_PER_DOC=2
TRADINGAGENTS_RAG_REPAIR_MAX_QUERIES=4
~~~

MCP 可以继续保持：

~~~text
TRADINGAGENTS_MCP_ENABLED=false
~~~

RAG 和 MCP 是两层独立能力。

## 3. 入库官方文档

### 3.1 已下载到本地的官方 PDF

只有在你已经核验官方披露日后，才使用 --publish-date-verified：

~~~bash
python scripts/rag_ingest.py \
  --file ./knowledge/601016/annual_report.pdf \
  --ticker 601016.SH \
  --publish-date YYYY-MM-DD \
  --doc-type annual_report \
  --publish-date-source CNINFO \
  --publish-date-confidence 1.0 \
  --publish-date-verified \
  --source-authority CNINFO \
  --source-url "官方原文URL"
~~~

### 3.2 直接从官方 URL 下载并入库

~~~bash
python scripts/rag_ingest.py \
  --url "官方PDF URL" \
  --ticker 601016.SH \
  --publish-date YYYY-MM-DD \
  --doc-type operating_announcement \
  --publish-date-source CNINFO \
  --publish-date-confidence 1.0 \
  --publish-date-verified \
  --source-authority CNINFO
~~~

### 3.3 行业政策只入库一次

~~~bash
python scripts/rag_ingest.py \
  --url "官方政策PDF URL" \
  --ticker GLOBAL \
  --publish-date YYYY-MM-DD \
  --doc-type policy \
  --publish-date-source NEA \
  --publish-date-confidence 1.0 \
  --publish-date-verified \
  --source-authority "国家能源局"
~~~

## 4. PIT 规则

历史研究不会仅因为“用户填了一个日期”就相信该日期。

- publish_date_verified=false：历史 cutoff 下 fail closed，不进入检索结果；
- publish_date_verified=true：可以用于其披露日之后的历史研究；
- 当前日期研究可以使用未核验上传文档，但结果会标记 unverified；
- GLOBAL 文档同样受 publish_date <= as_of_date 约束。

因此 --publish-date-verified 不是方便开关，而是数据治理声明。

## 5. 不用 LLM，先检查知识库

为了避免反复运行完整 Agent 浪费时间，先用便宜的确定性接口检查。

### 状态

~~~bash
curl http://127.0.0.1:8000/knowledge/status
~~~

### 查看某只股票已入库文档

~~~bash
curl "http://127.0.0.1:8000/knowledge/documents?ticker=601016.SH"
~~~

### 直接检索经营证据

~~~bash
curl -X POST http://127.0.0.1:8000/knowledge/search \
  -H 'Content-Type: application/json' \
  -d '{
    "ticker":"601016.SH",
    "query":"主营业务 装机容量 发电量 利用小时 运营规模",
    "as_of_date":"2026-09-06",
    "top_k":6
  }'
~~~

### 直接检索政策/弃风/补贴

~~~bash
curl -X POST http://127.0.0.1:8000/knowledge/search \
  -H 'Content-Type: application/json' \
  -d '{
    "ticker":"601016.SH",
    "query":"风电 政策 弃风限电 消纳 补贴 国补",
    "as_of_date":"2026-09-06",
    "top_k":6
  }'
~~~

第二个查询会同时搜索：

~~~text
601016.SH company documents
+
__GLOBAL__ shared policy documents
~~~

## 6. 检索链

~~~text
Missing Items
    ↓
deterministic repair query expansion
    ↓
Company scope + GLOBAL scope
    ↓
Dense retrieval + BM25
    ↓
RRF
    ↓
optional cross-encoder rerank
    ↓
parent-document diversity cap
    ↓
Evidence Pack
~~~

同一个长年报的多个页面不会再轻易占满全部 Top-K；默认每个 parent document 最多返回 2 个 chunk。

每条证据带稳定标识：

~~~text
evidence_id=RAG:<doc_id>#chunk-<index>
publish_date
source authority
verified/unverified/legacy
original source URL
~~~

便于 Grounding/Auditor 追踪。

## 7. 与 repair loop 的关系

RAG 启用后，补查顺序设计为：

~~~text
PARTIAL
  ↓
Fundamentals / News specialist
  ↓
仍有业务经营、政策等长文档缺口
  ↓
document_evidence_analysis
  ↓
focused multi-query RAG Evidence Pack
  ↓
Completion Gate
~~~

不会为了补缺口重新跑完整 Deep Research，也不会使用 company_comparison 之类无关能力。

## 8. 什么时候才跑完整 Agent

建议只在以下条件满足后做一次端到端验证：

- /knowledge/status 为 ready；
- /knowledge/documents 能看到预期公司文档；
- 两个 /knowledge/search 查询都能返回合理证据；
- 证据披露日和来源核验无误。

此时再运行一次 Deep Research + “继续补查”，观察 Completion/Evidence Coverage 是否改善即可。

这比反复运行 LLM/Agent Smoke Test 更省时间，也更容易定位问题。
