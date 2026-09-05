# Interview Guide — A股 Sector Discovery + 7-Agent Research System

> 目标：用这份文档在面试前 30~60 分钟快速复习项目。  
> 项目定位：**研究辅助系统，不是自动交易系统**。  
> 核心原则：**Deterministic where possible, evidence first, auditable by design.**

---

## 1. 90 秒项目介绍

可以直接这样回答：

> 这个项目基于 TradingAgents 做了比较深的二次开发。我没有继续堆更多 Agent，而是把问题拆成两层：前面用确定性 Python / 可选量化 Ranker 做 A 股行业研究优先级发现，后面只对少量代表性股票运行高成本的 7-Agent 深度研究。
>
> Discovery 侧先判断 Market Regime，再对全部申万一级行业计算 Momentum、Value、Dividend、Liquidity 四类 Style Score；Regime 只动态调整 Style 权重，不再像旧版那样用 Top Sector 对股票做硬门控。可选 LightGBM 作为二阶段横截面 Ranker，但 Rule Score 永远保留用于解释和 fallback。
>
> Top-K 行业之后，我增加了 Representative Research Pool：每个行业只根据行业指数权重、流动性、行业内相对强弱和数据完整性选 2~3 只研究入口，不用 PE、ROE、利润增长等指标再次偷偷做“选股”。这些代表股再进入 7-Agent。
>
> 7-Agent 主链是 Market / News / Fundamentals 三个 Analyst 并行，之后 Bull / Bear 并行，再由 Portfolio Manager 收敛，Decision Auditor 做独立审计。Analyst 使用私有 LangGraph Subgraph 和 ToolNode，支持 Local Tool / Finance MCP / PIT-aware RAG。
>
> 为了降低 Agent 之间的上下文重复，我还做了 Claim-aware Context Compression，把中间声明分成 FACT、CALCULATION、INFERENCE、CONDITIONAL；同时用 as_of_date / publish_date 做 PIT 数据约束，避免历史研究读取未来信息。
>
> 最后我把 Agent Evaluation 和 Outcome Backtest 分开：前者评估 Tool Choice、PIT、Trajectory、Grounding，后者才看实际收益和基准超额，避免把“市场偶然涨跌”误当成 Agent 工程质量。

---

## 2. 一张图讲清系统

~~~text
                         A股市场
                            |
                            v
                    Point-in-Time Data
                            |
                            v
                      Market Regime
                            |
                            v
                 申万一级行业全横截面
          +-----------------+------------------+
          |                 |                  |
          v                 v                  v
      Momentum          Value/Dividend      Liquidity
          \                 |                  /
           +----------------+-----------------+
                            |
                            v
                Regime-aware Rule Rank
                            |
                    [Optional LightGBM]
                            |
                            v
                Top-K Sector Shortlist
                            |
                            v
              Representative Research Pool
       Index Weight / Liquidity / Relative Strength /
                    Data Completeness
                            |
                            v
                Representative Stocks
                            |
                            v
                  7-Agent LangGraph
       +--------------------+--------------------+
       |                    |                    |
       v                    v                    v
 Market Analyst       News Analyst      Fundamentals Analyst
       \                    |                    /
        +------------- Analyst Fan-In ----------+
                            |
                 +----------+----------+
                 |                     |
                 v                     v
          Bull Researcher        Bear Researcher
                 \                     /
                  +---- Research Fan-In
                            |
                            v
                  Portfolio Manager
                            |
                            v
                  Decision Auditor
                     PASS / REVISE
                            |
                            v
                  Final Research Report

Cross-cutting:
- Claim-aware Context Compression
- Finance MCP
- Qdrant Dense + BM25 + RRF + optional reranker
- PIT data guard
- LangSmith trace
- Agent Evaluation
- Outcome Backtest
~~~

---

## 3. 为什么先做行业发现，而不是直接让 LLM 选股票

### 第一版的问题

旧版：

~~~text
Market Regime
 -> Top 4 Sector
 -> 每行业 35 只
 -> Stock Quant
 -> PIT Quality
 -> Top10
~~~

实际运行后暴露出两个明显结构问题。

### 3.1 Sector hard gate

如果一个行业排名第 5：

~~~text
行业第 5
 -> 不进入 Top 4
 -> 该行业所有股票都被删除
~~~

即使里面有非常强的成长股，也没有后续研究机会。

### 3.2 双重 Sector Exposure

旧版里：

1. Sector Score 先决定股票能否进池；
2. Sector Score 又进入股票 Final Score。

行业因素被计算了两次。

### V2 解决方式

现在：

> **所有申万一级行业先参加同一个横截面排名。**

Market Regime：

> 只改变 Style 权重，不改变行业资格。

这使科技成长和高股息资产可以通过不同路径进入研究池。

---

## 4. Sector Discovery 具体实现

核心路径：

~~~text
tradingagents/discovery/
  market.py
  sectors.py
  sector_ranker.py
  pipeline.py
~~~

### 4.1 Market Regime

文件：

~~~text
tradingagents/discovery/market.py
~~~

指数集合包括：

- 上证综指
- 沪深300
- 中证500
- 中证1000
- 创业板指

计算：

- 5/20/60 日收益；
- MA20 / MA60 gap；
- 20 日年化波动；
- 60 日最大回撤。

每个指数得到 0~100 的 index score，再取平均：

~~~text
score >= 60 -> Risk-On
score <= 40 -> Risk-Off
otherwise   -> Neutral
~~~

这里是确定性计算，不调用 LLM。

---

## 5. 四类 Sector Style

文件：

~~~text
tradingagents/discovery/sectors.py
~~~

### Momentum

主要由：

- 当日涨跌；
- 20 日收益；
- 60 日收益。

形成横截面 percentile score。

意义：

> 让科技、成长、周期行业能因为趋势强度进入研究池。

### Value

主要由：

- PE；
- PB。

这里是同一天申万一级行业横截面便宜度，不是“绝对估值真理”。

### Dividend

行业股息率横截面百分位。

意义：

> 给银行、煤炭、能源、公用事业等高股息行业独立的 Style 通道。

### Liquidity

主要由：

- 换手率；
- 成交额占比。

反映资金活跃度。

---

## 6. Regime-aware Weight

当前 baseline：

### Risk-On

~~~text
Momentum   55%
Value      10%
Dividend    5%
Liquidity  30%
~~~

### Neutral

~~~text
Momentum   40%
Value      20%
Dividend   15%
Liquidity  25%
~~~

### Risk-Off

~~~text
Momentum   25%
Value      20%
Dividend   35%
Liquidity  20%
~~~

面试时一定补一句：

> 这些是工程 baseline，不宣称是历史最优参数。真正生产使用应该通过 Walk-forward Validation 调整。

这是一个非常重要的诚实边界。

---

## 7. 为什么加 Optional LightGBM，而不是 RL

文件：

~~~text
tradingagents/discovery/sector_ranker.py
~~~

LightGBM 不作为强制依赖。

安装：

~~~bash
pip install -e ".[quant]"
~~~

### Feature Contract

包括：

- change_pct
- ret_20d
- ret_60d
- turnover
- amount_share
- pe
- pb
- dividend_yield
- momentum_score
- valuation_score
- dividend_score
- liquidity_score
- rule_score

### 为什么不是直接预测精确收益

项目真正关心的是：

> 同一天 31 个申万一级行业，谁更值得优先研究。

本质更接近：

> Cross-sectional Ranking / Learning to Rank。

所以 ML raw prediction 会先转换成：

> 当天行业横截面 percentile score。

然后：

~~~text
Final Sector Score
 = (1 - ml_weight) * Rule Score
 + ml_weight * ML Score
~~~

Rule Score 永远保留。

### 为什么不是 FinRL / PPO

面试回答：

> 当前 Discovery 是横截面排序问题，不是连续仓位控制问题。RL 需要额外定义 state、action、reward、transaction cost 和 environment，增加很多无法必要证明的假设。因此我把 RL 留给真正的 Portfolio Allocation，而不是 Discovery。

---

## 8. Representative Research Pool

这是 Sector Discovery 与 7-Agent 之间的桥。

文件：

~~~text
tradingagents/discovery/representatives.py
~~~

### 8.1 它不是第二个“选股模型”

Representative Score：

~~~text
35% 申万指数权重
30% 20日平均成交额
20% 行业内 20/60 日相对强弱
15% 数据完整性
~~~

刻意 **不使用**：

- PE/PB；
- ROE；
- 净利润增长；
- CFO/NP；
- Quality Score。

原因：

> 这一层只决定“研究谁”，不决定“买谁”。

### 8.2 为什么看 Relative Strength

不是因为“涨得多就买”。

而是：

> 如果同一行业有很多成分股，优先选择既有代表性、流动性好、当前市场关注度较高、数据又完整的公司，可以让后续高成本 Agent 研究更有效。

### 8.3 research_context

每个代表股都会生成：

~~~text
research_context
~~~

内容包括：

- 来源行业；
- Sector Style；
- Representative Score；
- 入选原因；
- 明确声明“NOT investment evidence”。

随后传入：

~~~python
TradingAgentsGraph.propagate(
    ticker,
    date,
    candidate_context=research_context,
)
~~~

---

## 9. 如何避免 Representative Score 造成确认偏误

这是一个很好的面试深挖点。

如果把：

~~~text
“它是 Top Sector 代表股”
~~~

直接告诉 Agent，模型可能产生 confirmation bias。

所以项目没有把它直接当证据。

状态里单独保存：

~~~text
candidate_context
~~~

然后统一通过：

~~~text
get_candidate_context_from_state()
~~~

渲染为：

> selection prior; NOT evidence

同时明确要求：

- 不可把代表性评分写成 FACT；
- 不可用 Style 标签证明公司基本面；
- 必须重新调用 Tool；
- 最终结论允许推翻 Discovery prior。

Decision Auditor 还会额外检查：

> 是否把行业得分、Style 或代表股入选原因误写成投资事实。

---

## 10. 为什么 candidate_context 还进入 Checkpoint Signature

文件：

~~~text
tradingagents/graph/trading_graph.py
~~~

LangGraph 支持 checkpoint resume。

问题：

如果：

~~~text
同一 ticker + 同一 date
~~~

第一次来自“银行高股息研究池”，第二次来自另一个不同研究入口，但直接复用旧 checkpoint，就会造成上下文错配。

因此：

~~~text
candidate_context
 -> SHA256
 -> origin hash
 -> checkpoint signature
~~~

不同研究来源不会错误 resume 同一个 checkpoint。

这是一个很典型的 Agent State Engineering 细节。

---

# 11. 7-Agent 主链

当前 Agent 数量是 7：

1. Market Analyst
2. News & Sentiment Analyst
3. Fundamentals Analyst
4. Bull Researcher
5. Bear Researcher
6. Portfolio Manager
7. Decision Auditor

不是越多越好。

旧 TradingAgents 有更多角色、多轮辩论和 Risk Debate。

本项目选择：

> **减少角色数量，保留真正不同的信息职责。**

---

# 12. 三个 Analyst 为什么用私有 Subgraph

核心路径：

~~~text
tradingagents/graph/analyst_subgraph.py
tradingagents/graph/analyst_execution.py
tradingagents/graph/setup.py
~~~

每个 Analyst：

~~~text
LLM
 |
 | tool_calls
 v
ToolNode
 |
 v
LLM
 |
 v
Final Analyst Report
~~~

而且三个 Analyst 并行。

父图只拿到：

- final report；
- compact tool trace。

不会把完整 ToolMessage 链路全部传播给其他 Agent。

优点：

1. 防止并行 ToolMessage 混线；
2. 降低父图 state 体积；
3. 降低后续 Token 重复消费；
4. 每个 Analyst 可以有自己的 Tool Surface。

---

# 13. Market Analyst

文件：

~~~text
tradingagents/agents/analysts/market_analyst.py
~~~

主要工具：

- get_stock_data
- get_indicators
- get_verified_market_snapshot

核心 Prompt 约束：

1. 先取价格数据；
2. 从技术指标列表中选最多约 8 个互补指标；
3. 最终精确 OHLCV / 指标值必须由 verified snapshot 支持；
4. 不允许没有日期/价格证据就宣称“历史支撑位验证”“反弹 X%”；
5. Tool 数据冲突时明确报告冲突，不自行编造一个折中数字。

它的核心价值不是“技术分析很聪明”，而是：

> **精确数字必须走 deterministic verified snapshot。**

---

# 14. News & Sentiment Analyst

文件：

~~~text
tradingagents/agents/analysts/news_analyst.py
~~~

主要工具可能包括：

- get_news
- get_global_news
- get_macro_indicators
- get_insider_transactions
- search_company_knowledge（RAG enabled 时）

关键约束：

- 研究信息不能晚于 as_of_date；
- 管理层持股变化只是 event evidence，不自动等于 bullish/bearish；
- RAG 检索必须带 as_of_date；
- 工具缺数据时要显式写 uncertainty；
- 不允许凭空编 sentiment probability。

---

# 15. Fundamentals Analyst

文件：

~~~text
tradingagents/agents/analysts/fundamentals_analyst.py
~~~

工具：

- get_fundamentals
- get_balance_sheet
- get_cashflow
- get_income_statement

关注：

- 财务结构；
- 盈利能力；
- 现金流；
- 历史财务变化。

另外在 Prompt / Claim 规则里强调：

> 派生计算必须能追溯输入和公式。

例如：

~~~text
增长率 = (本期 - 上期) / 上期
~~~

不能只给一个没有来源的百分比。

---

# 16. Bull Researcher

文件：

~~~text
tradingagents/agents/researchers/bull_researcher.py
~~~

它：

- 不调用工具；
- 只读取三个 Analyst 的紧凑证据；
- 形成一次性看多假设。

目标：

- 驱动因素；
- 竞争优势；
- 催化；
- 验证条件；
- 看多逻辑最脆弱的部分。

不允许：

- 编目标价；
- 编对手观点；
- 把推断写成事实。

---

# 17. Bear Researcher

文件：

~~~text
tradingagents/agents/researchers/bear_researcher.py
~~~

同样不调用工具。

关注：

- 估值风险；
- 盈利质量风险；
- 行业风险；
- 政策风险；
- 流动性风险；
- 技术面风险；
- 事件风险。

特别强调：

> 已发生负面事实 != 未来可能发生的风险情景。

---

# 18. 为什么 Bull / Bear 不做多轮 Debate

旧设计里，多轮 Debate 很容易变成：

~~~text
A复述B
 -> B复述A
 -> 加更多自然语言
 -> Token 增加
 -> 新证据并没有增加
~~~

所以现在：

~~~text
Analyst Evidence
    |
 +--+--+
 |     |
Bull  Bear
 |     |
 +--+--+
    |
   PM
~~~

Bull/Bear 是：

> 并行的互补 Hypothesis Generator。

而不是聊天机器人之间互相争论。

---

# 19. Portfolio Manager

文件：

~~~text
tradingagents/agents/managers/portfolio_manager.py
~~~

输入：

- 三类 Analyst Evidence；
- Bull Thesis；
- Bear Thesis；
- 历史 Reflection；
- Auditor revision feedback。

输出结构：

~~~text
PortfolioDecision
~~~

主要字段：

- rating
- confidence
- executive_summary
- investment_thesis
- key_risks
- catalysts
- invalidation_conditions
- position_guidance
- time_horizon

五档评级：

- Buy
- Overweight
- Hold
- Underweight
- Sell

注意：

> 项目仍然不自动执行下单。

---

# 20. Decision Auditor

文件：

~~~text
tradingagents/agents/auditors/decision_auditor.py
~~~

它不调用工具，不增加新事实。

只检查：

1. 数字是否有上游证据；
2. 是否违反 PIT；
3. 是否把 inference 写成 fact；
4. Rating 是否和证据冲突；
5. 同一指标有没有前后冲突；
6. 是否把 Sector Style / Representative Score 当成公司投资证据。

输出：

~~~text
PASS
或
REVISE
~~~

如果 REVISE：

~~~text
Auditor
  |
  v
Portfolio Manager
~~~

只有有限次数修订。

---

# 21. Claim-aware Context Compression

核心路径：

~~~text
tradingagents/agents/utils/evidence_claims.py
tradingagents/agents/utils/context_compaction.py
~~~

四类 Claim：

### FACT

工具 / 数据源直接支持。

例如：

> 截止 2026-09-05 收盘价为 X。

### CALCULATION

基于已知输入计算。

例如：

> 20 日收益 = 当前价 / 20 日前价格 - 1。

### INFERENCE

模型解释。

例如：

> 量价配合可能意味着趋势延续概率提高。

它不能被写成 FACT。

### CONDITIONAL

条件性未来情景。

例如：

> 如果下一季度毛利率继续回升且收入增速维持，则盈利修复逻辑增强。

必须保留条件。

---

# 22. Compression 为什么不用额外 LLM

旧实现只做字符截断。

当前实现：

~~~text
full report
 -> parse explicit Evidence Claims
 -> legacy deterministic fallback classification
 -> type-aware budget selection
 -> compact context
~~~

不额外调用 LLM。

原因：

> Compression 的目标就是降低 Token / latency，如果为了压缩再增加一次 LLM 调用，会抵消一部分工程收益。

原始报告仍保留用于审计。

---

# 23. PIT 数据治理

这是项目最重要的可靠性卖点之一。

研究截止日：

~~~text
as_of_date
~~~

原则：

> T 日研究，只能看到 T 日当时已经公开的信息。

主要守卫：

1. 行情查询截止 T；
2. 财务数据区分 report period 和 publish/available date；
3. CNInfo 公告时间 <= T；
4. RAG publish_date <= T；
5. Retriever 后再次做 defensive date filter；
6. Auditor 检查未来日期。

---

# 24. 为什么历史 Representative Pool 有额外限制

纯 Sector Discovery 可以历史运行，因为它直接拿历史行业日报。

但申万：

~~~text
index_component_sw
~~~

主要是当前成分 + entry_date，不能完整恢复已经退出的历史成分。

所以：

~~~text
历史 Top Sector
 -> 用今天的成分股
~~~

会产生 survivor bias。

因此 strict PIT 下：

> Representative Pool / legacy-stock 只允许当前或最近 7 天。

如果显式允许：

~~~text
--allow-historical-membership
~~~

项目会把它标记为调试模式，而不是严格回测。

面试时这点非常加分：

> 我没有为了“能跑历史回测”而假装当前成分就是历史成分。

---

# 25. Finance MCP

路径：

~~~text
tradingagents/mcp/
tradingagents/agents/utils/tool_registry.py
~~~

结构：

~~~text
Agent
  |
StructuredTool
  |
MCP Client
  |
Streamable HTTP
  |
Finance MCP Server
  |
Data / RAG
~~~

特点：

- local fallback；
- external MCP allowlist；
- MCP 不可用时核心本地工具仍能运行。

MCP 的价值：

> **解耦 Tool 的部署位置，而不是增加“智能”。**

---

# 26. Hybrid RAG

路径：

~~~text
tradingagents/rag/
~~~

流程：

~~~text
Dense Retrieval
      +
     BM25
      |
      v
     RRF
      |
      v
Optional Reranker
      |
      v
PIT Defensive Filter
      |
      v
Top-K Evidence
~~~

为什么 Hybrid：

金融数据有两种特点：

- 自然语言语义；
- 股票代码 / 指标 / 公告关键词精确匹配。

只用 Dense 或只用 BM25 都不够稳定。

---

# 27. Router + SQLite Conversation

路径：

~~~text
tradingagents/conversation/
~~~

意图：

- research
- discovery
- tool_chat

线程保存：

- thread_id
- current ticker
- as_of_date
- history
- last audited research context
- representative_contexts

一个重要优化：

> 用户只是追问上一轮结论，不重新跑完整 7-Agent。

另外现在：

~~~text
“给我值得关注的行业和每个行业代表股”
~~~

会生成 Representative Pool，并把：

~~~text
ticker -> research_context
~~~

存到 thread metadata。

之后用户对其中 ticker 发起深度研究时，会自动把对应 selection provenance 传入 7-Agent。

---

# 28. Evaluation 与 Backtest

## Agent Evaluation

路径：

~~~text
tradingagents/evaluation/
~~~

关注：

- Tool Choice
- 参数日期
- PIT
- Trajectory
- Report Grounding

回答：

> Agent 是否按工程约束正确执行？

## Outcome Backtest

路径：

~~~text
tradingagents/backtest/
~~~

指标包括：

- direction accuracy
- raw return
- excess return
- max drawdown
- Sharpe
- performance by rating

回答：

> 研究结论后来在市场上表现如何？

两者必须分开。

---

# 29. 项目中所谓“Skill”具体是什么

当前仓库**没有单独设计一个名为 Skills 的 DSL / skills/ 目录**。

不要在面试时说：

> “我实现了完整 Skills Framework。”

目前更准确的表述是：

> Agent 能力由 **Prompt Contract + Tool Group + State Contract + LangGraph Routing** 组合形成。

例如 Market Analyst 的“Skill”可以理解为：

~~~text
Technical Research Skill
 = Market Prompt
 + get_stock_data
 + get_indicators
 + verified snapshot
 + PIT cutoff
 + Claim rules
~~~

News Skill：

~~~text
Event Research Skill
 = News Prompt
 + CNInfo/news
 + macro tools
 + optional RAG
 + PIT rules
~~~

这是当前代码真实支持的描述。

---

# 30. Tool Calling 如何执行

每个 Analyst 的工具由：

~~~text
tool_registry.py
~~~

统一构建。

然后：

~~~text
GraphSetup
 -> Analyst Factory(llm, tool_group)
 -> llm.bind_tools(active_tools)
 -> private ToolNode
~~~

之前代码曾存在一个 wiring 问题：

~~~text
GraphSetup 传 (llm, tool_group)
但 Factory 只接受 llm
~~~

现在三个 Factory 均支持：

~~~python
create_xxx_analyst(llm, tools=None)
~~~

并新增真实 Graph construction smoke test 防止再次漂移。

---

# 31. 并行执行

阶段一：

~~~text
Market --------\
News ----------- Fan-In
Fundamentals ---/
~~~

阶段二：

~~~text
Bull ---\
         Fan-In
Bear ---/
~~~

相比串行：

~~~text
Market -> News -> Fundamentals -> Bull -> Bear
~~~

主要优化的是 critical path latency。

注意：

> 目前仓库没有一个可公开复现的“稳定降低 32%”benchmark，因此面试时不要主动说具体 32%，除非你自己保留了对应 LangSmith/raw timing 日志。

---

# 32. 项目最大的工程亮点

建议按优先级讲：

### 第一梯队

1. PIT Data Governance
2. 7-Agent private subgraph + Fan-Out/Fan-In
3. Decision Auditor
4. Sector Discovery -> Representative Pool 分层
5. Claim-aware Context Compression

### 第二梯队

6. Finance MCP + fallback
7. PIT-aware Hybrid RAG
8. Conversation Router + SQLite
9. Agent Evaluation / Outcome Backtest

### 第三梯队

10. FastAPI / Docker Compose
11. GitHub Actions
12. Python 3.11 / 3.12 reproducibility

---

# 33. 面试官：为什么不用 LLM 直接选行业？

推荐答法：

> 行业排名主要是结构化横截面数值问题，LLM 在精确数值计算、排序和日期约束方面不可验证，所以我用 Python 或量化 Ranker。LLM 留给 Tool Choice、新闻语义理解、多视角综合和证据审计。这个边界能明显减少幻觉，也更方便回测。

---

# 34. 面试官：为什么先 Sector 再 Representative Stock？

推荐答法：

> 我不再把 Sector 当作股票硬门槛，而是把 Sector Discovery 独立成“研究预算分配问题”。Top-K 行业确定后，代表股层只负责选行业内最适合被研究的公司，不判断谁收益最高。这样既避免全市场跨行业财务结构比较，又能把高成本 7-Agent 的调用数量控制下来。

---

# 35. 面试官：Representative Score 会不会又变成隐式选股？

答：

> 我刻意把它和投资质量因子分离。Representative Score 只用行业指数权重、成交额、行业内相对强弱、数据完整性，不用 PE、PB、ROE、利润增速。并且传给 Agent 时明确标记为 selection provenance，不允许作为最终评级证据，Auditor 也检查这一点。

---

# 36. 面试官：LightGBM 怎么避免未来数据泄漏？

答：

> 目前仓库只实现 inference adapter，不内置一个声称有效的模型。真正训练时要求按时间切分，特征只使用 t 时点可见数据，label 使用未来 20 日相对收益，做 walk-forward 或 expanding-window 验证，不能随机 shuffle。模型未通过这些验证前，rule rank 是默认结果。

---

# 37. 面试官：为什么需要 Auditor？

答：

> Portfolio Manager 是生成者，让它自己验证自己容易出现 self-consistency bias，所以我拆出一个无工具、无新事实权限的 Auditor，只检查 Grounding、PIT、数字一致性和 rating consistency。只有实质问题才 REVISE，并限制修订轮数，避免无限自我反思。

---

# 38. 面试官：为什么 Auditor 不调用工具？

答：

> Auditor 的目标是验证“最终结论是否忠实于已有研究证据”，而不是重新做一次研究。如果 Auditor 再调用新工具，就会同时成为证据生产者和验证者，职责边界又混在一起。

---

# 39. 面试官：为什么 Claim 需要四类？

答：

> 金融研究里最危险的问题之一是模型把“推断”逐轮复述成“事实”。所以我显式区分 Fact、Calculation、Inference、Conditional。这样下游能知道哪些是直接数据，哪些是派生值，哪些只是解释，以及哪些预测依赖触发条件；Auditor 也能检查类型升级错误。

---

# 40. 面试官：你的 RAG 和普通 RAG 有什么区别？

答：

> 普通 RAG 只解决 relevance；历史投研还必须解决 availability。我的检索除了 ticker 和语义相关性，还要求 publish_date <= as_of_date，并在检索后再做一次防御性过滤，所以它是 PIT-aware RAG。

---

# 41. 面试官：如何保证多 Agent 并发时消息不串？

答：

> 三个 Analyst 不共享同一条 Tool message channel，而是各自运行在私有 Subgraph。父图只接收 final report 和 trace，因此 ToolMessage 不会在并行 Analyst 之间互相污染。

---

# 42. 面试官：为什么不继续增加 Agent？

答：

> Agent 数量本身不是目标。每增加一个角色都会增加 Token、延迟和状态复杂度。我只保留信息职责真正不同的角色，并把数值任务交给 deterministic code，把证据生成和审计职责拆开。

---

# 43. 面试官：项目当前最大的局限是什么？

推荐坦诚回答：

1. 申万历史成分数据不完整，因此历史 Representative Pool 有 survivor-bias 边界；
2. Sector Rule Weight 目前是工程 baseline，还需要系统 walk-forward 验证；
3. LightGBM 目前是 adapter，不是已经证明有效的 alpha model；
4. 第三方 AKShare / BaoStock / CNInfo 的稳定性受外部接口影响；
5. LLM 输出仍存在非确定性，Auditor 只能降低而不能消灭 hallucination；
6. Outcome Backtest 不能证明未来盈利能力。

---

# 44. 如果让你继续迭代，下一步是什么？

推荐答：

> 第一优先不是再加 Agent，而是量化工程收益：并行 vs 串行 P50/P95 latency、Token 消耗、Auditor REVISE rate、PIT violation rate、RAG Recall@K/MRR。量化模型侧会补 Sector Ranker 的时间滚动训练和 Rank IC / Top-K excess return；数据侧会寻找可恢复历史行业成分的数据源，解决 survivor bias。

---

# 45. 简历怎么描述最稳妥

推荐：

> 基于 LangGraph 重构 A 股投研链路，设计 Sector-first Discovery → Representative Research Pool → 7-Agent Deep Research 分层架构：以确定性 Market Regime 与 Momentum/Value/Dividend/Liquidity Style Rank 分配研究预算，可选 LightGBM 横截面 Ranker；代表股仅按行业权重、流动性、相对强弱及数据完整性选择研究入口，并通过 selection-prior guard 防止候选来源被 Agent 升级为投资证据。

第二条：

> 将单股研究裁剪为 Market / News / Fundamentals → Bull & Bear → Portfolio Manager → Decision Auditor 的 7-Agent 两阶段 Fan-Out/Fan-In；Analyst 使用私有 Subgraph 完成 LLM↔ToolNode 调用，支持 Finance MCP / Local fallback / PIT-aware Hybrid RAG。

第三条：

> 设计 Claim-aware Context Compression，将中间声明区分 FACT / CALCULATION / INFERENCE / CONDITIONAL，按字符预算向下游传递紧凑 Evidence Package，同时保留原始报告用于审计；Auditor 对 Grounding、PIT、数字冲突和 inference-as-fact 进行 PASS/REVISE 校验。

---

# 46. 5 分钟现场 Demo

### 1. 行业发现

~~~bash
python scripts/discover_a_share.py \
  --mode all \
  --date 2026-09-05 \
  --top 6
~~~

讲：

- Market Regime；
- Style Weight；
- Top Sector；
- 为什么不同 Style 能胜出。

### 2. 代表性研究池

~~~bash
python scripts/discover_a_share.py \
  --mode pool \
  --date 2026-09-05 \
  --top 4 \
  --representatives-per-sector 2
~~~

讲：

> 这里不是选最佳股票，而是生成 8 个 Research Entry。

### 3. 7-Agent

~~~bash
python -m cli.main analyze
~~~

或 FastAPI：

~~~bash
uvicorn service.app:app --host 0.0.0.0 --port 8000
~~~

展示：

- Analyst parallel；
- Tool Calling；
- Bull/Bear；
- PM；
- Auditor。

---

# 47. 面试前最后记住的 12 句话

1. **不是 Auto Trading，是 Research System。**
2. **能确定性计算的，不交给 LLM。**
3. **Sector Discovery 是研究预算分配，不是买入推荐。**
4. **Representative Stock 是 Research Entry，不是最佳股票。**
5. **Selection prior 不是 Evidence。**
6. **Market Regime 调权重，不做资格硬门控。**
7. **Analyst 私有 Subgraph 防止并行 ToolMessage 污染。**
8. **Bull/Bear 是并行互补假设，不做无效多轮复述。**
9. **Claim 分 Fact / Calculation / Inference / Conditional。**
10. **PIT 不只是 query date，还包括 publish/available date。**
11. **Auditor 不生产新事实，只做独立校验。**
12. **Agent Evaluation 和市场 Outcome 必须分开。**

---

# 48. 推荐代码阅读顺序

如果面试前只有 20 分钟：

1. `tradingagents/discovery/pipeline.py`
2. `tradingagents/discovery/sectors.py`
3. `tradingagents/discovery/representatives.py`
4. `tradingagents/graph/setup.py`
5. `tradingagents/graph/analyst_subgraph.py`
6. `tradingagents/agents/analysts/market_analyst.py`
7. `tradingagents/agents/analysts/news_analyst.py`
8. `tradingagents/agents/analysts/fundamentals_analyst.py`
9. `tradingagents/agents/researchers/bull_researcher.py`
10. `tradingagents/agents/researchers/bear_researcher.py`
11. `tradingagents/agents/managers/portfolio_manager.py`
12. `tradingagents/agents/auditors/decision_auditor.py`
13. `tradingagents/agents/utils/evidence_claims.py`
14. `tradingagents/agents/utils/context_compaction.py`
15. `tradingagents/agents/utils/tool_registry.py`
16. `tradingagents/rag/retriever.py`
17. `tradingagents/conversation/agent.py`
18. `tradingagents/evaluation/`

---

## 最终项目定位

一句话：

> **这是一个把 Quant/Deterministic Ranking、Tool-using Multi-Agent、PIT Data Governance、Evidence Compression 和 Independent Audit 组合在一起的 A 股研究工程系统。**

不要把它包装成：

> “我做了一个能自动赚钱的 AI 炒股系统。”

前者可信，也更符合 Agent / LLM 应用研发岗位。
