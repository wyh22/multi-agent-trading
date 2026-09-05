# 最终架构说明

## 一、为什么从原始多角色链路精简

原始 TradingAgents 中，Research Manager、Trader 和三类 Risk Debator 会连续读取相近的分析报告，并对同一批事实做多轮自然语言复述。实际运行中这会带来三个问题：

1. Token 和延迟随中间文本快速累积；
2. 最终报告大量重复，不利于用户阅读；
3. 多次自然语言转述增加无依据推断被后续角色继承的概率。

最终版采用“证据分析 -> 双视角批判 -> 一次决策 -> 独立审计”的短链路。

## 二、单股 7-Agent 深度研究

### 1. 市场分析师

负责行情和技术面，只使用行情、技术指标与确定性行情核验工具。

### 2. 新闻与情绪分析师

合并旧版 News 与 Sentiment。当前免费数据链没有独立、稳定的社交情绪源，因此不再为同一批新闻数据单独启动一个情绪角色。

### 3. 基本面分析师

负责 PIT 财务报表、估值和现金流分析，严格区分报告期与披露日。

### 4. 看多研究员

基于三类分析师证据构建最多 5 个看多论点，不调用新工具，不多轮辩论。

### 5. 看空研究员

独立构建最多 5 个风险与证据缺口，与看多研究员并行运行。

### 6. 投资组合经理

只做一次证据收敛，输出五档研究评级、置信度、核心逻辑、风险、催化和失效条件。

### 7. 决策审计智能体

独立检查事实依据、PIT、数字一致性和无依据推断。首次失败允许投资组合经理修订一次，然后再次审计。

## 三、单股研究之前：Sector Discovery 与 Representative Pool

默认 Discovery 已经从“自动选 Top10 股票”重构为“行业研究优先级”：

```text
Market Regime
  -> 申万一级行业全量横截面
  -> Momentum / Value / Dividend / Liquidity
  -> Regime-aware Rule Rank
  -> Optional LightGBM
  -> Top-K Sector Research Shortlist
  -> Representative Research Pool
  -> 7-Agent
```

Representative Pool 不是第二套投资评分。它只用行业指数权重、流动性、行业内相对强弱和数据完整性选择少量 Research Entry；不使用 PE/PB、ROE、利润增长或旧 Quality Score。

每个 Research Entry 会携带 `candidate_context`，但该字段明确标记为 selection prior / NOT evidence，并作为不可信数据边界处理。三类 Analyst 的私有 Subgraph 会接收该字段，但仍必须用工具独立验证；Decision Auditor 还会检查是否把行业 Style 或 Representative Score 错写成投资事实。

## 四、可选协调 Agent

Discovery Coordinator 只负责规划只读工具调用和研究预算，不直接计算因子，也不能凭训练知识推荐股票。当前工具包括：

- 大盘环境分析工具；
- 申万一级行业排名工具；
- Sector-first Discovery 工具；
- Representative Research Pool 工具。

确定性的 Python / 可选量化 Ranker 负责“算对”，Coordinator 负责“决定调用什么、研究多少”。

## 五、报告体量控制

最终报告只拼接：

- 市场与技术面
- 新闻、公告、宏观与情绪
- 基本面
- 看多 / 看空研究假设
- 最终研究结论
- 决策审计

所有独立节点原文和工具轨迹都写入 `trace/`，不再重复塞入用户报告。

## 六、保留英文的范围

源码注释、文档和主要提示词均使用中文。以下内容为了兼容协议、库或模型接口保留原始英文标识：

- Python / LangChain / LangGraph 类名和函数名
- 股票指标缩写，如 PE、PB、ROE、RSI、MACD
- 研究评级枚举：Buy / Overweight / Hold / Underweight / Sell
- 环境变量、工具函数名、API 字段名
