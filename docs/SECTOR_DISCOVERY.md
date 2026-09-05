# Sector-first Discovery V2

## 1. 为什么从“自动选股”改成“行业研究优先级”

旧版 Discovery 流程是：

~~~text
Market Regime
  -> Top Sector hard gate
  -> sector components
  -> stock quant score
  -> PIT quality screen
  -> sector soft cap
  -> stock Top-N
~~~

这个结构适合作为第一版确定性 baseline，但存在三个明显问题：

1. **Sector-first hard gating**  
   行业没有进入前若干名时，该行业的个股会在股票因子计算前被永久删除。

2. **跨行业统一财务评分**  
   银行、保险、科技、制造、公用事业的财务结构不同，用同一套 ROE / CFO-to-NP / leverage 规则直接横向比较容易形成结构性偏置。

3. **重复 Sector Exposure**  
   行业先决定个股是否有资格进入股票池，随后 Sector Score 又参与股票 Final Score，行业因子被重复计算。

因此 V2 把 Discovery 的问题重新定义为：

> 在研究截止日，哪些申万一级行业更值得优先投入后续研究预算？

行业发现只负责 **Research Prioritization**，不直接输出“最值得买的股票”。

---

## 2. V2 架构

~~~text
A-share Market
      |
      v
PIT / as-of-date market data
      |
      v
Market Regime
      |
      v
SW Level-1 full cross-section
      |
      +---- Momentum Score
      +---- Value Score
      +---- Dividend Score
      +---- Liquidity Score
      |
      v
Regime-aware Rule Score
      |
      +---- optional LightGBM Ranker
      |
      v
Cross-sectional score blending
      |
      v
Top-K Sector Research Shortlist
      |
      v
Representative stock selection
      |
      v
7-Agent single-stock research
~~~

关键变化是：

- **所有行业都先参与横截面评分**；
- Market Regime 只改变 Style 权重，不决定行业是否有资格进入；
- Sector Ranking 是主输出；
- 个股不再被定义为全市场“最佳股票”，而是后续深度研究的代表性入口。

---

## 3. Rule-based Style Score

### Momentum

使用：

- 当日涨跌；
- 20 日相对收益；
- 60 日相对收益。

目的是让科技、成长、周期等趋势较强行业有独立胜出路径。

### Value

使用：

- PE；
- PB。

当前版本使用同日申万一级行业横截面百分位。它是一个 **Value Style**，不是全局真值，也不应被解释成“PE 越低行业一定越好”。

### Dividend

使用行业股息率横截面排名。

该维度为银行、煤炭、石油石化、公用事业等高股息行业提供独立解释，不再把收益来源全部压缩进 Value。

### Liquidity

使用：

- 换手率；
- 成交额占比。

它反映行业当前资金活跃度，而不是基本面质量。

---

## 4. Market Regime 如何使用

Regime 不做 Hard Gate，只调整权重。

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

这些权重是**工程 baseline**，不是经过大规模历史实证后宣称最优的参数。后续应该通过 Walk-forward Validation 决定是否调整。

---

## 5. Trend Penalty

V2 仍保留趋势惩罚，但从“资格过滤”改为 **Soft Penalty**。

例如行业 20 日和 60 日收益同时为负时会降低 Rule Score，但行业仍然留在完整横截面中。

这样避免：

~~~text
短期弱势
=> 行业直接被删除
=> 该行业后续没有任何研究机会
~~~

---

## 6. Optional LightGBM Sector Ranker

代码入口：

~~~text
tradingagents/discovery/sector_ranker.py
~~~

默认运行不需要 LightGBM。

安装：

~~~bash
pip install -e ".[quant]"
~~~

然后：

~~~bash
python scripts/discover_a_share.py \
  --mode all \
  --date 2026-09-05 \
  --top 6 \
  --ml-model ./models/sector_ranker.txt \
  --ml-weight 0.5
~~~

ML 模型不会覆盖 Rule Score，而是：

~~~text
LightGBM raw prediction
      |
      v
same-day cross-sectional percentile
      |
      v
ML Score (0-100)
      |
      + Rule Score
      |
      v
Final Sector Score
~~~

当 ml_weight=0.5 时：

~~~text
Final = 0.5 * Rule Score + 0.5 * ML Score
~~~

仓库不内置一个未经验证却宣称有效的预训练模型。

---

## 7. Stable ML Feature Contract

当前可选 Ranker 的稳定输入字段为：

~~~text
change_pct
ret_20d
ret_60d
turnover
amount_share
pe
pb
dividend_yield
momentum_score
valuation_score
dividend_score
liquidity_score
rule_score
~~~

缺失值在推理层使用横截面中位数填充；若整列缺失则使用 0。

推荐训练目标不是“预测未来精确收益”，而是：

> 同一天的行业横截面相对排序。

例如标签：

~~~text
20-day sector return - CSI300 20-day return
~~~

若使用 LightGBM，更推荐 Ranking / cross-sectional regression 后再转排序，而不是把模型输出解释成精确收益率。

---

## 8. 训练与验证要求

一个可信的模型至少应满足：

1. **按时间切分**，禁止随机 Shuffle Train/Test；
2. 标签只使用 t 之后数据，特征只使用 t 当时可见数据；
3. 做 Walk-forward / expanding-window 验证；
4. 报告 Rank IC / Top-K excess return / turnover 等，而不是只看训练误差；
5. 明确交易成本不是 Sector Discovery 的直接目标；
6. ML 不得绕过现有 as-of-date / PIT 数据约束。

在没有满足这些条件前，Rule Rank 应继续作为默认结果。

---

## 9. Legacy Stock Discovery

旧股票筛选没有删除，保留用于比较：

~~~bash
python scripts/discover_a_share.py \
  --mode legacy-stock \
  --date 2026-09-05 \
  --top 10
~~~

它仍执行：

~~~text
Top sectors
 -> stock factors
 -> PIT quality screen
 -> soft sector cap
~~~

代码保留的目的不是继续把它作为默认生产逻辑，而是：

- 便于 A/B 对比；
- 保留之前已经实现的数据和因子能力；
- 为后续“代表性股票选择”提供可复用模块。

---

## 10. 与 7-Agent 的边界

当前 7-Agent 是 **single-stock research graph**：

~~~text
Market / News / Fundamentals
      -> Bull / Bear
      -> Portfolio Manager
      -> Decision Auditor
~~~

因此 Sector Discovery 不直接把行业代码传进公司基本面 Agent。

推荐链路是：

~~~text
Top-K sectors
   -> choose representative stocks
   -> 7-Agent single-stock research
~~~

这保持了职责边界：

- Quant / deterministic layer：做横截面排序；
- LLM layer：做语义研究、工具选择、多视角综合；
- Auditor：做证据与 PIT 一致性校验。

---

## 11. 面试时如何解释

推荐回答：

> 第一版是 Sector-first 股票筛选，优点是简单、确定性强，但实际运行后我发现它会产生 hard gating 和双重行业暴露，而且银行、科技、高股息资产并不适合用一套个股财务模型直接横向排名。所以我把 Discovery 重新定义为行业研究优先级问题：全量申万一级行业先计算 Momentum、Value、Dividend、Liquidity 四类 Style Score，Market Regime 只动态调整 Style 权重；同时保留一个可选 LightGBM 横截面 Ranker，Rule Score 永远保留用于可解释性和 fallback。Top-K 行业之后再选择代表性股票进入 7-Agent 深度研究。


---

## 12. Representative Research Pool

Sector Discovery 主输出仍然是行业，不会把 Top-K 行业直接解释为股票推荐。为了接入单股 7-Agent，新增：

~~~text
tradingagents/discovery/representatives.py
~~~

评分：

~~~text
35% SW index weight
30% 20-day average trading amount
20% within-sector 20/60-day relative strength
15% data coverage
~~~

这里不使用 PE/PB、ROE、利润增速或旧 Quality Score，因此它只回答：

> 哪些公司更适合作为这个行业的深度研究入口？

而不是：

> 哪些公司最值得买？

严格 PIT 下，Representative Pool 依赖申万当前成分数据，因此历史日期超过最近交易窗口时会主动拒绝，避免用当前成分回填历史产生 survivor bias。

每条代表股记录附带 \`research_context\`。它进入 7-Agent 后被标记为 selection prior / NOT evidence，Analyst 仍必须重新调用市场、新闻和基本面工具验证，Auditor 也检查是否把候选来源升级成投资事实。
