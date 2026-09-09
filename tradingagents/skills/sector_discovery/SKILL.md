# sector_discovery

## Purpose

为 A 股研究生成 PIT-safe 的行业研究优先级，并可进一步构建少量代表性研究入口。该 Skill 负责“研究谁”，不输出自动交易指令，也不把行业排名当成个股投资证据。

## When to use

- 用户要求行业发现、行业研究优先级或 Sector shortlist。
- 用户希望从行业层面筛选后续深度研究入口。
- 用户需要 Representative Research Pool，而不是直接比较公司基本面优劣。

## Inputs

- `as_of_date`: 研究截止日期。
- `top_n`: 返回行业数量，可选。
- Representative Pool 模式下可附带每行业代表股数量等参数。

## Execution

1. 使用确定性 Python 计算 Market Regime。
2. 对申万一级行业计算 Momentum / Value / Dividend / Liquidity 等横截面特征。
3. 使用 Regime-aware Rule Rank；可选 LightGBM 仅作为二阶段排序器。
4. 如请求代表性研究池，再依据行业指数权重、流动性、行业内相对强弱和数据完整性选择研究入口。

## Output

- 行业研究优先级或 Representative Research Pool。
- 返回研究来源上下文，但明确标记为 selection prior，不作为投资事实。

## Constraints

- 不让 LLM 直接计算横截面排名。
- 遵守 `as_of_date` 和历史成分数据边界。
- 行业/代表股结果是研究路由，不是买入清单。
