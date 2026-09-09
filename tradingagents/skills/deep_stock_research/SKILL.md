# deep_stock_research

## Purpose

执行复杂单股综合研究。该 Skill 是高成本、受控的完整研究工作流，只在用户需要跨行情、新闻/公告和基本面证据的综合分析时启用。

## When to use

- 用户明确要求“深度分析”“完整研究”“全面分析”或类似综合投研任务。
- 问题同时要求多个专业维度，原子 Tool 或单一专业分析不足以完成。
- 简单价格、指标、单份公告查询不应使用本 Skill。

## Inputs

- `ticker`: 股票代码。
- `as_of_date`: 研究截止日期。
- 可选历史 research context / candidate context。

## Execution

1. Market / News / Fundamentals 三个研究分支并行取证。
2. 上游证据汇总后，由 Bull / Bear 节点生成互补研究假设。
3. Portfolio Manager 综合证据、风险与条件形成研究结论。
4. Auditor 检查事实、数字、PIT、证据一致性及推断越界。
5. 如审查要求 REVISE，按 repair target 进行有界定向补证，再重新综合。

## Output

- 结构化研究结论、关键风险、催化因素与失效条件。
- 保存为可版本化的 Research Version，供多轮追问和 rollback 使用。

## Constraints

- 研究辅助用途，不执行自动交易。
- 重要事实必须有上游 Tool / RAG 证据。
- 严格遵守 `as_of_date`。
- 修复与重审具有最大轮次限制，避免无界循环。
