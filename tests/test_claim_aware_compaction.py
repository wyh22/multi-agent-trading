"""Regression tests for typed evidence claims and claim-aware compression."""

from pathlib import Path

from tradingagents.agents.utils.context_compaction import build_analyst_context
from tradingagents.agents.utils.evidence_claims import (
    ClaimType,
    classify_claim,
    compress_claims,
    extract_claims,
)

ROOT = Path(__file__).resolve().parents[1]


def test_explicit_claim_tags_are_preserved():
    text = """
## Evidence Claims
- [FACT] 截止研究日，收盘价为 100 元。
- [CALCULATION] 由收入 120 和成本 72 计算得毛利率 = 40%。
- [INFERENCE] 这表明盈利质量较上一期有所改善。
- [CONDITIONAL] 如果需求继续恢复，则利润弹性可能进一步释放。
"""
    claims = extract_claims(text, "基本面")
    assert [claim.claim_type for claim in claims] == [
        ClaimType.FACT,
        ClaimType.CALCULATION,
        ClaimType.INFERENCE,
        ClaimType.CONDITIONAL,
    ]
    assert all(claim.explicit for claim in claims)


def test_heuristic_fallback_keeps_claim_boundaries():
    assert classify_claim("截至 2026-08-20，公司披露营业收入 120 亿元。") is ClaimType.FACT
    assert (
        classify_claim("由收入 120 和成本 72 计算得毛利率 = 40%。")
        is ClaimType.CALCULATION
    )
    assert classify_claim("这一变化表明盈利能力可能改善。") is ClaimType.INFERENCE
    assert (
        classify_claim("如果需求恢复，则毛利率可能继续改善。")
        is ClaimType.CONDITIONAL
    )


def test_compression_prioritizes_grounding_claims_under_budget():
    text = """
- [FACT] 收盘价 100 元。
- [CALCULATION] 由 120/200 计算得收入占比 = 60%。
- [INFERENCE] 这是一条非常长的解释性判断，用于描述市场可能如何解读当前信息，但它本身不是直接观察事实，也不应该优先挤占事实证据预算。
- [CONDITIONAL] 如果行业需求恢复，则盈利弹性可能释放。
"""
    result = compress_claims(text, source_section="测试", max_chars=105)
    types = {claim.claim_type for claim in result.claims}
    assert ClaimType.FACT in types
    assert ClaimType.CALCULATION in types
    assert result.dropped_count > 0
    assert len(result.rendered) <= 105


def test_analyst_context_uses_tagged_claim_interface_instead_of_verbose_body():
    state = {
        "market_report": (
            "这是一段很长的市场分析正文，不应该在已有显式 Claim 时重复进入下游。\n"
            "## Evidence Claims\n"
            "- [FACT] 截止研究日收盘价为 100 元。\n"
            "- [INFERENCE] 价格结构表明趋势偏强。"
        ),
        "news_report": (
            "## Evidence Claims\n"
            "- [FACT] 公司在研究截止日前发布了公告。\n"
            "- [CONDITIONAL] 如果订单兑现，则收入预期可能上修。"
        ),
        "fundamentals_report": (
            "## Evidence Claims\n"
            "- [FACT] 已披露营业收入为 120 亿元。\n"
            "- [CALCULATION] 由收入 120 和成本 72 计算得毛利率 = 40%。"
        ),
    }

    context = build_analyst_context(state, per_report_chars=500)
    assert "[FACT]" in context
    assert "[CALCULATION]" in context
    assert "[INFERENCE]" in context
    assert "[CONDITIONAL]" in context
    assert "不应该在已有显式 Claim 时重复进入下游" not in context


def test_claim_aware_contract_is_wired_into_all_reasoning_stages():
    analyst_paths = [
        "tradingagents/agents/analysts/market_analyst.py",
        "tradingagents/agents/analysts/news_analyst.py",
        "tradingagents/agents/analysts/fundamentals_analyst.py",
    ]
    for path in analyst_paths:
        source = (ROOT / path).read_text(encoding="utf-8")
        assert "claim_boundary_instruction()" in source

    downstream_paths = [
        "tradingagents/agents/researchers/bull_researcher.py",
        "tradingagents/agents/researchers/bear_researcher.py",
        "tradingagents/agents/managers/portfolio_manager.py",
        "tradingagents/agents/auditors/decision_auditor.py",
    ]
    for path in downstream_paths:
        source = (ROOT / path).read_text(encoding="utf-8")
        assert "claim_usage_instruction()" in source
