"""Regression tests for sector-first candidate discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.discovery.models import MarketRegimeResult, SectorRankingResult
from tradingagents.discovery.pipeline import discovery_markdown, run_discovery
from tradingagents.discovery.sector_ranker import blend_sector_scores
from tradingagents.discovery.sectors import (
    rank_sector_snapshots,
    sector_style_weights,
)

ROOT = Path(__file__).resolve().parents[1]


def _sector_snapshots():
    current = pd.DataFrame(
        [
            {
                "sector_code": "TECH",
                "sector_name": "科技",
                "close": 150.0,
                "change_pct": 5.0,
                "turnover": 10.0,
                "amount_share": 20.0,
                "pe": 60.0,
                "pb": 8.0,
                "dividend_yield": 0.2,
            },
            {
                "sector_code": "BANK",
                "sector_name": "银行",
                "close": 110.0,
                "change_pct": 1.0,
                "turnover": 2.0,
                "amount_share": 10.0,
                "pe": 6.0,
                "pb": 0.6,
                "dividend_yield": 5.0,
            },
            {
                "sector_code": "COAL",
                "sector_name": "煤炭",
                "close": 120.0,
                "change_pct": 0.5,
                "turnover": 3.0,
                "amount_share": 8.0,
                "pe": 10.0,
                "pb": 1.0,
                "dividend_yield": 6.0,
            },
            {
                "sector_code": "UTILITY",
                "sector_name": "公用事业",
                "close": 105.0,
                "change_pct": 0.2,
                "turnover": 1.0,
                "amount_share": 5.0,
                "pe": 15.0,
                "pb": 1.5,
                "dividend_yield": 4.0,
            },
        ]
    )
    anchor20 = pd.DataFrame(
        [
            {"sector_code": "TECH", "close": 100.0},
            {"sector_code": "BANK", "close": 105.0},
            {"sector_code": "COAL", "close": 110.0},
            {"sector_code": "UTILITY", "close": 100.0},
        ]
    )
    anchor60 = pd.DataFrame(
        [
            {"sector_code": "TECH", "close": 80.0},
            {"sector_code": "BANK", "close": 100.0},
            {"sector_code": "COAL", "close": 115.0},
            {"sector_code": "UTILITY", "close": 100.0},
        ]
    )
    return current, anchor20, anchor60


def test_regime_changes_style_weights_not_sector_eligibility():
    risk_on = sector_style_weights("Risk-On")
    risk_off = sector_style_weights("Risk-Off")
    assert risk_on["momentum"] > risk_off["momentum"]
    assert risk_off["dividend"] > risk_on["dividend"]
    assert sum(risk_on.values()) == 1.0
    assert sum(risk_off.values()) == 1.0


def test_growth_and_dividend_sectors_can_win_for_different_reasons():
    current, anchor20, anchor60 = _sector_snapshots()
    risk_on = rank_sector_snapshots(
        current,
        anchor20,
        anchor60,
        market_regime="Risk-On",
    )
    risk_off = rank_sector_snapshots(
        current,
        anchor20,
        anchor60,
        market_regime="Risk-Off",
    )

    assert set(risk_on["sector_code"]) == {"TECH", "BANK", "COAL", "UTILITY"}
    assert set(risk_off["sector_code"]) == {"TECH", "BANK", "COAL", "UTILITY"}

    tech_on = risk_on.set_index("sector_code").loc["TECH"]
    coal_off = risk_off.set_index("sector_code").loc["COAL"]
    assert tech_on["primary_style"] in {"Momentum", "Liquidity"}
    assert coal_off["primary_style"] in {"Dividend", "Value"}
    assert risk_on.iloc[0]["sector_code"] == "TECH"
    assert risk_off.iloc[0]["sector_code"] != "TECH"


class _FakeRanker:
    def predict(self, frame: pd.DataFrame):
        # Make the last rule-ranked row the strongest ML candidate.
        return np.arange(len(frame), dtype=float)


def test_optional_ml_ranker_preserves_rule_score_and_reranks():
    sectors = pd.DataFrame(
        {
            "sector_code": ["A", "B", "C"],
            "rule_score": [90.0, 70.0, 50.0],
        }
    )
    out = blend_sector_scores(
        sectors,
        ranker=_FakeRanker(),
        ml_weight=1.0,
    )
    assert list(out["sector_code"]) == ["C", "B", "A"]
    assert "rule_score" in out.columns
    assert "ml_score" in out.columns
    assert set(out["rank_source"]) == {"rule+ml"}


def test_run_discovery_returns_sector_shortlist_without_stock_hard_gate():
    market = MarketRegimeResult(
        as_of_date="2026-09-05",
        regime="Neutral",
        score=50.0,
        indices=pd.DataFrame(),
        summary="test",
    )
    raw = pd.DataFrame(
        {
            "sector_code": ["A", "B", "C"],
            "sector_name": ["行业A", "行业B", "行业C"],
            "rule_score": [88.0, 77.0, 66.0],
            "sector_score": [88.0, 77.0, 66.0],
            "momentum_score": [90.0, 70.0, 50.0],
            "valuation_score": [50.0, 80.0, 60.0],
            "dividend_score": [20.0, 90.0, 60.0],
            "liquidity_score": [90.0, 60.0, 40.0],
        }
    )

    def market_analyzer(_date):
        return market

    def sector_analyzer(_date, *, market_regime, top_n):
        assert market_regime == "Neutral"
        assert top_n == 0
        return SectorRankingResult(
            as_of_date="2026-09-05",
            current_data_date="2026-09-05",
            anchor_20d_date="2026-08-06",
            anchor_60d_date="2026-06-07",
            sectors=raw,
        )

    result = run_discovery(
        "2026-09-05",
        top_n=2,
        market_analyzer=market_analyzer,
        sector_analyzer=sector_analyzer,
    )
    assert list(result.sectors.sectors["sector_code"]) == ["A", "B"]
    assert len(result.sector_universe) == 3
    assert result.metadata["sector_universe_size"] == 3
    assert result.metadata["rank_source"] == "rule"


def test_sector_report_explicitly_avoids_stock_buy_list():
    market = MarketRegimeResult(
        as_of_date="2026-09-05",
        regime="Neutral",
        score=50.0,
        indices=pd.DataFrame(),
        summary="test",
    )
    sectors = SectorRankingResult(
        as_of_date="2026-09-05",
        current_data_date="2026-09-05",
        anchor_20d_date="2026-08-06",
        anchor_60d_date="2026-06-07",
        sectors=pd.DataFrame(
            {
                "sector_code": ["A"],
                "sector_name": ["行业A"],
                "rule_score": [80.0],
                "sector_score": [80.0],
            }
        ),
    )

    def market_analyzer(_date):
        return market

    def sector_analyzer(_date, *, market_regime, top_n):
        return sectors

    result = run_discovery(
        "2026-09-05",
        top_n=1,
        market_analyzer=market_analyzer,
        sector_analyzer=sector_analyzer,
    )
    text = discovery_markdown(result)
    assert "行业研究优先级" in text
    assert "不是个股买入清单" in text
    assert "Sector Research Shortlist" in text


def test_application_surfaces_sector_discovery():
    service = (ROOT / "service" / "app.py").read_text(encoding="utf-8")
    conversation = (
        ROOT / "tradingagents" / "conversation" / "agent.py"
    ).read_text(encoding="utf-8")
    coordinator = (
        ROOT / "tradingagents" / "discovery" / "coordinator_agent.py"
    ).read_text(encoding="utf-8")

    assert '"sectors": result.sectors.sectors.to_dict' in service
    assert "result.sectors.sectors.head" in conversation
    assert "discover_sectors_tool" in coordinator
    assert "discover_candidates_tool" not in coordinator
