"""Optional ML reranking for sector discovery.

The default discovery path remains deterministic and dependency-light.  A trained
LightGBM model can be plugged in as a second-stage cross-sectional ranker without
replacing the rule-based score or the PIT data layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


SECTOR_MODEL_FEATURES: tuple[str, ...] = (
    "change_pct",
    "ret_20d",
    "ret_60d",
    "turnover",
    "amount_share",
    "pe",
    "pb",
    "dividend_yield",
    "momentum_score",
    "valuation_score",
    "dividend_score",
    "liquidity_score",
    "rule_score",
)


class SectorRanker(Protocol):
    """Small adapter contract so tests/custom models do not depend on LightGBM."""

    def predict(self, frame: pd.DataFrame) -> list[float] | np.ndarray:
        ...


def build_sector_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the stable numeric feature matrix used by optional ML rankers."""

    features = pd.DataFrame(index=frame.index)
    for name in SECTOR_MODEL_FEATURES:
        if name in frame.columns:
            values = pd.to_numeric(frame[name], errors="coerce")
        else:
            values = pd.Series(np.nan, index=frame.index, dtype=float)
        values = values.replace([np.inf, -np.inf], np.nan)
        median = values.median(skipna=True)
        fill_value = float(median) if pd.notna(median) else 0.0
        features[name] = values.fillna(fill_value).astype(float)
    return features


class LightGBMSectorRanker:
    """Load a pre-trained LightGBM Booster as an optional sector ranker.

    No model is bundled with the repository.  Training/validation must be done
    separately with time-aware splits; this adapter only handles inference.
    """

    def __init__(self, model_path: str | Path):
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "LightGBM 未安装；如需启用 Sector ML Ranker，请执行 "
                "`pip install -e '.[quant]'`。"
            ) from exc

        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Sector ML model 不存在: {path}")
        self.model_path = path
        self.booster = lgb.Booster(model_file=str(path))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        features = build_sector_feature_frame(frame)
        model_names = list(self.booster.feature_name() or [])
        if model_names and all(name in features.columns for name in model_names):
            features = features[model_names]

        expected = int(self.booster.num_feature())
        if expected != features.shape[1]:
            raise ValueError(
                "Sector ML model 特征数与当前 Feature Contract 不一致: "
                f"model={expected}, current={features.shape[1]}. "
                f"当前特征={list(features.columns)}"
            )
        return np.asarray(self.booster.predict(features), dtype=float)


def _cross_section_score(values: pd.Series) -> pd.Series:
    """Convert arbitrary predictions into an interpretable 0-100 percentile score."""

    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    out = pd.Series(50.0, index=numeric.index, dtype=float)
    if valid.empty:
        return out
    if valid.nunique() <= 1:
        out.loc[valid.index] = 50.0
        return out
    out.loc[valid.index] = valid.rank(method="average", pct=True) * 100.0
    return out.clip(0.0, 100.0)


def blend_sector_scores(
    sectors: pd.DataFrame,
    *,
    ranker: SectorRanker | None = None,
    ml_weight: float = 0.5,
) -> pd.DataFrame:
    """Blend deterministic rule rank with an optional ML cross-sectional rank.

    The ML prediction is converted to a same-day percentile before blending so
    arbitrary regression/ranking score scales do not leak into the rule score.
    """

    if sectors is None or sectors.empty:
        return pd.DataFrame() if sectors is None else sectors.copy()

    out = sectors.copy()
    if "rule_score" not in out.columns:
        if "sector_score" not in out.columns:
            raise ValueError("Sector ranking 缺少 rule_score / sector_score")
        out["rule_score"] = pd.to_numeric(out["sector_score"], errors="coerce")

    weight = float(ml_weight)
    if weight < 0.0 or weight > 1.0:
        raise ValueError("ml_weight 必须位于 [0, 1]")

    if ranker is None or weight == 0.0:
        out["ml_score"] = np.nan
        out["sector_score"] = pd.to_numeric(out["rule_score"], errors="coerce").fillna(0.0)
        out["rank_source"] = "rule"
        return out.sort_values("sector_score", ascending=False).reset_index(drop=True)

    prediction = np.asarray(ranker.predict(out), dtype=float)
    if prediction.shape[0] != len(out):
        raise ValueError(
            f"Sector ML Ranker 返回 {prediction.shape[0]} 个预测，但行业数为 {len(out)}"
        )

    out["ml_prediction"] = prediction
    out["ml_score"] = _cross_section_score(pd.Series(prediction, index=out.index))
    rule = pd.to_numeric(out["rule_score"], errors="coerce").fillna(50.0)
    out["sector_score"] = ((1.0 - weight) * rule + weight * out["ml_score"]).clip(0.0, 100.0)
    out["rank_source"] = "rule+ml"
    return out.sort_values("sector_score", ascending=False).reset_index(drop=True)
