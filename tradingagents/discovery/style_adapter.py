from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


STYLES = ("momentum", "valuation", "dividend", "liquidity")


@dataclass(frozen=True)
class AdaptiveStyleWeights:
    weights: dict[str, float]
    used: bool
    observations: int
    signals: dict[str, float]
    warning: str = ""


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    clean = {key: max(0.0, float(weights.get(key, 0.0))) for key in STYLES}
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("style weights sum must be positive")
    return {key: value / total for key, value in clean.items()}


def load_adaptive_style_weights(
    as_of_date: str,
    *,
    base_weights: dict[str, float],
    history_path: str | Path,
    strength: float = 0.5,
    min_observations: int = 12,
    trailing_observations: int = 60,
    halflife: float = 12.0,
) -> AdaptiveStyleWeights:
    """Blend regime priors with trailing, available-at-the-time style IC.

    Expected CSV columns:
      date,available_date,momentum_ic,valuation_ic,dividend_ic,liquidity_ic

    date is the signal cross-section date. available_date is the date when the
    forward-return label used to compute that IC had fully matured.
    Historical research only consumes rows with available_date < as_of_date.

    Missing available_date fails closed to the rule prior because filtering on
    signal date alone would leak future returns into historical weights.
    """

    base = _normalize(base_weights)
    path = Path(history_path).expanduser()
    if not path.exists():
        return AdaptiveStyleWeights(
            weights=base,
            used=False,
            observations=0,
            signals={},
            warning=f"adaptive style history not found: {path}",
        )

    frame = pd.read_csv(path)
    required_meta = {"date", "available_date"}
    missing_meta = sorted(required_meta - set(frame.columns))
    if missing_meta:
        return AdaptiveStyleWeights(
            weights=base,
            used=False,
            observations=0,
            signals={},
            warning=(
                "adaptive style history missing temporal provenance columns: "
                f"{missing_meta}; fallback to rule weights"
            ),
        )

    required = [f"{style}_ic" for style in STYLES]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return AdaptiveStyleWeights(
            weights=base,
            used=False,
            observations=0,
            signals={},
            warning=f"adaptive style history missing columns: {missing}",
        )

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["available_date"] = pd.to_datetime(
        frame["available_date"],
        errors="coerce",
    )
    cutoff = pd.Timestamp(date.fromisoformat(as_of_date[:10]))
    frame = frame[
        frame["date"].notna()
        & frame["available_date"].notna()
        & (frame["available_date"] < cutoff)
    ]
    frame = frame.sort_values(
        ["available_date", "date"]
    ).tail(max(1, int(trailing_observations)))

    valid_rows = frame[required].apply(pd.to_numeric, errors="coerce")
    observations = int(valid_rows.dropna(how="all").shape[0])
    if observations < int(min_observations):
        return AdaptiveStyleWeights(
            weights=base,
            used=False,
            observations=observations,
            signals={},
            warning=(
                f"adaptive style history insufficient: "
                f"{observations} < {min_observations}"
            ),
        )

    signals: dict[str, float] = {}
    positive: dict[str, float] = {}
    for style in STYLES:
        series = pd.to_numeric(
            valid_rows[f"{style}_ic"],
            errors="coerce",
        ).dropna()
        if series.empty:
            signal = 0.0
        else:
            signal = float(
                series.ewm(
                    halflife=max(1.0, float(halflife)),
                    adjust=False,
                ).mean().iloc[-1]
            )
        signals[style] = signal
        positive[style] = max(0.0, signal)

    if sum(positive.values()) <= 0:
        return AdaptiveStyleWeights(
            weights=base,
            used=False,
            observations=observations,
            signals=signals,
            warning=(
                "all trailing style signals are non-positive; "
                "fallback to rule weights"
            ),
        )

    learned = _normalize(positive)
    alpha = max(0.0, min(1.0, float(strength)))
    blended = _normalize(
        {
            style: (1.0 - alpha) * base[style] + alpha * learned[style]
            for style in STYLES
        }
    )
    return AdaptiveStyleWeights(
        weights=blended,
        used=True,
        observations=observations,
        signals=signals,
    )
