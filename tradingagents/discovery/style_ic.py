from __future__ import annotations

from pathlib import Path

import pandas as pd


STYLE_SCORE_COLUMNS = {
    "momentum": "momentum_score",
    "valuation": "valuation_score",
    "dividend": "dividend_score",
    "liquidity": "liquidity_score",
}


def build_style_ic_history(
    panel: pd.DataFrame,
    *,
    forward_periods: int = 20,
    min_sector_count: int = 8,
) -> pd.DataFrame:
    """Build cross-sectional Style IC history with label-availability provenance.

    Required long-panel columns:
      date, sector_code, close,
      momentum_score, valuation_score, dividend_score, liquidity_score

    For each sector, the forward label is close[t + forward_periods] / close[t] - 1.
    The resulting IC row stores available_date, the latest date needed to know that
    forward return. Adaptive weighting must filter on available_date, not signal date.
    """

    if forward_periods <= 0:
        raise ValueError("forward_periods must be > 0")
    if min_sector_count < 3:
        raise ValueError("min_sector_count must be >= 3")

    required = {
        "date",
        "sector_code",
        "close",
        *STYLE_SCORE_COLUMNS.values(),
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"style panel missing columns: {missing}")

    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    for column in STYLE_SCORE_COLUMNS.values():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["date", "sector_code", "close"])
    frame = frame.sort_values(["sector_code", "date"]).reset_index(drop=True)
    groups = frame.groupby("sector_code", sort=False)
    frame["future_close"] = groups["close"].shift(-int(forward_periods))
    frame["available_date"] = groups["date"].shift(-int(forward_periods))
    frame["forward_return"] = frame["future_close"] / frame["close"] - 1.0

    rows: list[dict] = []
    for signal_date, cross_section in frame.groupby("date", sort=True):
        valid_label = cross_section.dropna(
            subset=["forward_return", "available_date"]
        )
        if len(valid_label) < int(min_sector_count):
            continue

        row = {
            "date": pd.Timestamp(signal_date).date().isoformat(),
            "available_date": (
                pd.to_datetime(valid_label["available_date"]).max().date().isoformat()
            ),
            "sector_count": int(len(valid_label)),
            "forward_periods": int(forward_periods),
        }
        valid_style_count = 0
        for style, column in STYLE_SCORE_COLUMNS.items():
            sample = valid_label[[column, "forward_return"]].dropna()
            if len(sample) < int(min_sector_count):
                row[f"{style}_ic"] = float("nan")
                continue
            # Spearman = Pearson correlation of ranks. Compute it directly
            # so the core project does not need scipy only for this statistic.
            ranked_style = sample[column].rank(method="average")
            ranked_return = sample["forward_return"].rank(method="average")
            row[f"{style}_ic"] = float(
                ranked_style.corr(ranked_return, method="pearson")
            )
            valid_style_count += 1

        if valid_style_count:
            rows.append(row)

    columns = [
        "date",
        "available_date",
        "sector_count",
        "forward_periods",
        "momentum_ic",
        "valuation_ic",
        "dividend_ic",
        "liquidity_ic",
    ]
    return pd.DataFrame(rows, columns=columns)


def write_style_ic_history(
    panel_path: str | Path,
    output_path: str | Path,
    *,
    forward_periods: int = 20,
    min_sector_count: int = 8,
) -> pd.DataFrame:
    panel = pd.read_csv(Path(panel_path))
    history = build_style_ic_history(
        panel,
        forward_periods=forward_periods,
        min_sector_count=min_sector_count,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(target, index=False)
    return history
