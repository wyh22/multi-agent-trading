"""Build PIT-safe Style IC history from a long-form sector style panel."""

from __future__ import annotations

import argparse
from pathlib import Path

from tradingagents.discovery.style_ic import write_style_ic_history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel",
        type=Path,
        required=True,
        help=(
            "CSV with date,sector_code,close,momentum_score,valuation_score,"
            "dividend_score,liquidity_score"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/data/style_ic_history.csv"),
    )
    parser.add_argument(
        "--forward-periods",
        type=int,
        default=20,
        help="Forward observations used for the return label.",
    )
    parser.add_argument(
        "--min-sector-count",
        type=int,
        default=8,
    )
    args = parser.parse_args()

    history = write_style_ic_history(
        args.panel,
        args.output,
        forward_periods=args.forward_periods,
        min_sector_count=args.min_sector_count,
    )
    print(
        f"wrote {len(history)} PIT-safe IC rows to {args.output}; "
        "adaptive weighting must filter by available_date"
    )


if __name__ == "__main__":
    main()
