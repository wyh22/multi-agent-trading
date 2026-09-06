"""Collect a historical sector style panel using the existing PIT sector analyzer.

For a cheap smoke test use weekly sampling. For a formal 20-trading-day IC
experiment use business-day sampling and --forward-periods 20 in the next step.

Example:
    python scripts/build_sector_style_panel.py \
      --start 2024-01-01 --end 2026-08-31 --freq W-FRI \
      --output evaluation/data/sector_style_panel.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from tradingagents.discovery.sectors import analyze_sectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--freq",
        default="W-FRI",
        help="pandas date_range frequency; use B for business-day sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/data/sector_style_panel.csv"),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Delay between snapshots to reduce upstream API pressure.",
    )
    parser.add_argument(
        "--max-dates",
        type=int,
        help="Optional cap for smoke tests.",
    )
    args = parser.parse_args()

    dates = list(pd.date_range(args.start, args.end, freq=args.freq))
    if args.max_dates is not None:
        dates = dates[: max(0, int(args.max_dates))]

    rows = []
    seen_data_dates = set()
    failures = []
    for index, value in enumerate(dates, start=1):
        requested = value.date().isoformat()
        print(f"[{index}/{len(dates)}] {requested}")
        try:
            result = analyze_sectors(
                requested,
                market_regime="Neutral",
                top_n=0,
            )
            data_date = result.current_data_date
            if data_date in seen_data_dates:
                continue
            seen_data_dates.add(data_date)
            frame = result.sectors.copy()
            frame["date"] = data_date
            frame["requested_date"] = requested
            keep = [
                "date",
                "requested_date",
                "sector_code",
                "sector_name",
                "close",
                "momentum_score",
                "valuation_score",
                "dividend_score",
                "liquidity_score",
            ]
            keep = [column for column in keep if column in frame.columns]
            rows.extend(frame[keep].to_dict(orient="records"))
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "requested_date": requested,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if args.sleep > 0:
            time.sleep(args.sleep)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    failure_path = args.output.with_suffix(".failures.jsonl")
    if failures:
        import json

        with failure_path.open("w", encoding="utf-8") as handle:
            for row in failures:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif failure_path.exists():
        failure_path.unlink()

    print(
        f"wrote {len(rows)} sector rows across {len(seen_data_dates)} "
        f"snapshots to {args.output}; failures={len(failures)}"
    )


if __name__ == "__main__":
    main()
