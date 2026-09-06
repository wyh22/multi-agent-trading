"""Build a compact, interview-friendly RAG corpus from official disclosures."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.rag.bootstrap import (
    default_reporting_years,
    fetch_cninfo_disclosures,
    select_high_value_disclosures,
)


DEFAULT_UNIVERSE = (
    PROJECT_ROOT / "evaluation" / "data" / "autumn_rag_universe_v1.json"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "evaluation" / "data" / "autumn_rag_manifest.generated.jsonl"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "results" / "rag_bootstrap" / "autumn_rag_bootstrap_report.json"
)


def _load_universe(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stocks = payload.get("stocks", [])
    if not isinstance(stocks, list) or not stocks:
        raise ValueError("universe contains no stocks")
    return payload


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    annual_default, interim_default = default_reporting_years()

    parser = argparse.ArgumentParser(
        description=(
            "Discover a compact cross-industry official-filing corpus and "
            "optionally ingest it into Qdrant."
        )
    )
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--annual-year", type=int, default=annual_default)
    parser.add_argument("--interim-year", type=int, default=interim_default)
    parser.add_argument("--max-docs-per-company", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument(
        "--stop-after-consecutive-failures",
        type=int,
        default=3,
        help=(
            "Stop discovery early when the official disclosure endpoint is "
            "clearly unavailable, instead of wasting time on the whole universe."
        ),
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="After manifest generation, batch-ingest the selected documents into Qdrant.",
    )
    args = parser.parse_args()

    universe = _load_universe(args.universe)
    rows: list[dict] = []
    company_reports: list[dict] = []
    failures: list[dict] = []
    consecutive_failures = 0

    start_date = f"{args.annual_year}-01-01"
    end_date = date.today().isoformat()

    for index, stock in enumerate(universe["stocks"], start=1):
        ticker = str(stock["ticker"])
        name = str(stock.get("name") or ticker)
        industry = str(stock.get("industry") or "")
        print(
            f"[{index}/{len(universe['stocks'])}] discover {ticker} {name}...",
            flush=True,
        )
        try:
            disclosures = fetch_cninfo_disclosures(
                ticker,
                start_date=start_date,
                end_date=end_date,
                attempts=args.attempts,
            )
            selected = select_high_value_disclosures(
                disclosures,
                ticker=ticker,
                annual_year=args.annual_year,
                interim_year=args.interim_year,
                max_docs=args.max_docs_per_company,
            )
            consecutive_failures = 0
        except Exception as exc:  # noqa: BLE001
            consecutive_failures += 1
            failures.append(
                {
                    "ticker": ticker,
                    "name": name,
                    "stage": "discovery",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  unavailable: {exc}", flush=True)
            if (
                consecutive_failures
                >= max(1, int(args.stop_after_consecutive_failures))
            ):
                print(
                    "Official disclosure discovery appears unavailable; "
                    "stopping early to avoid repeated failed requests.",
                    flush=True,
                )
                break
            continue

        selected_types = {item.doc_type for item in selected}
        expected_types = {
            "annual_report",
            "semiannual_report",
        }
        fallback_group_present = bool(
            {"investor_relation", "quarterly_report"} & selected_types
        )
        missing = sorted(expected_types - selected_types)
        if not fallback_group_present:
            missing.append("investor_relation_or_quarterly_report")

        company_reports.append(
            {
                "ticker": ticker,
                "name": name,
                "industry": industry,
                "selected_documents": len(selected),
                "selected_doc_types": sorted(selected_types),
                "missing": missing,
            }
        )

        for item in selected:
            rows.append(
                {
                    "scope_type": "company",
                    "ticker": item.ticker,
                    "industry": industry,
                    "title": item.title,
                    "url": item.url,
                    "publish_date": item.publish_date,
                    "doc_type": item.doc_type,
                    "publish_date_source": "CNINFO",
                    "publish_date_confidence": 1.0,
                    "publish_date_verified": True,
                    "source_authority": "CNINFO",
                    "source_url": item.url,
                    "metadata": {
                        "corpus_profile": universe.get("name", ""),
                        "company_name": name,
                        "selection_policy": (
                            "annual + semiannual + latest investor relation; "
                            "fallback q1"
                        ),
                    },
                }
            )
            print(
                f"  + {item.doc_type}: {item.publish_date} {item.title}",
                flush=True,
            )

    _write_jsonl(args.manifest, rows)

    report = {
        "profile": universe.get("name"),
        "objective": universe.get("objective"),
        "annual_year": args.annual_year,
        "interim_year": args.interim_year,
        "companies_requested": len(universe["stocks"]),
        "companies_processed": len(company_reports),
        "documents_selected": len(rows),
        "target_documents": (
            len(universe["stocks"]) * args.max_docs_per_company
        ),
        "companies": company_reports,
        "failures": failures,
        "manifest": str(args.manifest),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"manifest={args.manifest} documents={len(rows)} "
        f"companies={len(company_reports)} failures={len(failures)}",
        flush=True,
    )

    if not rows:
        raise SystemExit(
            "No documents were discovered. Keep the report and retry later "
            "instead of running the full Agent workflow."
        )

    if args.ingest:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "rag_ingest.py"),
            "--manifest",
            str(args.manifest),
        ]
        print("ingesting selected corpus into Qdrant...", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
