from __future__ import annotations

from collections import Counter


CORE_COMPANY_DOCUMENT_GROUPS = {
    "annual_reporting": {
        "annual_report",
    },
    "interim_reporting": {
        "semiannual_report",
        "quarterly_report",
    },
    "company_events": {
        "announcement",
        "investor_relation",
        "earnings_release",
        "corporate_action",
    },
}


def build_corpus_coverage(
    documents: list[dict],
    *,
    ticker: str | None = None,
) -> dict:
    """Summarize corpus coverage without pretending that coverage equals quality."""

    rows = list(documents)
    company_rows = [
        row
        for row in rows
        if row.get("scope_type") == "company"
        and (ticker is None or row.get("ticker") == ticker)
    ]
    shared_rows = [
        row
        for row in rows
        if row.get("scope_type") != "company"
    ]

    doc_types = Counter(
        str(row.get("doc_type") or "document")
        for row in company_rows
    )
    scope_types = Counter(
        str(row.get("scope_type") or "legacy")
        for row in rows
    )
    verified = sum(
        1
        for row in company_rows
        if row.get("publish_date_verified") is True
    )
    industries = sorted(
        {
            str(row.get("industry") or "").strip()
            for row in company_rows
            if str(row.get("industry") or "").strip()
        }
    )

    missing_groups = []
    for group, accepted_types in CORE_COMPANY_DOCUMENT_GROUPS.items():
        if not any(doc_types.get(doc_type, 0) for doc_type in accepted_types):
            missing_groups.append(group)

    latest_company_date = max(
        (
            str(row.get("publish_date") or "")
            for row in company_rows
        ),
        default="",
    )

    return {
        "ticker": ticker,
        "company_documents": len(company_rows),
        "shared_documents_visible_in_inventory": len(shared_rows),
        "verified_company_documents": verified,
        "unverified_or_legacy_company_documents": (
            len(company_rows) - verified
        ),
        "industries": industries,
        "company_doc_type_counts": dict(sorted(doc_types.items())),
        "scope_type_counts": dict(sorted(scope_types.items())),
        "latest_company_publish_date": latest_company_date,
        "missing_core_document_groups": missing_groups,
        "coverage_note": (
            "This is corpus coverage only. It does not imply evidence quality, "
            "grounding quality, or investment conclusion completeness."
        ),
    }
