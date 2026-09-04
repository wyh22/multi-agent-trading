"""Reusable report-tree writer for the current seven-agent research workflow."""

from datetime import datetime
from pathlib import Path


def write_report_tree(final_state: dict, ticker: str, save_path) -> Path:
    """Persist the seven-agent report tree and return complete_report.md."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []

    # 1. Three analyst reports.
    analysts_dir = save_path / "1_analysts"
    analyst_specs = [
        ("market_report", "market.md", "Market Analyst"),
        ("news_report", "news.md", "News & Sentiment Analyst"),
        ("fundamentals_report", "fundamentals.md", "Fundamentals Analyst"),
    ]
    analyst_parts = []
    for key, filename, label in analyst_specs:
        value = str(final_state.get(key, "") or "").strip()
        if not value:
            continue
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / filename).write_text(value, encoding="utf-8")
        analyst_parts.append((label, value))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # 2. Independent Bull/Bear hypotheses.
    research_dir = save_path / "2_research"
    research_parts = []
    for key, filename, label in [
        ("bull_thesis", "bull.md", "Bull Researcher"),
        ("bear_thesis", "bear.md", "Bear Researcher"),
    ]:
        value = str(final_state.get(key, "") or "").strip()
        if not value:
            continue
        research_dir.mkdir(exist_ok=True)
        (research_dir / filename).write_text(value, encoding="utf-8")
        research_parts.append((label, value))
    if research_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
        sections.append(f"## II. Bull / Bear Research\n\n{content}")

    # 3. Portfolio Manager decision.
    decision = str(final_state.get("final_trade_decision", "") or "").strip()
    if decision:
        portfolio_dir = save_path / "3_portfolio"
        portfolio_dir.mkdir(exist_ok=True)
        (portfolio_dir / "decision.md").write_text(decision, encoding="utf-8")
        sections.append(f"## III. Portfolio Manager Decision\n\n{decision}")

    # 4. Independent Decision Auditor report.
    audit = str(final_state.get("audit_report", "") or "").strip()
    if audit:
        audit_dir = save_path / "4_audit"
        audit_dir.mkdir(exist_ok=True)
        (audit_dir / "audit.md").write_text(audit, encoding="utf-8")
        sections.append(f"## IV. Decision Audit\n\n{audit}")

    header = (
        f"# Trading Analysis Report: {ticker}\n\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    complete = save_path / "complete_report.md"
    complete.write_text(header + "\n\n".join(sections), encoding="utf-8")
    return complete
