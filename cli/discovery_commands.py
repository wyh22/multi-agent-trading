from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

os.environ.setdefault("TQDM_DISABLE", "1")

from tradingagents.discovery.market import analyze_market_regime
from tradingagents.discovery.pipeline import run_discovery, write_discovery_report
from tradingagents.discovery.sectors import analyze_sectors

console = Console()
discovery_app = typer.Typer(
    help="A股大盘环境、申万行业轮动与自动研究候选发现",
    no_args_is_help=True,
)


def _date(value: str | None) -> str:
    return value or dt.date.today().isoformat()


def _fmt_pct(value) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "-"


@discovery_app.command("market")
def market_command(
    date: str | None = typer.Option(None, "--date", help="分析日期 YYYY-MM-DD，默认今天"),
):
    result = analyze_market_regime(_date(date))
    console.print(f"\n[bold]Market Regime:[/bold] {result.regime}  [bold]Score:[/bold] {result.score:.1f}/100")
    console.print(result.summary)
    table = Table(title=f"A股大盘环境 ({result.as_of_date})")
    for col in ["指数", "20D", "60D", "MA20偏离", "波动率", "回撤", "Score"]:
        table.add_column(col)
    for _, row in result.indices.iterrows():
        table.add_row(
            str(row["index_name"]),
            _fmt_pct(row["ret_20d"]),
            _fmt_pct(row["ret_60d"]),
            _fmt_pct(row["ma20_gap"]),
            _fmt_pct(row["vol_20d"]),
            _fmt_pct(row["max_drawdown_60d"]),
            f"{float(row['index_score']):.1f}",
        )
    console.print(table)
    for warning in result.warnings:
        console.print(f"[yellow]WARN: {warning}[/yellow]")


@discovery_app.command("sectors")
def sectors_command(
    date: str | None = typer.Option(None, "--date", help="分析日期 YYYY-MM-DD，默认今天"),
    top: int = typer.Option(10, "--top", min=1, max=31),
):
    d = _date(date)
    market = analyze_market_regime(d)
    result = analyze_sectors(d, market_regime=market.regime, top_n=top)
    table = Table(title=f"申万一级行业排名 ({result.current_data_date}) | {market.regime}")
    for col in ["Rank", "行业", "20D", "60D", "PE", "PB", "Momentum", "Score"]:
        table.add_column(col)
    for i, row in result.sectors.iterrows():
        table.add_row(
            str(i + 1),
            str(row["sector_name"]),
            _fmt_pct(row["ret_20d"]),
            _fmt_pct(row["ret_60d"]),
            f"{float(row['pe']):.2f}" if row.get("pe") == row.get("pe") else "-",
            f"{float(row['pb']):.2f}" if row.get("pb") == row.get("pb") else "-",
            f"{float(row['momentum_score']):.1f}",
            f"{float(row['sector_score']):.1f}",
        )
    console.print(table)


@discovery_app.command("all")
def all_command(
    date: str | None = typer.Option(None, "--date"),
    sectors: int = typer.Option(4, "--sectors", min=1, max=10),
    per_sector: int = typer.Option(35, "--per-sector", min=10, max=100),
    top: int = typer.Option(10, "--top", min=1, max=50),
    max_sector_picks: int = typer.Option(0, "--max-sector-picks", min=0, max=50),
    quality_pool: int = typer.Option(0, "--quality-pool", min=0, max=200),
    quality_weight: float = typer.Option(0.25, "--quality-weight", min=0.0, max=0.5),
    skip_quality: bool = typer.Option(False, "--skip-quality"),
    allow_historical_membership: bool = typer.Option(False, "--allow-historical-membership"),
    output: Path | None = typer.Option(None, "--output"),
):
    d = _date(date)
    result = run_discovery(
        d,
        sector_count=sectors,
        per_sector=per_sector,
        top_n=top,
        max_shortlist_per_sector=(max_sector_picks or None),
        quality_pool_size=(quality_pool or None),
        quality_weight=quality_weight,
        quality_enabled=not skip_quality,
        strict_pit=not allow_historical_membership,
    )

    console.print(f"[green]✓ 大盘:[/green] {result.market.regime} ({result.market.score:.1f}/100)")
    table = Table(title="Research Shortlist")
    for col in ["Rank", "代码", "名称", "行业", "Final", "Flag"]:
        table.add_column(col)
    for i, row in result.stocks.candidates.iterrows():
        table.add_row(
            str(i + 1),
            str(row["ticker"]),
            str(row.get("name", "")),
            str(row.get("sector_name", "")),
            f"{float(row['final_score']):.1f}",
            str(row.get("valuation_quality_flag", "")),
        )
    console.print(table)

    if output is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path.cwd() / "reports" / f"discovery_{d.replace('-', '')}_{stamp}"
    report = write_discovery_report(result, output)
    console.print(f"\n[green]✓ 报告已保存:[/green] {report.resolve()}")
