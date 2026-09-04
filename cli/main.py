import datetime
import os
import sys
import time
from collections import deque
from functools import wraps
from pathlib import Path

import typer
from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from cli.announcements import display_announcements, fetch_announcements
from cli.stats_handler import StatsCallbackHandler
from cli.utils import (
    ask_anthropic_effort,
    ask_gemini_thinking_config,
    ask_glm_region,
    ask_minimax_region,
    ask_openai_reasoning_effort,
    ask_output_language,
    ask_qwen_region,
    confirm_ollama_endpoint,
    detect_asset_type,
    ensure_api_key,
    get_ticker,
    prompt_openai_compatible_url,
    resolve_backend_url,
    select_analysts,
    select_deep_thinking_agent,
    select_llm_provider,
    select_shallow_thinking_agent,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
    sync_analyst_tracker_from_chunk,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.reporting import write_report_tree

console = Console()

# prompt_toolkit's win32 output module is importable only on Windows (it asserts
# the platform at import time), so gate on the platform rather than catching the
# failure — that way a genuinely broken prompt_toolkit on Windows still surfaces
# instead of silently disabling the handler below. Off Windows this stays an
# empty tuple, which `except` accepts and never matches (#1138).
if sys.platform == "win32":  # pragma: no cover - platform dependent
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError

    _NO_CONSOLE_ERRORS: tuple[type[BaseException], ...] = (NoConsoleScreenBufferError,)
else:
    _NO_CONSOLE_ERRORS = ()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    # Fixed agents that always run after the selected analyst fan-out.
    FIXED_AGENTS = {
        "Research Team": ["Bull Researcher", "Bear Researcher"],
        "Decision Team": ["Portfolio Manager", "Decision Auditor"],
    }

    ANALYST_MAPPING = {
        "market": "Market Analyst",
        "social": "News & Sentiment Analyst",  # saved-config compatibility
        "news": "News & Sentiment Analyst",
        "fundamentals": "Fundamentals Analyst",
    }

    REPORT_SECTIONS = {
        "market_report": ("market", "Market Analyst"),
        "news_report": ("news", "News & Sentiment Analyst"),
        "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
        "bull_thesis": (None, "Bull Researcher"),
        "bear_thesis": (None, "Bear Researcher"),
        "final_trade_decision": (None, "Portfolio Manager"),
        "audit_report": (None, "Decision Auditor"),
    }
    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # Store the complete final report
        self.agent_status = {}
        self.current_agent = None
        self.report_sections = {}
        self.selected_analysts = []
        self._processed_message_ids = set()

    def init_for_analysis(self, selected_analysts):
        """Initialize agent status and report sections based on selected analysts.

        Args:
            selected_analysts: List of analyst type strings (e.g., ["market", "news"])
        """
        normalized = []
        for raw in selected_analysts:
            key = getattr(raw, "value", raw)
            key = str(key).lower()
            key = "news" if key == "social" else key
            if key not in normalized:
                normalized.append(key)
        self.selected_analysts = normalized

        # Build agent_status dynamically
        self.agent_status = {}

        # Add selected analysts
        for analyst_key in self.selected_analysts:
            if analyst_key in self.ANALYST_MAPPING:
                self.agent_status[self.ANALYST_MAPPING[analyst_key]] = "pending"

        # Add fixed teams
        for team_agents in self.FIXED_AGENTS.values():
            for agent in team_agents:
                self.agent_status[agent] = "pending"

        # Build report_sections dynamically
        self.report_sections = {}
        for section, (analyst_key, _) in self.REPORT_SECTIONS.items():
            if analyst_key is None or analyst_key in self.selected_analysts:
                self.report_sections[section] = None

        # Reset other state
        self.current_report = None
        self.final_report = None
        self.current_agent = None
        self.messages.clear()
        self.tool_calls.clear()
        self._processed_message_ids.clear()

    def get_completed_reports_count(self):
        """Count reports that are finalized (their finalizing agent is completed).

        A report is considered complete when:
        1. The report section has content (not None), AND
        2. The agent responsible for finalizing that report has status "completed"

        This prevents interim updates (like debate rounds) from counting as completed.
        """
        count = 0
        for section in self.report_sections:
            if section not in self.REPORT_SECTIONS:
                continue
            _, finalizing_agent = self.REPORT_SECTIONS[section]
            # Report is complete if it has content AND its finalizing agent is done
            has_content = self.report_sections.get(section) is not None
            agent_done = self.agent_status.get(finalizing_agent) == "completed"
            if has_content and agent_done:
                count += 1
        return count

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        latest_section = None
        latest_content = None
        for section, content in self.report_sections.items():
            if content:
                latest_section = section
                latest_content = content

        if latest_section and latest_content:
            section_titles = {
                "market_report": "Market Analysis",
                "news_report": "News / Announcements / Macro",
                "fundamentals_report": "Fundamentals Analysis",
                "bull_thesis": "Bull Thesis",
                "bear_thesis": "Bear Thesis",
                "final_trade_decision": "Portfolio Manager Decision",
                "audit_report": "Decision Audit",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []
        analyst_sections = [
            ("market_report", "Market Analysis"),
            ("news_report", "News / Announcements / Macro"),
            ("fundamentals_report", "Fundamentals Analysis"),
        ]
        analyst_parts = [
            f"### {title}\n{self.report_sections[key]}"
            for key, title in analyst_sections
            if self.report_sections.get(key)
        ]
        if analyst_parts:
            report_parts.append("## Analyst Team Reports")
            report_parts.extend(analyst_parts)

        if self.report_sections.get("bull_thesis"):
            report_parts.append("## Bull Thesis")
            report_parts.append(self.report_sections["bull_thesis"])
        if self.report_sections.get("bear_thesis"):
            report_parts.append("## Bear Thesis")
            report_parts.append(self.report_sections["bear_thesis"])
        if self.report_sections.get("final_trade_decision"):
            report_parts.append("## Portfolio Manager Decision")
            report_parts.append(self.report_sections["final_trade_decision"])
        if self.report_sections.get("audit_report"):
            report_parts.append("## Decision Audit")
            report_parts.append(self.report_sections["audit_report"])

        self.final_report = "\n\n".join(report_parts) if report_parts else None

message_buffer = MessageBuffer()


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def format_tokens(n):
    """Format token count for display."""
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)


def update_display(layout, spinner_text=None, stats_handler=None, start_time=None):
    # Header with welcome message
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="Welcome to TradingAgents",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # Progress panel showing agent status
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,  # Use simple header with horizontal lines
        title=None,  # Remove the redundant Progress title
        padding=(0, 2),  # Add horizontal padding
        expand=True,  # Make table expand to fill available space
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    # Group agents exactly as they exist in the current graph.
    all_teams = {
        "Analyst Team": [
            "Market Analyst",
            "News & Sentiment Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher"],
        "Decision Team": ["Portfolio Manager", "Decision Auditor"],
    }
    # Filter teams to only include agents that are in agent_status
    teams = {}
    for team, agents in all_teams.items():
        active_agents = [a for a in agents if a in message_buffer.agent_status]
        if active_agents:
            teams[team] = active_agents

    for team, agents in teams.items():
        # Add first agent with team name
        first_agent = agents[0]
        status = message_buffer.agent_status.get(first_agent, "pending")
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        # Add remaining agents in team
        for agent in agents[1:]:
            status = message_buffer.agent_status.get(agent, "pending")
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        # Add horizontal line after each team
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # Messages panel showing recent messages and tool calls
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,  # Make table expand to fill available space
        box=box.MINIMAL,  # Use minimal box style for a lighter look
        show_lines=True,  # Keep horizontal lines
        padding=(0, 1),  # Add some padding between columns
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column(
        "Content", style="white", no_wrap=False, ratio=1
    )  # Make content column expand

    # Combine tool calls and messages
    all_messages = []

    # Add tool calls
    for timestamp, tool_name, args in message_buffer.tool_calls:
        formatted_args = format_tool_args(args)
        all_messages.append((timestamp, "Tool", f"{tool_name}: {formatted_args}"))

    # Add regular messages
    for timestamp, msg_type, content in message_buffer.messages:
        content_str = str(content) if content else ""
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    # Sort by timestamp descending (newest first)
    all_messages.sort(key=lambda x: x[0], reverse=True)

    # Calculate how many messages we can show based on available space
    max_messages = 12

    # Get the first N messages (newest ones)
    recent_messages = all_messages[:max_messages]

    # Add messages to table (already in newest-first order)
    for timestamp, msg_type, content in recent_messages:
        # Format content with word wrapping
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Analysis panel showing current report
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Footer with statistics
    # Agent progress - derived from agent_status dict
    agents_completed = sum(
        1 for status in message_buffer.agent_status.values() if status == "completed"
    )
    agents_total = len(message_buffer.agent_status)

    # Report progress - based on agent completion (not just content existence)
    reports_completed = message_buffer.get_completed_reports_count()
    reports_total = len(message_buffer.report_sections)

    # Build stats parts
    stats_parts = [f"Agents: {agents_completed}/{agents_total}"]

    # LLM and tool stats from callback handler
    if stats_handler:
        stats = stats_handler.get_stats()
        stats_parts.append(f"LLM: {stats['llm_calls']}")
        stats_parts.append(f"Tools: {stats['tool_calls']}")

        # Token display with graceful fallback
        if stats["tokens_in"] > 0 or stats["tokens_out"] > 0:
            tokens_str = f"Tokens: {format_tokens(stats['tokens_in'])}\u2191 {format_tokens(stats['tokens_out'])}\u2193"
        else:
            tokens_str = "Tokens: --"
        stats_parts.append(tokens_str)

    stats_parts.append(f"Reports: {reports_completed}/{reports_total}")

    # Elapsed time
    if start_time:
        elapsed = time.time() - start_time
        elapsed_str = f"\u23f1 {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        stats_parts.append(elapsed_str)

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(" | ".join(stats_parts))

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def get_user_selections():
    """Get all user selections before starting the analysis display."""
    # Display ASCII art welcome message
    with open(Path(__file__).parent / "static" / "welcome.txt", encoding="utf-8") as f:
        welcome_ascii = f.read()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Bull/Bear Research → III. Portfolio Manager → IV. Decision Auditor\n\n"
    welcome_content += (
        "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
    )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()
    console.print()  # Add vertical space before announcements

    # Fetch and display announcements (silent on failure)
    announcements = fetch_announcements()
    display_announcements(console, announcements)

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    def thinking_value_or_prompt(env_var, config_key, label, box_title, box_body, prompt_fn):
        """Return the env-configured reasoning/thinking value, or prompt for it.

        When ``env_var`` is set the interactive choice is skipped and the value
        the env overlay placed on DEFAULT_CONFIG is used — mirroring the
        env-precedence rule applied to the other selection steps.
        """
        if os.environ.get(env_var):
            value = DEFAULT_CONFIG[config_key]
            console.print(f"[green]✓ {label} from environment:[/green] {value}")
            return value
        console.print(create_question_box(box_title, box_body))
        return prompt_fn()

    # Step 1: Ticker symbol
    console.print(
        create_question_box(
            "Step 1: Ticker Symbol",
            "Enter the ticker, with exchange suffix when needed (e.g. SPY, 0700.HK, BTC-USD)",
            "SPY",
        )
    )
    selected_ticker = get_ticker()
    asset_type = detect_asset_type(selected_ticker)
    # Only announce when it's not the default stock path, to avoid printing
    # "stock" on every run.
    if asset_type.value != "stock":
        console.print(
            f"[green]Detected asset type:[/green] {asset_type.value}"
        )

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 2: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    # Step 3: Output language (skipped when set via TRADINGAGENTS_OUTPUT_LANGUAGE)
    if os.environ.get("TRADINGAGENTS_OUTPUT_LANGUAGE"):
        output_language = DEFAULT_CONFIG["output_language"]
        console.print(
            f"[green]✓ Output language from environment:[/green] {output_language}"
        )
    else:
        console.print(
            create_question_box(
                "Step 3: Output Language",
                "Select the language for analyst reports and final decision"
            )
        )
        output_language = ask_output_language()

    # Step 4: Select analysts
    console.print(
        create_question_box(
            "Step 4: Analysts Team", "Select your LLM analyst agents for the analysis"
        )
    )
    selected_analysts = select_analysts(asset_type)
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # Step 5: LLM Provider    # Step 5: LLM Provider (skipped when set via TRADINGAGENTS_LLM_PROVIDER).
    # The backend URL comes from TRADINGAGENTS_LLM_BACKEND_URL when set,
    # otherwise the provider's default endpoint — the same value the menu
    # would have picked.
    provider_from_env = bool(os.environ.get("TRADINGAGENTS_LLM_PROVIDER"))
    if provider_from_env:
        selected_llm_provider = DEFAULT_CONFIG["llm_provider"].lower()
        backend_url = resolve_backend_url(
            selected_llm_provider, env_url=DEFAULT_CONFIG["backend_url"]
        )
        console.print(f"[green]✓ LLM provider from environment:[/green] {selected_llm_provider}")
        console.print(f"[green]✓ Backend URL:[/green] {backend_url}")
        # Still confirm/persist the API key so the run doesn't fail later.
        ensure_api_key(selected_llm_provider)
    else:
        console.print(
            create_question_box(
                "Step 5: LLM Provider", "Select your LLM provider"
            )
        )
        selected_llm_provider, backend_url = select_llm_provider()

        # Providers with regional endpoints prompt for the region as a secondary
        # step so the main dropdown stays clean (mainland China and international
        # accounts cannot share API keys).
        if selected_llm_provider == "qwen":
            selected_llm_provider, backend_url = ask_qwen_region()
        elif selected_llm_provider == "minimax":
            selected_llm_provider, backend_url = ask_minimax_region()
        elif selected_llm_provider == "glm":
            selected_llm_provider, backend_url = ask_glm_region()

        # Honor an explicit env backend URL even when the provider was chosen
        # interactively, so it isn't overwritten by the menu default (#978).
        backend_url = resolve_backend_url(
            selected_llm_provider, backend_url, env_url=DEFAULT_CONFIG["backend_url"]
        )

        # The generic OpenAI-compatible endpoint has no default; ask for it if
        # neither the menu nor the environment supplied one.
        if selected_llm_provider == "openai_compatible" and not backend_url:
            backend_url = prompt_openai_compatible_url()

        # For Ollama, surface the resolved endpoint (OLLAMA_BASE_URL vs default)
        # before model selection so it's obvious where we're connecting.
        if selected_llm_provider == "ollama":
            confirm_ollama_endpoint(backend_url)

        # Confirm the provider's API key is present; prompt the user to paste
        # one and persist it to .env if it's missing, so the analysis run
        # doesn't fail later at the first API call.
        ensure_api_key(selected_llm_provider)

    # Step 6: Thinking agents (skipped when either model is set via environment)
    if os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM") or os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM"):
        selected_shallow_thinker = DEFAULT_CONFIG["quick_think_llm"]
        selected_deep_thinker = DEFAULT_CONFIG["deep_think_llm"]
        console.print(
            f"[green]✓ Thinking agents from environment:[/green] "
            f"quick={selected_shallow_thinker}, deep={selected_deep_thinker}"
        )
    else:
        console.print(
            create_question_box(
                "Step 6: Thinking Agents", "Select your thinking agents for analysis"
            )
        )
        selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
        selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    # Step 7: Provider-specific reasoning/thinking configuration. Each knob is
    # settable via its TRADINGAGENTS_* env var; when that var is set (or the
    # provider itself came from env) the prompt is skipped and the configured
    # value is used — same env-precedence rule as the steps above. None = each
    # provider's own default.
    thinking_level = None
    reasoning_effort = None
    anthropic_effort = None

    provider_lower = selected_llm_provider.lower()
    if provider_from_env:
        thinking_level = DEFAULT_CONFIG["google_thinking_level"]
        reasoning_effort = DEFAULT_CONFIG["openai_reasoning_effort"]
        anthropic_effort = DEFAULT_CONFIG["anthropic_effort"]
    elif provider_lower == "google":
        thinking_level = thinking_value_or_prompt(
            "TRADINGAGENTS_GOOGLE_THINKING_LEVEL", "google_thinking_level",
            "Gemini thinking mode", "Step 7: Thinking Mode",
            "Configure Gemini thinking mode", ask_gemini_thinking_config,
        )
    elif provider_lower == "openai":
        reasoning_effort = thinking_value_or_prompt(
            "TRADINGAGENTS_OPENAI_REASONING_EFFORT", "openai_reasoning_effort",
            "Reasoning effort", "Step 7: Reasoning Effort",
            "Configure OpenAI reasoning effort level", ask_openai_reasoning_effort,
        )
    elif provider_lower == "anthropic":
        anthropic_effort = thinking_value_or_prompt(
            "TRADINGAGENTS_ANTHROPIC_EFFORT", "anthropic_effort",
            "Claude effort", "Step 7: Effort Level",
            "Configure Claude effort level", ask_anthropic_effort,
        )

    return {
        "ticker": selected_ticker,
        "asset_type": asset_type.value,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
        "anthropic_effort": anthropic_effort,
        "output_language": output_language,
    }


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # Validate date format and ensure it's not in the future
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


def save_report_to_disk(final_state, ticker: str, save_path: Path):
    """Save the complete analysis report to disk (shared CLI/API writer)."""
    return write_report_tree(final_state, ticker, save_path)


def display_complete_report(final_state):
    """Display the current seven-agent report sequentially."""
    console.print()
    console.print(Rule("Complete Analysis Report", style="bold green"))

    analysts = []
    if final_state.get("market_report"):
        analysts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("news_report"):
        analysts.append(("News & Sentiment Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analysts:
        console.print(Panel("[bold]I. Analyst Team Reports[/bold]", border_style="cyan"))
        for title, content in analysts:
            console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    research = []
    if final_state.get("bull_thesis"):
        research.append(("Bull Researcher", final_state["bull_thesis"]))
    if final_state.get("bear_thesis"):
        research.append(("Bear Researcher", final_state["bear_thesis"]))
    if research:
        console.print(Panel("[bold]II. Bull / Bear Research[/bold]", border_style="magenta"))
        for title, content in research:
            console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    if final_state.get("final_trade_decision"):
        console.print(Panel("[bold]III. Portfolio Manager Decision[/bold]", border_style="green"))
        console.print(Panel(Markdown(final_state["final_trade_decision"]), title="Portfolio Manager", border_style="blue", padding=(1, 2)))

    if final_state.get("audit_report"):
        console.print(Panel("[bold]IV. Decision Audit[/bold]", border_style="yellow"))
        console.print(Panel(Markdown(final_state["audit_report"]), title="Decision Auditor", border_style="blue", padding=(1, 2)))


# Analyst status mappings for the three parallel private subgraphs.
ANALYST_ORDER = ["market", "news", "fundamentals"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "news": "News & Sentiment Analyst",
    "fundamentals": "Fundamentals Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


def update_analyst_statuses(message_buffer, chunk, wall_time_tracker=None):
    """Synchronize progress for analysts that actually run in parallel."""
    selected = message_buffer.selected_analysts

    if wall_time_tracker is not None:
        sync_analyst_tracker_from_chunk(wall_time_tracker, chunk)

    all_complete = bool(selected)
    for analyst_key in ANALYST_ORDER:
        if analyst_key not in selected:
            continue
        agent_name = ANALYST_AGENT_NAMES[analyst_key]
        report_key = ANALYST_REPORT_MAP[analyst_key]
        if chunk.get(report_key):
            message_buffer.update_report_section(report_key, chunk[report_key])
        has_report = bool(message_buffer.report_sections.get(report_key))
        message_buffer.update_agent_status(
            agent_name, "completed" if has_report else "in_progress"
        )
        all_complete = all_complete and has_report

    if all_complete:
        if message_buffer.agent_status.get("Bull Researcher") == "pending":
            message_buffer.update_agent_status("Bull Researcher", "in_progress")
        if message_buffer.agent_status.get("Bear Researcher") == "pending":
            message_buffer.update_agent_status("Bear Researcher", "in_progress")

def extract_content_string(content):
    """Extract string content from various message formats.
    Returns None if no meaningful text content is found.
    """
    import ast

    def is_empty(val):
        """Check if value is empty using Python's truthiness."""
        if val is None or val == '':
            return True
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return True
            try:
                return not bool(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                return False  # Can't parse = real text
        return not bool(val)

    if is_empty(content):
        return None

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        text = content.get('text', '')
        return text.strip() if not is_empty(text) else None

    if isinstance(content, list):
        text_parts = [
            item.get('text', '').strip() if isinstance(item, dict) and item.get('type') == 'text'
            else (item.strip() if isinstance(item, str) else '')
            for item in content
        ]
        result = ' '.join(t for t in text_parts if t and not is_empty(t))
        return result if result else None

    return str(content).strip() if not is_empty(content) else None


def classify_message_type(message) -> tuple[str, str | None]:
    """Classify LangChain message into display type and extract content.

    Returns:
        (type, content) - type is one of: User, Agent, Data, Control
                        - content is extracted string or None
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    content = extract_content_string(getattr(message, 'content', None))

    if isinstance(message, HumanMessage):
        if content and content.strip() == "Continue":
            return ("Control", content)
        return ("User", content)

    if isinstance(message, ToolMessage):
        return ("Data", content)

    if isinstance(message, AIMessage):
        return ("Agent", content)

    # Fallback for unknown types
    return ("System", content)


def format_tool_args(args, max_length=80) -> str:
    """Format tool arguments for terminal display."""
    result = str(args)
    if len(result) > max_length:
        return result[:max_length - 3] + "..."
    return result

def _build_run_config(selections: dict, checkpoint: bool | None) -> dict:
    """Assemble run config while preserving environment overrides."""
    config = DEFAULT_CONFIG.copy()
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")
    config["anthropic_effort"] = selections.get("anthropic_effort")
    config["output_language"] = selections.get("output_language", "English")
    if checkpoint is not None:
        config["checkpoint_enabled"] = checkpoint
    return config

def run_analysis(checkpoint: bool | None = None):
    # First get all user selections
    selections = get_user_selections()

    config = _build_run_config(selections, checkpoint)

    # Create stats callback handler for tracking LLM/tool calls
    stats_handler = StatsCallbackHandler()

    # Normalize analyst selection to predefined order (selection is a 'set', order is fixed)
    selected_set = {analyst.value for analyst in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]
    analyst_execution_plan = build_analyst_execution_plan(selected_analyst_keys)
    analyst_wall_time_tracker = AnalystWallTimeTracker(analyst_execution_plan)

    # Initialize the graph with callbacks bound to LLMs
    graph = TradingAgentsGraph(
        selected_analyst_keys,
        config=config,
        debug=True,
        callbacks=[stats_handler],
    )

    # Initialize message buffer with selected analysts
    message_buffer.init_for_analysis(selected_analyst_keys)

    # Track start time for elapsed display
    start_time = time.time()

    # Create result directory
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper

    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    text = "\n".join(str(item) for item in content) if isinstance(content, list) else content
                    with open(report_dir / file_name, "w", encoding="utf-8") as f:
                        f.write(text)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    # Now start the display layout
    layout = create_layout()

    with Live(layout, refresh_per_second=4):
        # Initial display
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Add initial messages
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        if selections["asset_type"] != "stock":
            message_buffer.add_message("System", f"Detected asset type: {selections['asset_type']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # All selected analysts fan out from START and begin concurrently.
        for spec in analyst_execution_plan.specs:
            message_buffer.update_agent_status(spec.agent_node, "in_progress")
            analyst_wall_time_tracker.mark_started(spec.key)
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Create spinner text
        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text, stats_handler=stats_handler, start_time=start_time)

        # Initialize state and get graph args with callbacks.
        # Resolve the instrument identity once here so all agents anchor to
        # the real company (#814); the CLI builds state directly rather than
        # going through propagate(), so this must happen on the CLI path too.
        instrument_context = graph.resolve_instrument_context(
            selections["ticker"], selections["asset_type"]
        )
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"],
            selections["analysis_date"],
            asset_type=selections["asset_type"],
            instrument_context=instrument_context,
        )
        # Pass callbacks to graph config for tool execution tracking
        # (LLM tracking is handled separately via LLM constructor)
        args = graph.propagator.get_graph_args(callbacks=[stats_handler])

        # Recompile with a checkpointer and inject the thread_id so --checkpoint
        # actually saves and resumes on the CLI path (#1249); a no-op when
        # checkpointing is disabled. Torn down in the finally below.
        checkpoint_tid = graph.begin_checkpoint(
            selections["ticker"], selections["analysis_date"], selections["asset_type"]
        )
        if checkpoint_tid is not None:
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = checkpoint_tid

        # Stream the analysis. On resume, feed None so LangGraph continues the
        # interrupted run instead of re-appending the initial state (#1249); the
        # try/finally tears the checkpointer down even if the stream raises.
        trace = []
        try:
            for chunk in graph.graph.stream(graph.checkpoint_input(init_agent_state), **args):
                # Process all messages in chunk, deduplicating by message ID
                for message in chunk.get("messages", []):
                    msg_id = getattr(message, "id", None)
                    if msg_id is not None:
                        if msg_id in message_buffer._processed_message_ids:
                            continue
                        message_buffer._processed_message_ids.add(msg_id)

                    msg_type, content = classify_message_type(message)
                    if content and content.strip():
                        message_buffer.add_message(msg_type, content)

                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            if isinstance(tool_call, dict):
                                message_buffer.add_tool_call(tool_call["name"], tool_call["args"])
                            else:
                                message_buffer.add_tool_call(tool_call.name, tool_call.args)

                # Update analyst statuses based on report state (runs on every chunk)
                update_analyst_statuses(
                    message_buffer,
                    chunk,
                    wall_time_tracker=analyst_wall_time_tracker,
                )

                # Bull/Bear write independent hypotheses into distinct state keys.
                if chunk.get("bull_thesis"):
                    message_buffer.update_report_section("bull_thesis", chunk["bull_thesis"])
                    message_buffer.update_agent_status("Bull Researcher", "completed")
                if chunk.get("bear_thesis"):
                    message_buffer.update_report_section("bear_thesis", chunk["bear_thesis"])
                    message_buffer.update_agent_status("Bear Researcher", "completed")

                bull_done = message_buffer.agent_status.get("Bull Researcher") == "completed"
                bear_done = message_buffer.agent_status.get("Bear Researcher") == "completed"
                if bull_done and bear_done and message_buffer.agent_status.get("Portfolio Manager") == "pending":
                    message_buffer.update_agent_status("Portfolio Manager", "in_progress")

                if chunk.get("final_trade_decision"):
                    message_buffer.update_report_section(
                        "final_trade_decision", chunk["final_trade_decision"]
                    )
                    message_buffer.update_agent_status("Portfolio Manager", "completed")
                    message_buffer.update_agent_status("Decision Auditor", "in_progress")

                if chunk.get("audit_report"):
                    message_buffer.update_report_section("audit_report", chunk["audit_report"])
                    message_buffer.update_agent_status("Decision Auditor", "completed")
                    if str(chunk.get("audit_status", "")).upper() == "REVISE":
                        message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                # Update the display
                update_display(layout, stats_handler=stats_handler, start_time=start_time)

                trace.append(chunk)

            # Clean run: drop this run's checkpoint so a later run starts fresh.
            # A mid-stream failure skips this, keeping the checkpoint for resume.
            graph.clear_checkpoint_on_success(
                selections["ticker"], selections["analysis_date"], selections["asset_type"]
            )
        finally:
            # Always restore the plain uncheckpointed graph, even on failure.
            graph.end_checkpoint()

        # Streamed chunks are per-node deltas, not full state. Merge them
        # so every report field populated across the run is present.
        final_state = {}
        for chunk in trace:
            final_state.update(chunk)

        # Update all agent statuses to completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "System", f"Completed analysis for {selections['analysis_date']}"
        )
        message_buffer.add_message("System", analyst_wall_time_tracker.format_summary())

        # Update final report sections
        for section in message_buffer.report_sections:
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        update_display(layout, stats_handler=stats_handler, start_time=start_time)

    # Post-analysis prompts (outside Live context for clean interaction)
    console.print("\n[bold cyan]Analysis Complete![/bold cyan]\n")
    console.print(f"[dim]{analyst_wall_time_tracker.format_summary()}[/dim]")

    # Prompt to save report
    save_choice = typer.prompt("Save report?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path.cwd() / "reports" / f"{selections['ticker']}_{timestamp}"
        save_path_str = typer.prompt(
            "Save path (press Enter for default)",
            default=str(default_path)
        ).strip()
        save_path = Path(save_path_str)
        try:
            report_file = save_report_to_disk(final_state, selections["ticker"], save_path)
            console.print(f"\n[green]✓ Report saved to:[/green] {save_path.resolve()}")
            console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")

    # Prompt to display full report
    display_choice = typer.prompt("\nDisplay full report on screen?", default="Y").strip().upper()
    if display_choice in ("Y", "YES", ""):
        display_complete_report(final_state)


@app.command()
def analyze(
    checkpoint: bool | None = typer.Option(
        None,
        "--checkpoint/--no-checkpoint",
        help="Enable/disable checkpoint-resume (save state after each node so a "
        "crashed run can resume). Omit to honor TRADINGAGENTS_CHECKPOINT_ENABLED.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
):
    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    try:
        run_analysis(checkpoint=checkpoint)
    except _NO_CONSOLE_ERRORS:
        # A terminal with no console buffer cannot host the interactive prompts.
        # Emit one actionable line on stderr instead of a prompt_toolkit
        # traceback; plain text, since rich may not render here either (#1138).
        typer.echo(
            "Error: no Windows console available. The interactive CLI needs a real "
            "console buffer — run it from Windows Terminal, PowerShell, or cmd.exe "
            "rather than a piped or embedded terminal.",
            err=True,
        )
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
