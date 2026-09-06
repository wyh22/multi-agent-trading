from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.evidence_claims import claim_boundary_instruction

def _needs_insider_transactions(messages) -> bool:
    """Expose the expensive insider tool only for explicitly relevant requests."""

    keywords = (
        "高管持股",
        "高管增持",
        "高管减持",
        "董监高",
        "管理层持股",
        "内部人交易",
        "insider",
        "增减持",
    )
    text = " ".join(
        str(getattr(message, "content", "") or "")
        for message in (messages or [])
    ).lower()
    return any(keyword in text for keyword in keywords)


from tradingagents.agents.utils.agent_utils import (
    get_candidate_context_from_state,
    get_global_news,
    get_instrument_context_from_state,
    get_language_instruction,
    get_macro_indicators,
    get_news,
)


def create_news_analyst(llm, tools=None):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)
        candidate_context = get_candidate_context_from_state(state)

        active_tools = list(
            tools
            or [
                get_news,
                get_global_news,
                get_macro_indicators,
            ]
        )
        if not _needs_insider_transactions(state.get("messages", [])):
            active_tools = [
                tool
                for tool in active_tools
                if tool.name != "get_insider_transactions"
            ]
        active_tool_names = {tool.name for tool in active_tools}
        optional_guidance = ""
        if "get_insider_transactions" in active_tool_names:
            optional_guidance += (
                " Use get_insider_transactions only because this request explicitly "
                "concerns management/related-person holding changes; treat them as "
                "event evidence, not as an automatic bullish/bearish signal."
            )
        if "search_company_knowledge" in active_tool_names:
            optional_guidance += (
                " When available, use search_company_knowledge for PIT-aware "
                "company documents such as filings/announcements; every retrieval "
                "must use the current research cutoff date."
            )

        system_message = (
            f"You are the News & Sentiment Analyst for an A-share research workflow. Analyze company announcements/news, broader market information, and Chinese macro conditions available no later than the research cutoff date. Use get_news(ticker, start_date, end_date) for {asset_label}-specific evidence. For vague terms such as 'recent' or '近期', default to roughly the latest 90 calendar days and do not request multi-year history unless the user explicitly asks for it. Use get_global_news(curr_date, look_back_days, limit) for broader market information, and get_macro_indicators(indicator, curr_date, look_back_days) for PIT-aware Chinese macro series such as cpi, ppi, gdp, pmi, lpr, m2, or unemployment. Treat missing data explicitly as uncertainty and never fabricate facts, probabilities, or market sentiment."
            + optional_guidance
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + claim_boundary_instruction()
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " Produce research evidence only; do not emit an automatic trading instruction or transaction proposal."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in active_tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(candidate_context=candidate_context)

        chain = prompt | llm.bind_tools(active_tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
