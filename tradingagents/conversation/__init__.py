"""Conversation layer for the TradingAgents application service."""

from .store import ConversationStore

__all__ = ["ConversationAgent", "ConversationStore"]


def __getattr__(name):
    if name == "ConversationAgent":
        from .agent import ConversationAgent
        return ConversationAgent
    raise AttributeError(name)
