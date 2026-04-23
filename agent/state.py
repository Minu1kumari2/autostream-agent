"""
State definitions for the AutoStream LangGraph agent.
"""

from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


# Intent classification literals
IntentType = Literal[
    "casual_greeting",
    "product_inquiry",
    "high_intent_lead",
    "unknown",
]


class LeadInfo(TypedDict, total=False):
    """Partial lead data collected during the conversation."""
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]


class AgentState(TypedDict):
    """
    Shared state object that flows through every LangGraph node.

    Fields:
        messages:       Full conversation history (user + assistant turns).
                        Uses add_messages reducer so turns are appended, not replaced.
        intent:         Latest classified intent for the most recent user message.
        lead_info:      Partial lead data collected so far.
        lead_captured:  Whether mock_lead_capture() has been successfully called.
        rag_context:    Retrieved knowledge-base snippets for the current turn.
        awaiting_field: Which lead field the agent is currently waiting for
                        ("name" | "email" | "platform" | None).
    """
    messages: Annotated[list, add_messages]
    intent: IntentType
    lead_info: LeadInfo
    lead_captured: bool
    rag_context: str
    awaiting_field: Optional[Literal["name", "email", "platform"]]
