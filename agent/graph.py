"""
AutoStream Conversational AI Agent — LangGraph Implementation.

Graph topology:
  __start__
      │
  [retrieve_context]   ← RAG: fetch KB snippets
      │
  [classify_intent]    ← LLM call: label the user's intent
      │
      ├─ casual_greeting / product_inquiry / unknown
      │         └─► [generate_response]  ← LLM call: answer using RAG context
      │
      └─ high_intent_lead
                ├─ lead already captured? → [generate_response]
                └─ fields missing?
                        ├─ awaiting field → [extract_field] → [generate_response]
                        └─ not yet asked  → [generate_response] (ask for next field)

All nodes read/write the shared AgentState TypedDict.
"""

import json
import os
import re
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.rag_pipeline import retrieve_context
from agent.state import AgentState, IntentType, LeadInfo
from tools.lead_capture import mock_lead_capture

# ---------------------------------------------------------------------------
# LLM initialisation
# ---------------------------------------------------------------------------

def _build_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Get a free key at https://aistudio.google.com and export it before running."
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.3,
        max_output_tokens=512,
    )


# Module-level singleton (initialised lazily to avoid import-time errors)
_llm: Optional[ChatGoogleGenerativeAI] = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = _build_llm()
    return _llm


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _last_human_message(state: AgentState) -> str:
    """Return the text of the most recent HumanMessage."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _extract_email(text: str) -> Optional[str]:
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else None


def _next_missing_field(lead_info: LeadInfo) -> Optional[str]:
    """Return the first field that hasn't been collected yet."""
    for field in ("name", "email", "platform"):
        if not lead_info.get(field):
            return field
    return None  # all three collected


# ---------------------------------------------------------------------------
# Node: retrieve_context
# ---------------------------------------------------------------------------

def node_retrieve_context(state: AgentState) -> dict:
    """RAG step: retrieve relevant KB snippets for the latest user message."""
    user_msg = _last_human_message(state)
    context = retrieve_context(user_msg)
    return {"rag_context": context}


# ---------------------------------------------------------------------------
# Node: classify_intent
# ---------------------------------------------------------------------------

_INTENT_SYSTEM = """\
You are an intent classifier for AutoStream, a video-editing SaaS.

Classify the user's LATEST message into exactly ONE of these intents:
- casual_greeting   : simple hello / hi / greeting, no product interest yet
- product_inquiry   : asking about features, pricing, plans, policies, or how things work
- high_intent_lead  : clearly wants to sign up, try, purchase, or subscribe to a plan

Respond with a JSON object only — no prose, no markdown fences:
{"intent": "<one of the three values above>"}

Context so far (conversation history is included in the messages).
"""


def node_classify_intent(state: AgentState) -> dict:
    """LLM call to classify the user's latest intent."""
    llm = get_llm()
    history = state["messages"][-6:]  # last 3 turns for context

    prompt_messages = [SystemMessage(content=_INTENT_SYSTEM)] + history

    response = llm.invoke(prompt_messages)
    raw = response.content.strip()

    # Strip any accidental markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        parsed = json.loads(raw)
        intent: IntentType = parsed.get("intent", "unknown")
        if intent not in ("casual_greeting", "product_inquiry", "high_intent_lead"):
            intent = "unknown"
    except (json.JSONDecodeError, AttributeError):
        intent = "unknown"

    return {"intent": intent}


# ---------------------------------------------------------------------------
# Node: extract_field
# ---------------------------------------------------------------------------

def node_extract_field(state: AgentState) -> dict:
    """
    If the agent is awaiting a specific lead field, try to extract it from
    the user's latest message without another LLM call for simple cases.
    Falls back to an LLM extraction for ambiguous input.
    """
    awaiting = state.get("awaiting_field")
    if not awaiting:
        return {}

    user_msg = _last_human_message(state)
    lead_info = dict(state.get("lead_info") or {})

    if awaiting == "email":
        email = _extract_email(user_msg)
        if email:
            lead_info["email"] = email
            return {"lead_info": lead_info, "awaiting_field": None}

    elif awaiting == "name":
        # Accept anything that looks like a real name (2-50 chars, letters/spaces)
        cleaned = user_msg.strip()
        if 2 <= len(cleaned) <= 50 and re.match(r"^[A-Za-z][\w .'-]*$", cleaned):
            lead_info["name"] = cleaned.title()
            return {"lead_info": lead_info, "awaiting_field": None}

    elif awaiting == "platform":
        known_platforms = ["youtube", "instagram", "tiktok", "twitter", "facebook",
                           "twitch", "linkedin", "snapchat", "pinterest"]
        lower = user_msg.lower()
        for p in known_platforms:
            if p in lower:
                lead_info["platform"] = p.capitalize()
                return {"lead_info": lead_info, "awaiting_field": None}
        # Accept freeform platform name if it looks reasonable
        cleaned = user_msg.strip()
        if 2 <= len(cleaned) <= 40:
            lead_info["platform"] = cleaned.capitalize()
            return {"lead_info": lead_info, "awaiting_field": None}

    # Could not extract — keep awaiting the same field
    return {"lead_info": lead_info}


# ---------------------------------------------------------------------------
# Node: generate_response
# ---------------------------------------------------------------------------

_AGENT_SYSTEM_TEMPLATE = """\
You are Alex, a friendly and knowledgeable sales assistant for AutoStream — an AI-powered
video editing SaaS for content creators.

KNOWLEDGE BASE CONTEXT (use this to answer questions accurately):
{rag_context}

GUIDELINES:
1. Be warm, concise, and professional.
2. Answer product/pricing questions strictly from the Knowledge Base above.
3. When the user shows HIGH INTENT (wants to sign up / try a plan):
   - Collect: full name, email, and creator platform — ONE at a time, in that order.
   - Do NOT ask for multiple fields in the same message.
   - Once all three are collected, confirm the sign-up enthusiastically.
4. Never make up features, prices, or policies not in the Knowledge Base.
5. Keep responses under 120 words unless a detailed explanation is needed.

CURRENT LEAD INFO COLLECTED:
{lead_info_summary}

LEAD CAPTURED: {lead_captured}
"""


def node_generate_response(state: AgentState) -> dict:
    """Main LLM call: generate the assistant reply for the current turn."""
    llm = get_llm()

    lead_info: LeadInfo = state.get("lead_info") or {}
    lead_captured: bool = state.get("lead_captured", False)
    intent: IntentType = state.get("intent", "unknown")
    rag_context: str = state.get("rag_context", "")

    # Summarise what lead data we have so far
    collected = {k: v for k, v in lead_info.items() if v}
    missing = [f for f in ("name", "email", "platform") if not lead_info.get(f)]

    lead_info_summary = (
        ", ".join(f"{k}: {v}" for k, v in collected.items())
        if collected else "none yet"
    )

    system_prompt = _AGENT_SYSTEM_TEMPLATE.format(
        rag_context=rag_context,
        lead_info_summary=lead_info_summary,
        lead_captured=lead_captured,
    )

    # ── High-intent path: lead capture flow ──────────────────────────────────
    new_lead_info = dict(lead_info)
    new_lead_captured = lead_captured
    awaiting_field = state.get("awaiting_field")

    if intent == "high_intent_lead" and not lead_captured:
        # Check if all fields are now present (may have just been extracted)
        if not missing:
            # 🎯 All three fields collected — fire the tool
            try:
                mock_lead_capture(
                    name=new_lead_info["name"],
                    email=new_lead_info["email"],
                    platform=new_lead_info["platform"],
                )
                new_lead_captured = True
                # Craft a confirmation message without an extra LLM call
                confirmation = (
                    f"🎉 You're all set, {new_lead_info['name']}! "
                    f"I've captured your details and our team will reach out to "
                    f"{new_lead_info['email']} shortly to get you started on AutoStream Pro. "
                    f"Welcome aboard — can't wait to see what you create on {new_lead_info['platform']}!"
                )
                return {
                    "messages": [AIMessage(content=confirmation)],
                    "lead_captured": True,
                    "awaiting_field": None,
                }
            except Exception as e:
                # Tool error — inform user gracefully
                error_msg = (
                    "I ran into a small issue saving your details. "
                    f"Could you double-check your email address? ({e})"
                )
                return {"messages": [AIMessage(content=error_msg)]}

        # Still missing fields — ask for the next one
        next_field = missing[0]
        awaiting_field = next_field

        field_questions = {
            "name": (
                "I'd love to get you signed up! Could you share your full name first?"
            ),
            "email": (
                f"Great, {collected.get('name', 'there')}! "
                "What's the best email address to reach you at?"
            ),
            "platform": (
                "Almost there! Which creator platform do you mainly use? "
                "(e.g. YouTube, Instagram, TikTok…)"
            ),
        }
        reply = field_questions[next_field]
        return {
            "messages": [AIMessage(content=reply)],
            "awaiting_field": next_field,
            "lead_info": new_lead_info,
        }

    # ── General response path ─────────────────────────────────────────────────
    history = state["messages"][-8:]  # last 4 turns of context
    prompt_messages = [SystemMessage(content=system_prompt)] + history
    response = llm.invoke(prompt_messages)

    return {
        "messages": [AIMessage(content=response.content)],
        "lead_info": new_lead_info,
        "lead_captured": new_lead_captured,
        "awaiting_field": awaiting_field,
    }


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def route_after_intent(state: AgentState) -> str:
    """
    After intent classification:
    - If high-intent AND awaiting a field (user just replied with a value) → extract first
    - Otherwise → go straight to generate_response
    """
    intent = state.get("intent", "unknown")
    awaiting = state.get("awaiting_field")

    if intent == "high_intent_lead" and awaiting:
        return "extract_field"
    return "generate_response"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("retrieve_context", node_retrieve_context)
    builder.add_node("classify_intent", node_classify_intent)
    builder.add_node("extract_field", node_extract_field)
    builder.add_node("generate_response", node_generate_response)

    # Edges
    builder.add_edge(START, "retrieve_context")
    builder.add_edge("retrieve_context", "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "extract_field": "extract_field",
            "generate_response": "generate_response",
        },
    )
    builder.add_edge("extract_field", "generate_response")
    builder.add_edge("generate_response", END)

    return builder.compile()


# Module-level compiled graph (lazy init)
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
