"""
RAG (Retrieval-Augmented Generation) Pipeline for AutoStream Knowledge Base.

Uses simple keyword/semantic matching over the local JSON knowledge base
to retrieve relevant context before passing to the LLM.
"""

import json
import re
from pathlib import Path
from typing import Optional


KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"


def load_knowledge_base() -> dict:
    """Load the JSON knowledge base from disk."""
    with open(KB_PATH, "r") as f:
        return json.load(f)


def _text_matches(text: str, query_tokens: list[str]) -> bool:
    """Return True if any query token appears in the lowercased text."""
    text_lower = text.lower()
    return any(tok in text_lower for tok in query_tokens)


def retrieve_context(query: str) -> str:
    """
    Retrieve relevant knowledge-base snippets for a given user query.

    Strategy:
    - Tokenise the query into lowercase keywords.
    - Score each KB section (company, plans, policies, FAQs) against those keywords.
    - Return the top-matching sections as a formatted string.

    Args:
        query: The raw user message.

    Returns:
        A plain-text context block to inject into the system prompt.
    """
    kb = load_knowledge_base()
    query_tokens = re.findall(r"\w+", query.lower())

    # --- price / plan keywords -------------------------------------------------
    plan_keywords = {"price", "pricing", "cost", "plan", "basic", "pro", "cheap",
                     "affordable", "subscription", "monthly", "pay", "dollar",
                     "upgrade", "tier", "package", "4k", "unlimited", "caption",
                     "resolution", "video", "videos", "features", "include"}

    # --- policy keywords -------------------------------------------------------
    policy_keywords = {"refund", "cancel", "cancellation", "support", "help",
                       "policy", "rule", "return", "money", "back", "storage",
                       "download", "24/7", "team", "agency"}

    # --- faq keywords ----------------------------------------------------------
    faq_keywords = {"platform", "youtube", "instagram", "tiktok", "trial", "free",
                    "upgrade", "team", "export", "account", "how", "can", "does"}

    sections: list[str] = []

    # Always include company info for greetings or generic queries
    company_triggers = {"hi", "hello", "hey", "what", "who", "about", "tell",
                        "autostream", "product", "service"}
    if any(t in company_triggers for t in query_tokens) or not sections:
        c = kb["company"]
        sections.append(
            f"Company: {c['name']}\nTagline: {c['tagline']}\nDescription: {c['description']}"
        )

    # Plans section
    if any(t in plan_keywords for t in query_tokens):
        plan_texts = []
        for plan in kb["plans"]:
            feats = "\n  - ".join(plan["features"])
            lims = ""
            if plan["limitations"]:
                lims = "\n  Limitations:\n  - " + "\n  - ".join(plan["limitations"])
            plan_texts.append(
                f"{plan['name']}: ${plan['price_monthly']}/month\n"
                f"  Features:\n  - {feats}{lims}"
            )
        sections.append("PRICING PLANS:\n" + "\n\n".join(plan_texts))

    # Policies section
    if any(t in policy_keywords for t in query_tokens):
        pol_texts = [f"{p['policy']}: {p['details']}" for p in kb["policies"]]
        sections.append("COMPANY POLICIES:\n" + "\n".join(pol_texts))

    # FAQs
    if any(t in faq_keywords for t in query_tokens):
        matched_faqs = [
            f"Q: {faq['question']}\nA: {faq['answer']}"
            for faq in kb["faqs"]
            if _text_matches(faq["question"] + " " + faq["answer"], query_tokens)
        ]
        if matched_faqs:
            sections.append("RELEVANT FAQs:\n" + "\n\n".join(matched_faqs))

    # Fallback: return entire KB summary if nothing matched
    if not sections:
        sections.append(
            "Company: AutoStream — automated video editing SaaS for content creators.\n"
            "Plans: Basic ($29/month, 10 videos, 720p) | Pro ($79/month, unlimited, 4K, AI captions).\n"
            "Policies: No refunds after 7 days. 24/7 support on Pro only."
        )

    return "\n\n---\n\n".join(sections)
