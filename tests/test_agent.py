"""
Unit tests for AutoStream agent components.
Run with: python -m pytest tests/ -v
"""

import json
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.lead_capture import mock_lead_capture
from agent.rag_pipeline import retrieve_context, load_knowledge_base


# ─────────────────────────────────────────────────────────────
# Tests: Knowledge Base
# ─────────────────────────────────────────────────────────────

class TestKnowledgeBase:
    def test_kb_loads(self):
        kb = load_knowledge_base()
        assert "plans" in kb
        assert "policies" in kb
        assert len(kb["plans"]) == 2

    def test_basic_plan_price(self):
        kb = load_knowledge_base()
        basic = next(p for p in kb["plans"] if p["name"] == "Basic Plan")
        assert basic["price_monthly"] == 29

    def test_pro_plan_price(self):
        kb = load_knowledge_base()
        pro = next(p for p in kb["plans"] if p["name"] == "Pro Plan")
        assert pro["price_monthly"] == 79

    def test_pro_plan_has_4k(self):
        kb = load_knowledge_base()
        pro = next(p for p in kb["plans"] if p["name"] == "Pro Plan")
        features_text = " ".join(pro["features"]).lower()
        assert "4k" in features_text

    def test_refund_policy_exists(self):
        kb = load_knowledge_base()
        policies_text = " ".join(p["details"] for p in kb["policies"]).lower()
        assert "7 days" in policies_text

    def test_support_policy_pro_only(self):
        kb = load_knowledge_base()
        support = next(p for p in kb["policies"] if "Support" in p["policy"])
        assert "Pro" in support["details"]


# ─────────────────────────────────────────────────────────────
# Tests: RAG Pipeline
# ─────────────────────────────────────────────────────────────

class TestRAGPipeline:
    def test_pricing_query_returns_plan_info(self):
        context = retrieve_context("What is the price of your plans?")
        assert "29" in context or "79" in context

    def test_pro_query_returns_pro_details(self):
        context = retrieve_context("Tell me about the Pro plan features")
        assert "Pro" in context

    def test_refund_query_returns_policy(self):
        context = retrieve_context("What is your refund policy?")
        assert "7 days" in context or "refund" in context.lower()

    def test_greeting_returns_company_info(self):
        context = retrieve_context("Hi, tell me about AutoStream")
        assert "AutoStream" in context

    def test_unknown_query_returns_fallback(self):
        context = retrieve_context("xyzzy foobar nonsense")
        # Should still return something meaningful
        assert len(context) > 10


# ─────────────────────────────────────────────────────────────
# Tests: Lead Capture Tool
# ─────────────────────────────────────────────────────────────

class TestLeadCapture:
    def test_successful_capture(self, capsys):
        result = mock_lead_capture(
            name="Priya Sharma",
            email="priya@example.com",
            platform="YouTube",
        )
        assert result["status"] == "success"
        assert result["lead"]["name"] == "Priya Sharma"
        assert result["lead"]["email"] == "priya@example.com"
        assert result["lead"]["platform"] == "YouTube"

    def test_print_output(self, capsys):
        mock_lead_capture("Alex Test", "alex@test.com", "Instagram")
        captured = capsys.readouterr()
        assert "Alex Test" in captured.out
        assert "alex@test.com" in captured.out
        assert "Instagram" in captured.out

    def test_invalid_email_raises(self):
        with pytest.raises(ValueError, match="Invalid email"):
            mock_lead_capture("Test User", "not-an-email", "YouTube")

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="required"):
            mock_lead_capture("", "test@example.com", "YouTube")

    def test_missing_email_raises(self):
        with pytest.raises(ValueError, match="required"):
            mock_lead_capture("Test User", "", "YouTube")

    def test_missing_platform_raises(self):
        with pytest.raises(ValueError, match="required"):
            mock_lead_capture("Test User", "test@example.com", "")

    def test_result_has_timestamp(self):
        result = mock_lead_capture("Demo User", "demo@example.com", "TikTok")
        assert "captured_at" in result["lead"]

    def test_result_has_source(self):
        result = mock_lead_capture("Demo User", "demo@example.com", "TikTok")
        assert "AutoStream" in result["lead"]["source"]


# ─────────────────────────────────────────────────────────────
# Tests: Agent State helpers (no LLM needed)
# ─────────────────────────────────────────────────────────────

class TestStateHelpers:
    def test_import_state(self):
        pytest.importorskip("langgraph", reason="langgraph not installed")
        from agent.state import AgentState, LeadInfo
        state: AgentState = {
            "messages": [],
            "intent": "unknown",
            "lead_info": {},
            "lead_captured": False,
            "rag_context": "",
            "awaiting_field": None,
        }
        assert state["lead_captured"] is False

    def test_lead_info_partial(self):
        pytest.importorskip("langgraph", reason="langgraph not installed")
        from agent.state import LeadInfo
        info: LeadInfo = {"name": "Priya"}
        assert info.get("email") is None
        assert info.get("name") == "Priya"
