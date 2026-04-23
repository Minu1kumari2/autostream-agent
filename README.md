# AutoStream — Social-to-Lead Agentic Workflow

> **ServiceHive / Inflx — Machine Learning Intern Assignment**

A production-grade conversational AI agent that converts social media conversations into qualified business leads, built with **LangGraph**, **Claude 3 Haiku**, and a local RAG pipeline.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [How to Run Locally](#how-to-run-locally)
5. [Project Structure](#project-structure)
6. [Example Conversation](#example-conversation)
7. [Running Tests](#running-tests)
8. [WhatsApp Deployment via Webhooks](#whatsapp-deployment-via-webhooks)

---

## Project Overview

AutoStream is a fictional SaaS product providing automated video-editing tools for content creators. This agent acts as **Alex**, an AI sales assistant that:

- Greets users naturally
- Answers product and pricing questions using a local knowledge base (RAG)
- Detects when a user is ready to sign up (high-intent detection)
- Collects name, email, and creator platform — one field at a time
- Fires a `mock_lead_capture()` API call only once all three values are confirmed

---

## Architecture

### Why LangGraph?

LangGraph was chosen over vanilla LangChain chains or AutoGen for three reasons:

1. **Explicit state machine** — The agent's behaviour at each turn depends on accumulated state (which lead fields have been collected, what the current intent is, whether a lead has already been captured). LangGraph's `StateGraph` makes these transitions transparent and debuggable, unlike opaque chain pipelines.

2. **Fine-grained control over tool execution** — The assignment requires that `mock_lead_capture()` is *never* called prematurely. LangGraph's conditional edges make it easy to gate the tool call behind a strict "all three fields present" check, rather than relying on an LLM to decide when to fire a tool.

3. **Built-in memory across turns** — The `add_messages` reducer on the `messages` field means conversation history accumulates automatically across 5–6 (or more) turns without any manual buffer management.

### Graph Topology

```
START
  │
  ▼
[retrieve_context]    ← RAG: fetch KB snippets relevant to user message
  │
  ▼
[classify_intent]     ← LLM call: label as greeting / inquiry / high_intent_lead
  │
  ├── casual_greeting / product_inquiry / unknown
  │         └──► [generate_response]   ← LLM answers using RAG context
  │
  └── high_intent_lead
            ├── awaiting_field (user just answered a question)
            │         └──► [extract_field] ──► [generate_response]
            └── not awaiting
                      └──► [generate_response]  (asks for next missing field,
                                                  or fires tool if all collected)
  │
  ▼
 END
```

### State Management

The `AgentState` TypedDict carries:

| Field | Type | Purpose |
|-------|------|---------|
| `messages` | `list` (append-only) | Full conversation history |
| `intent` | `str` | Latest classified intent |
| `lead_info` | `dict` | Collected name / email / platform |
| `lead_captured` | `bool` | Whether the tool has fired |
| `rag_context` | `str` | KB snippets for the current turn |
| `awaiting_field` | `str \| None` | Which field the agent is waiting for |

State persists across all turns in a single `dict` that is threaded through every `graph.invoke()` call.

### RAG Pipeline

The local knowledge base is stored in `knowledge_base/autostream_kb.json`. The `retrieve_context()` function tokenises the user query and scores it against four KB sections (company info, plans, policies, FAQs) using keyword overlap. The top-matching sections are concatenated and injected into the system prompt before every LLM call — no vector database required.

---

## Features

- ✅ **Intent Detection** — 3-class classification (greeting / inquiry / high-intent) via LLM
- ✅ **RAG Knowledge Retrieval** — local JSON knowledge base, zero external dependencies
- ✅ **Multi-turn Memory** — 5–6+ turn context window via LangGraph state
- ✅ **Lead Capture Tool** — fires only after all three fields are collected
- ✅ **Graceful field extraction** — regex-based fast path for email; LLM fallback for ambiguous input
- ✅ **Demo mode** — scripted conversation for recording the submission video

---

## How to Run Locally

### Prerequisites

- Python 3.9 or higher
- An Anthropic API key (Claude 3 Haiku)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/autostream-agent.git
cd autostream-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # macOS / Linux
set ANTHROPIC_API_KEY=sk-ant-...          # Windows CMD
$env:ANTHROPIC_API_KEY="sk-ant-..."       # Windows PowerShell
```

Alternatively, create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

And add this line at the top of `main.py`:

```python
from dotenv import load_dotenv; load_dotenv()
```

### 5. Run the agent

**Interactive mode** (type your own messages):

```bash
python main.py
```

**Demo / scripted mode** (great for the submission video):

```bash
python main.py --demo
```

---

## Project Structure

```
autostream-agent/
├── main.py                          # CLI entry point (interactive + demo modes)
├── requirements.txt
├── README.md
│
├── knowledge_base/
│   └── autostream_kb.json           # Local RAG knowledge base
│
├── agent/
│   ├── __init__.py
│   ├── state.py                     # AgentState TypedDict definition
│   ├── graph.py                     # LangGraph nodes, edges, routing logic
│   └── rag_pipeline.py              # KB loading + context retrieval
│
├── tools/
│   ├── __init__.py
│   └── lead_capture.py              # mock_lead_capture() tool function
│
└── tests/
    ├── __init__.py
    └── test_agent.py                # pytest unit tests (no LLM required)
```

---

## Example Conversation

```
You: Hi, tell me about your pricing.

Alex: AutoStream has two plans:
• Basic ($29/month): 10 videos/month, 720p, email support.
• Pro ($79/month): Unlimited videos, 4K, AI captions, and 24/7 support.
Would you like more details on either plan?

You: What's the refund policy?

Alex: AutoStream does not issue refunds after 7 days from the purchase date.
Refund requests must be made within the first 7 days.

You: That sounds good. I want to sign up for Pro for my YouTube channel.

Alex: I'd love to get you signed up! Could you share your full name first?

You: Priya Sharma

Alex: Great, Priya Sharma! What's the best email address to reach you at?

You: priya.sharma@example.com

Alex: Almost there! Which creator platform do you mainly use?
(e.g. YouTube, Instagram, TikTok…)

You: YouTube

[Lead captured successfully: Priya Sharma, priya.sharma@example.com, YouTube]

Alex: 🎉 You're all set, Priya Sharma! I've captured your details and our team
will reach out to priya.sharma@example.com shortly to get you started on
AutoStream Pro. Welcome aboard — can't wait to see what you create on YouTube!
```

---

## Running Tests

The test suite covers the knowledge base, RAG pipeline, and lead capture tool — **no LLM calls required**.

```bash
pip install pytest
python -m pytest tests/ -v
```

Expected output:

```
tests/test_agent.py::TestKnowledgeBase::test_kb_loads            PASSED
tests/test_agent.py::TestKnowledgeBase::test_basic_plan_price    PASSED
...
tests/test_agent.py::TestLeadCapture::test_successful_capture    PASSED
tests/test_agent.py::TestLeadCapture::test_invalid_email_raises  PASSED
...
========================= 18 passed in 0.12s =========================
```

---

## WhatsApp Deployment via Webhooks

Integrating this agent with WhatsApp Business API requires three components:

### 1. Register a Webhook Endpoint

Deploy a FastAPI (or Flask) server that exposes a `POST /webhook` route. WhatsApp will POST every inbound message to this URL.

```python
# app.py (FastAPI example)
from fastapi import FastAPI, Request
import httpx

app = FastAPI()
sessions = {}   # in-memory session store; use Redis in production

@app.post("/webhook")
async def handle_whatsapp(request: Request):
    body = await request.json()
    message = body["entry"][0]["changes"][0]["value"]["messages"][0]
    sender   = message["from"]        # WhatsApp phone number
    text     = message["text"]["body"]

    # Restore or initialise session state
    state = sessions.get(sender) or new_state()
    state["messages"].append(HumanMessage(content=text))

    # Run the LangGraph agent
    result = graph.invoke(state)
    sessions[sender] = result          # persist updated state

    # Extract assistant reply and send via WhatsApp Cloud API
    reply = get_last_ai_message(result)
    await send_whatsapp_message(sender, reply)
    return {"status": "ok"}
```

### 2. Send Replies via the Cloud API

```python
async def send_whatsapp_message(to: str, body: str):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload, headers=headers)
```

### 3. Verify the Webhook (one-time setup)

WhatsApp sends a `GET /webhook?hub.verify_token=...&hub.challenge=...` to verify ownership. Return the `hub.challenge` value to confirm.

```python
@app.get("/webhook")
def verify(hub_mode: str, hub_verify_token: str, hub_challenge: str):
    if hub_verify_token == os.getenv("VERIFY_TOKEN"):
        return int(hub_challenge)
    return {"error": "Invalid token"}, 403
```

### Key Considerations

| Concern | Solution |
|---------|----------|
| Session persistence | Replace `dict` with Redis (`redis-py`) keyed by phone number |
| Scaling | Deploy on Render / Railway / AWS Lambda + API Gateway |
| Rate limits | WhatsApp limits ~80 messages/second per number |
| Media messages | Check `message["type"]` — handle `image`, `audio` separately |
| Security | Validate the `X-Hub-Signature-256` header on every POST |

---

*Built for the ServiceHive / Inflx Machine Learning Intern assignment.*
