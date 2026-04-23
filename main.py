"""
AutoStream Social-to-Lead Agent — CLI Runner.

Usage:
    python main.py

Set ANTHROPIC_API_KEY in your environment before running.
"""

import sys
from langchain_core.messages import HumanMessage
from agent.graph import get_graph
from agent.state import AgentState

WELCOME_BANNER = """
╔══════════════════════════════════════════════════════════╗
║          AutoStream — AI Sales Assistant (Alex)          ║
║     Social-to-Lead Agentic Workflow  |  ServiceHive      ║
╠══════════════════════════════════════════════════════════╣
║  Type your message and press Enter.                      ║
║  Type 'quit' or 'exit' to end the session.               ║
╚══════════════════════════════════════════════════════════╝
"""


def run_interactive_session() -> None:
    """Run a multi-turn CLI conversation with the AutoStream agent."""
    print(WELCOME_BANNER)

    graph = get_graph()

    # Initialise blank state
    state: AgentState = {
        "messages": [],
        "intent": "unknown",
        "lead_info": {},
        "lead_captured": False,
        "rag_context": "",
        "awaiting_field": None,
    }

    print("Alex: Hi there! 👋 I'm Alex, your AutoStream assistant. How can I help you today?\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAlex: Goodbye! 👋")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
            print("Alex: Thanks for chatting! Feel free to come back anytime. 👋")
            break

        # Append the user message to state
        state["messages"].append(HumanMessage(content=user_input))

        # Invoke the graph
        try:
            result = graph.invoke(state)
        except Exception as exc:
            print(f"Alex: Oops — something went wrong on my end. ({exc})")
            continue

        # Merge returned state
        state.update(result)

        # Print the latest assistant response
        for msg in reversed(state["messages"]):
            from langchain_core.messages import AIMessage
            if isinstance(msg, AIMessage):
                print(f"\nAlex: {msg.content}\n")
                break


def run_demo_script() -> None:
    """
    Run a scripted demo conversation to showcase all agent capabilities.
    Useful for screen recording / assignment demo video.
    """
    import time

    demo_turns = [
        "Hi! Tell me about your pricing.",
        "What's included in the Pro plan?",
        "Do you have a refund policy?",
        "That sounds great! I want to sign up for the Pro plan for my YouTube channel.",
        "My name is Priya Sharma",
        "priya.sharma@example.com",
        "YouTube",
    ]

    print(WELCOME_BANNER)
    print("[ DEMO MODE — scripted conversation ]\n")

    graph = get_graph()

    state: AgentState = {
        "messages": [],
        "intent": "unknown",
        "lead_info": {},
        "lead_captured": False,
        "rag_context": "",
        "awaiting_field": None,
    }

    print("Alex: Hi there! 👋 I'm Alex, your AutoStream assistant. How can I help you today?\n")
    time.sleep(1)

    for user_input in demo_turns:
        print(f"You: {user_input}")
        time.sleep(0.5)

        state["messages"].append(HumanMessage(content=user_input))

        result = graph.invoke(state)
        state.update(result)

        for msg in reversed(state["messages"]):
            from langchain_core.messages import AIMessage
            if isinstance(msg, AIMessage):
                print(f"\nAlex: {msg.content}\n")
                break

        time.sleep(1)

        if state.get("lead_captured"):
            print("\n[ ✅ Lead successfully captured — workflow complete ]\n")
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AutoStream AI Agent")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the scripted demo conversation instead of interactive mode",
    )
    args = parser.parse_args()

    if args.demo:
        run_demo_script()
    else:
        run_interactive_session()
