"""LangGraph Optimization Workflow — Web4AGI

Uses LangGraph to build a stateful multi-step optimization pipeline
for each parcel agent. Integrates with Sentient Foundation models
for AI-driven decision making.

Workflow steps:
  1. Assess  — evaluate current parcel state and market conditions
  2. Plan    — generate optimization strategies via LLM
  3. Execute — apply the top strategy (trade, update, communicate)
  4. Reflect — score outcome and update memory
"""

import os
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime, UTC

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "__end__"

try:
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None


# ── State Schema ───────────────────────────────────────────────────────────────

class ParcelOptState(TypedDict):
    parcel_state: Dict[str, Any]
    context: Dict[str, Any]
    assessment: Optional[str]
    strategies: List[str]
    chosen_strategy: Optional[str]
    actions_taken: List[Dict]
    reflection: Optional[str]
    score: float
    iteration: int


# ── Node Functions ─────────────────────────────────────────────────────────────

def _get_llm():
    """Return the configured LLM (Sentient Foundation or OpenAI fallback)."""
    sentient_key = os.getenv("SENTIENT_API_KEY")
    sentient_url = os.getenv("SENTIENT_BASE_URL", "https://api.sentientfoundation.ai/v1")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not LANGCHAIN_AVAILABLE:
        return None

    if sentient_key:
        return ChatOpenAI(
            model=os.getenv("SENTIENT_MODEL", "sentient-70b"),
            api_key=sentient_key,
            base_url=sentient_url,
            temperature=0.3,
        )
    if openai_key:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    return None


def assess_node(state: ParcelOptState) -> ParcelOptState:
    """Assess the current parcel state and market conditions."""
    ps = state["parcel_state"]
    ctx = state["context"]

    llm = _get_llm()
    if llm:
        prompt = f"""You are an optimizer for a metaverse parcel agent.
Parcel state: {ps}
Market context: {ctx}

In 2-3 sentences, assess the parcel's current situation and key opportunities."""
        response = llm.invoke([HumanMessage(content=prompt)])
        assessment = response.content
    else:
        # Fallback heuristic assessment
        balance = ps.get("balance_usdx", 0)
        assessment = (
            f"Parcel {ps.get('parcel_id', 'unknown')} has {balance:.2f} USDx balance. "
            f"Location: {ps.get('location', {})}. "
            "Consider trading excess balance or leasing unused capacity."
        )

    return {**state, "assessment": assessment}


def plan_node(state: ParcelOptState) -> ParcelOptState:
    """Generate optimization strategies based on the assessment."""
    llm = _get_llm()
    if llm:
        prompt = f"""Assessment: {state['assessment']}
Parcel state: {state['parcel_state']}

List 3 concrete optimization strategies as a numbered list.
Each strategy should be a single actionable sentence."""
        response = llm.invoke([HumanMessage(content=prompt)])
        lines = [l.strip() for l in response.content.split("\n") if l.strip() and l[0].isdigit()]
        strategies = lines[:3] if lines else [response.content]
    else:
        ps = state["parcel_state"]
        balance = ps.get("balance_usdx", 0)
        strategies = [
            f"Transfer {balance * 0.1:.2f} USDx to neighboring parcels to build alliance",
            "Update parcel metadata to increase discoverability",
            "Open a 30-day lease offer at market rate",
        ]

    return {**state, "strategies": strategies}


def execute_node(state: ParcelOptState) -> ParcelOptState:
    """Choose and simulate executing the best strategy."""
    strategies = state.get("strategies", [])
    chosen = strategies[0] if strategies else "No strategy available"

    # In production, this would call actual agent methods.
    # Here we record the simulated action.
    action = {
        "strategy": chosen,
        "executed_at": datetime.now(UTC).isoformat(),
        "status": "simulated",
    }
    actions = state.get("actions_taken", []) + [action]
    return {**state, "chosen_strategy": chosen, "actions_taken": actions}


def reflect_node(state: ParcelOptState) -> ParcelOptState:
    """Score the outcome and generate a reflection."""
    llm = _get_llm()
    if llm:
        prompt = f"""Strategy executed: {state['chosen_strategy']}
Actions taken: {state['actions_taken']}

In 1-2 sentences, reflect on the outcome and assign a score from 0.0 to 1.0.
Respond in format: SCORE: 0.X | REFLECTION: <text>"""
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content
        score = 0.5
        reflection = text
        if "SCORE:" in text:
            try:
                score_part = text.split("SCORE:")[1].split("|")[0].strip()
                score = float(score_part)
            except (ValueError, IndexError):
                pass
            if "REFLECTION:" in text:
                reflection = text.split("REFLECTION:")[1].strip()
    else:
        score = 0.7 if state.get("chosen_strategy") else 0.3
        reflection = f"Executed: '{state.get('chosen_strategy', 'none')}'. Iteration {state.get('iteration', 0)} complete."

    return {**state, "score": score, "reflection": reflection}


def should_continue(state: ParcelOptState) -> str:
    """Decide whether to run another optimization iteration."""
    if state.get("score", 0) >= 0.8:
        return END
    if state.get("iteration", 0) >= 3:
        return END
    return "assess"


# ── Graph Builder ───────────────────────────────────────────────────────────────

def build_optimization_graph():
    """Build and compile the LangGraph optimization workflow."""
    if not LANGGRAPH_AVAILABLE:
        return None

    g = StateGraph(ParcelOptState)
    g.add_node("assess", assess_node)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_node("reflect", reflect_node)

    g.set_entry_point("assess")
    g.add_edge("assess", "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "reflect")
    g.add_conditional_edges("reflect", should_continue)

    return g.compile(checkpointer=MemorySaver())


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_optimization_graph()
    return _GRAPH


# ── Public Entry Point ───────────────────────────────────────────────────────────

async def run_parcel_optimization(
    parcel_state: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the optimization workflow for a parcel and return the final state."""
    initial: ParcelOptState = {
        "parcel_state": parcel_state,
        "context": context or {},
        "assessment": None,
        "strategies": [],
        "chosen_strategy": None,
        "actions_taken": [],
        "reflection": None,
        "score": 0.0,
        "iteration": 0,
    }

    graph = _get_graph()
    if graph is None or not LANGGRAPH_AVAILABLE:
        # Fallback: run nodes directly without LangGraph
        state = assess_node(initial)
        state = plan_node(state)
        state = execute_node(state)
        state = reflect_node(state)
        return state

    config = {"configurable": {"thread_id": parcel_state.get("parcel_id", "default")}}
    result = await graph.ainvoke(initial, config=config)
    return result
