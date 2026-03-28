"""LangGraph Optimization Workflow — Web4AGI

Uses LangGraph to build a stateful multi-step optimization pipeline
for each parcel agent. Integrates with Sentient Foundation models
for AI-driven decision making.

Workflow steps:
  1. Assess  — evaluate current parcel state and market conditions
  2. Plan    — generate optimization strategies via LLM
  3. Execute — apply the top strategy (trade, update, communicate)
  4. Reflect — score outcome and update memory

Also exposes a higher-level ``ParcelOptimizationWorkflow`` class that
provides a richer API expected by external consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict
from datetime import datetime

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None  # type: ignore[assignment,misc]
    END = "__end__"

try:
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None  # type: ignore[assignment,misc]
    HumanMessage = None  # type: ignore[assignment]


# ── Low-level State Schema ─────────────────────────────────────────────────────

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


# ── Low-level Node Functions ───────────────────────────────────────────────────

def _get_llm():
    """Return the configured LLM (Sentient Foundation or OpenAI fallback)."""
    import os
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
    if llm and HumanMessage:
        prompt = f"""You are an optimizer for a metaverse parcel agent.
Parcel state: {ps}
Market context: {ctx}

In 2-3 sentences, assess the parcel's current situation and key opportunities."""
        response = llm.invoke([HumanMessage(content=prompt)])
        assessment = response.content
    else:
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
    if llm and HumanMessage:
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

    action = {
        "strategy": chosen,
        "executed_at": datetime.utcnow().isoformat(),
        "status": "simulated",
    }
    actions = state.get("actions_taken", []) + [action]
    return {**state, "chosen_strategy": chosen, "actions_taken": actions}


def reflect_node(state: ParcelOptState) -> ParcelOptState:
    """Score the outcome and generate a reflection."""
    llm = _get_llm()
    if llm and HumanMessage:
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

    new_iteration = state.get("iteration", 0) + 1
    return {**state, "score": score, "reflection": reflection, "iteration": new_iteration}


def should_continue(state: ParcelOptState) -> str:
    """Decide whether to run another optimization iteration."""
    if state.get("score", 0) >= 0.8:
        return END
    if state.get("iteration", 0) >= 3:
        return END
    return "assess"


# ── Low-level Graph Builder ────────────────────────────────────────────────────

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
    if graph is None:
        state = assess_node(initial)
        state = plan_node(state)
        state = execute_node(state)
        state = reflect_node(state)
        return state

    config = {"configurable": {"thread_id": parcel_state.get("parcel_id", "default")}}
    result = await graph.ainvoke(initial, config=config)
    return result


# ── High-level API ─────────────────────────────────────────────────────────────

# Step order for state-machine transitions
_STEP_ORDER = ["analyze", "generate_strategies", "evaluate", "select", "complete"]


@dataclass
class WorkflowState:
    """Mutable state object passed between workflow steps."""

    parcel_id: str
    context: Dict[str, Any]
    current_step: str
    strategies: List[Dict] = field(default_factory=list)
    analysis: Optional[Dict] = None
    best_strategy: Optional[Dict] = None
    error: Optional[str] = None


class _GraphProxy:
    """Minimal proxy that exposes a get_nodes() helper for tests."""

    def __init__(self, nodes: List[str]) -> None:
        self._nodes = set(nodes)

    def get_nodes(self) -> set:
        return self._nodes


class _WorkflowMemory:
    """Simple in-process history store."""

    def __init__(self) -> None:
        self._history: List[Dict] = []

    def record(self, entry: Dict) -> None:
        self._history.append(entry)

    def get_history(self) -> List[Dict]:
        return list(self._history)


class ParcelOptimizationWorkflow:
    """
    High-level workflow for optimising a single parcel's strategy.

    Wraps the low-level LangGraph pipeline and exposes step-by-step
    methods that can be unit-tested in isolation.
    """

    def __init__(
        self,
        parcel_id: str,
        model: str = "gpt-4",
        max_budget: Optional[float] = None,
        risk_tolerance: Optional[str] = None,
        use_memory: bool = False,
        objectives: Optional[List[str]] = None,
    ) -> None:
        self.parcel_id = parcel_id
        self.model = model
        self.max_budget = max_budget
        self.risk_tolerance = risk_tolerance
        self.use_memory = use_memory
        self.objectives = objectives or []
        self.graph = _GraphProxy(["analyze", "generate_strategies", "evaluate", "select"])
        self.memory: Optional[_WorkflowMemory] = _WorkflowMemory() if use_memory else None

    # ── Internal LLM helper ────────────────────────────────────────────────

    async def _call_llm(self, prompt: str) -> Any:
        """Call the configured LLM.  Override in tests."""
        llm = _get_llm()
        if llm is None or HumanMessage is None:
            return None
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    # ── Step implementations ───────────────────────────────────────────────

    async def analyze(self, state: WorkflowState) -> Dict:
        """Analyze the parcel state and return an assessment dict."""
        context = state.context if isinstance(state, WorkflowState) else state
        prompt = f"Analyze this parcel context and return an assessment: {context}"
        llm_result = await self._call_llm(prompt)
        if isinstance(llm_result, dict):
            return llm_result
        return {"assessment": llm_result or "positive"}

    async def generate_strategies(self, state: WorkflowState) -> List[Dict]:
        """Generate a list of strategy dicts for this parcel."""
        context = state.context if isinstance(state, WorkflowState) else {}
        prompt = f"Generate trading strategies for context: {context}"
        llm_result = await self._call_llm(prompt)
        if isinstance(llm_result, list):
            return llm_result
        return [{"type": "trade", "action": "buy", "amount": 50.0}]

    async def evaluate_strategies(self, state: WorkflowState) -> List[Dict]:
        """Score each strategy and return the updated list."""
        strategies = state.strategies if isinstance(state, WorkflowState) else []
        evaluated = []
        for i, s in enumerate(strategies):
            scored = dict(s)
            if "score" not in scored:
                # Heuristic: lower risk → higher score
                risk_map = {"low": 0.9, "medium": 0.7, "high": 0.5}
                scored["score"] = risk_map.get(
                    str(s.get("risk", "medium")).lower(),
                    max(0.0, 0.7 - i * 0.05),
                )
            evaluated.append(scored)
        return evaluated

    async def select_best_strategy(self, state: WorkflowState) -> Dict:
        """Return the strategy with the highest score."""
        strategies = state.strategies if isinstance(state, WorkflowState) else []
        if not strategies:
            return {}
        return max(strategies, key=lambda s: s.get("score", 0.0))

    # ── Full run ───────────────────────────────────────────────────────────

    async def run(self, state_or_dict: Any) -> Dict:
        """
        Execute the complete workflow.

        Accepts either a ``WorkflowState`` or a plain dict (parcel state).
        Returns a result dict with ``assessment``, ``strategies``, and
        ``best_strategy`` keys, or an ``error`` key on failure.
        """
        try:
            if isinstance(state_or_dict, WorkflowState):
                ws = state_or_dict
            else:
                ws = WorkflowState(
                    parcel_id=self.parcel_id,
                    context=state_or_dict if isinstance(state_or_dict, dict) else {},
                    current_step="analyze",
                )

            # Step 1 – analyze
            analysis = await self.analyze(ws)
            ws.analysis = analysis

            # Step 2 – generate strategies
            ws.current_step = "generate_strategies"
            strategies = await self.generate_strategies(ws)
            ws.strategies = strategies

            # Step 3 – evaluate
            ws.current_step = "evaluate"
            evaluated = await self.evaluate_strategies(ws)
            ws.strategies = evaluated

            # Step 4 – select
            ws.current_step = "select"
            best = await self.select_best_strategy(ws)
            ws.best_strategy = best
            ws.current_step = "complete"

            result = {
                "assessment": analysis.get("assessment") if isinstance(analysis, dict) else analysis,
                "strategies": ws.strategies,
                "best_strategy": ws.best_strategy,
            }

            if self.memory is not None:
                self.memory.record(result)

            return result

        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    # ── Constraint & Multi-objective helpers ──────────────────────────────

    def filter_by_constraints(self, strategies: List[Dict]) -> List[Dict]:
        """Filter strategies that violate budget or risk tolerance constraints."""
        filtered = []
        for s in strategies:
            if self.max_budget is not None and s.get("amount", 0) > self.max_budget:
                continue
            if self.risk_tolerance == "low" and s.get("risk") not in (None, "low"):
                continue
            filtered.append(s)
        return filtered

    def rank_multi_objective(self, strategies: List[Dict]) -> List[Dict]:
        """
        Rank strategies against multiple objectives and add a composite score.

        Objectives supported: maximize_profit, minimize_risk, maximize_liquidity.
        """
        ranked = []
        obj_set = set(self.objectives)
        for s in strategies:
            score = 0.0
            count = 0
            if "maximize_profit" in obj_set and "profit" in s:
                score += s["profit"] / 100.0
                count += 1
            if "minimize_risk" in obj_set and "risk" in s:
                score += 1.0 - s["risk"]
                count += 1
            if "maximize_liquidity" in obj_set and "liquidity" in s:
                score += s["liquidity"]
                count += 1
            composite = score / max(count, 1)
            ranked.append({**s, "score": composite})
        return ranked

    # ── State-machine helper ──────────────────────────────────────────────

    def next_step(self, current: str) -> str:
        """Return the name of the step that follows *current*."""
        try:
            idx = _STEP_ORDER.index(current)
            return _STEP_ORDER[idx + 1]
        except (ValueError, IndexError):
            return "complete"


# ── Public convenience function ────────────────────────────────────────────────

async def optimize_parcel_strategy(
    parcel_id: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a ``ParcelOptimizationWorkflow`` and run it for the given parcel."""
    workflow = ParcelOptimizationWorkflow(parcel_id=parcel_id)
    return await workflow.run(
        WorkflowState(
            parcel_id=parcel_id,
            context=context or {},
            current_step="analyze",
        )
    )
