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
    StateGraph = None
    END = "__end__"

try:
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None


# ── Low-level State Schema (used by the raw LangGraph pipeline) ─────────────────

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


# ── High-level Workflow Abstractions ──────────────────────────────────────────────

@dataclass
class WorkflowState:
    """High-level state object used by ParcelOptimizationWorkflow."""

    parcel_id: str
    context: Dict[str, Any]
    current_step: str
    strategies: List[Dict] = field(default_factory=list)
    analysis: Optional[Dict] = None


class WorkflowMemory:
    """Simple in-process memory store for workflow history."""

    def __init__(self) -> None:
        self._history: List[Dict] = []

    def add(self, entry: Dict) -> None:
        self._history.append(entry)

    def get_history(self) -> List[Dict]:
        return list(self._history)


class WorkflowGraph:
    """Lightweight graph descriptor for the optimization workflow."""

    _NODES = ["analyze", "generate_strategies", "evaluate", "select", "complete"]

    def get_nodes(self) -> Dict[str, Any]:
        return {node: {} for node in self._NODES}


class ParcelOptimizationWorkflow:
    """High-level parcel optimization workflow wrapping the LangGraph pipeline."""

    _STEP_ORDER = ["analyze", "generate_strategies", "evaluate", "select", "complete"]
    _RISK_ORDER = ["low", "medium", "high"]

    def __init__(
        self,
        parcel_id: str,
        model: str = "gpt-4",
        max_budget: Optional[float] = None,
        risk_tolerance: Optional[str] = None,
        objectives: Optional[List[str]] = None,
        use_memory: bool = False,
    ) -> None:
        self.parcel_id = parcel_id
        self.model = model
        self.max_budget = max_budget
        self.risk_tolerance = risk_tolerance
        self.objectives = objectives or []
        self.graph = WorkflowGraph()
        self.memory: Optional[WorkflowMemory] = WorkflowMemory() if use_memory else None

    # ── Internal LLM helper ──────────────────────────────────────────────────

    async def _call_llm(self, prompt: str) -> Any:
        """Call the configured LLM and return its response content."""
        llm = _get_llm()
        if llm is None:
            return None
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    # ── Workflow steps ────────────────────────────────────────────────────────

    async def analyze(self, state: WorkflowState) -> Dict:
        """Assess parcel state and market conditions via LLM or heuristic."""
        result = await self._call_llm(
            f"Analyze parcel {state.parcel_id}: {state.context}"
        )
        if result is None:
            return {"assessment": "neutral", "risk": "medium"}
        if isinstance(result, dict):
            return result
        return {"assessment": str(result)}

    async def generate_strategies(self, state: WorkflowState) -> List[Dict]:
        """Generate a list of actionable strategies via LLM or heuristic."""
        result = await self._call_llm(
            f"Generate strategies for parcel {state.parcel_id}"
        )
        if result is None:
            return [{"type": "hold", "action": "maintain_current_state"}]
        if isinstance(result, list):
            return result
        return [{"type": "general", "action": str(result)}]

    async def evaluate_strategies(self, state: WorkflowState) -> List[Dict]:
        """Score each strategy, falling back to 0.5 when LLM is unavailable."""
        scored: List[Dict] = []
        for strategy in state.strategies:
            try:
                llm_result = await self._call_llm(f"Evaluate strategy: {strategy}")
                if isinstance(llm_result, dict):
                    score = float(llm_result.get("score", 0.5))
                else:
                    score = 0.5
            except Exception:  # noqa: BLE001 — LLM call may fail with varied errors; use default score
                score = 0.5
            scored.append({**strategy, "score": score})
        return scored

    async def select_best_strategy(self, state: WorkflowState) -> Dict:
        """Return the strategy with the highest score."""
        if not state.strategies:
            return {}
        return max(state.strategies, key=lambda s: s.get("score", 0.0))

    def next_step(self, current: str) -> str:
        """Return the next step name in the linear workflow."""
        try:
            idx = self._STEP_ORDER.index(current)
            if idx < len(self._STEP_ORDER) - 1:
                return self._STEP_ORDER[idx + 1]
        except ValueError:
            pass
        return "complete"

    def filter_by_constraints(self, strategies: List[Dict]) -> List[Dict]:
        """Remove strategies that violate budget or risk-tolerance constraints."""
        result = []
        for s in strategies:
            if self.max_budget is not None and s.get("amount", 0) > self.max_budget:
                continue
            if self.risk_tolerance is not None and self.risk_tolerance in self._RISK_ORDER:
                allowed_idx = self._RISK_ORDER.index(self.risk_tolerance)
                s_risk = s.get("risk", "high")
                s_risk_idx = (
                    self._RISK_ORDER.index(s_risk)
                    if s_risk in self._RISK_ORDER
                    else len(self._RISK_ORDER)
                )
                if s_risk_idx > allowed_idx:
                    continue
            result.append(s)
        return result

    def rank_multi_objective(self, strategies: List[Dict]) -> List[Dict]:
        """Score strategies by combining multiple objectives with equal weights."""
        objectives = self.objectives or ["maximize_profit"]
        weight = 1.0 / len(objectives)
        ranked = []
        for s in strategies:
            score = 0.0
            for obj in objectives:
                if obj == "maximize_profit":
                    score += weight * min(s.get("profit", 0) / 100, 1.0)
                elif obj == "minimize_risk":
                    score += weight * (1.0 - min(s.get("risk", 0.5), 1.0))
                elif obj == "maximize_liquidity":
                    score += weight * min(s.get("liquidity", 0.5), 1.0)
            ranked.append({**s, "score": round(score, 3)})
        return ranked

    async def run(self, parcel_state: Any) -> Dict:
        """Execute the full optimization workflow and return the result."""
        try:
            if isinstance(parcel_state, dict):
                state = WorkflowState(
                    parcel_id=parcel_state.get("parcel_id", self.parcel_id),
                    context=parcel_state,
                    current_step="analyze",
                )
            else:
                state = parcel_state

            analysis = await self.analyze(state)

            gen_state = WorkflowState(
                parcel_id=state.parcel_id,
                context=state.context,
                current_step="generate_strategies",
                analysis=analysis,
            )
            strategies = await self.generate_strategies(gen_state)

            eval_state = WorkflowState(
                parcel_id=state.parcel_id,
                context=state.context,
                current_step="evaluate",
                strategies=strategies,
            )
            evaluated = await self.evaluate_strategies(eval_state)

            select_state = WorkflowState(
                parcel_id=state.parcel_id,
                context=state.context,
                current_step="select",
                strategies=evaluated,
            )
            best = await self.select_best_strategy(select_state)

            result: Dict = {
                "parcel_id": state.parcel_id,
                "assessment": analysis,
                "strategies": evaluated,
                "best_strategy": best,
            }

            if self.memory is not None:
                self.memory.add(result)

            return result
        except Exception as exc:  # noqa: BLE001 — propagate all workflow errors as error dict
            return {"error": str(exc)}


async def optimize_parcel_strategy(
    parcel_id: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the high-level optimization workflow for a parcel."""
    workflow = ParcelOptimizationWorkflow(parcel_id=parcel_id)
    return await workflow.run(context or {})


# ── Node Functions ─────────────────────────────────────────────────────────────

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
        "executed_at": datetime.utcnow().isoformat(),
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

    iteration = state.get("iteration", 0) + 1
    return {**state, "score": score, "reflection": reflection, "iteration": iteration}


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
    if graph is None:
        # Fallback: run nodes directly without LangGraph
        state = assess_node(initial)
        state = plan_node(state)
        state = execute_node(state)
        state = reflect_node(state)
        return state

    config = {"configurable": {"thread_id": parcel_state.get("parcel_id", "default")}}
    result = await graph.ainvoke(initial, config=config)
    return result
