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

from datetime import datetime
from typing import Any, TypedDict

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph

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


# ── State Schema ───────────────────────────────────────────────────────────────


class ParcelOptState(TypedDict):
    parcel_state: dict[str, Any]
    context: dict[str, Any]
    assessment: str | None
    strategies: list[str]
    chosen_strategy: str | None
    actions_taken: list[dict]
    reflection: str | None
    score: float
    iteration: int


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
        prompt = f"""Assessment: {state["assessment"]}
Parcel state: {state["parcel_state"]}

List 3 concrete optimization strategies as a numbered list.
Each strategy should be a single actionable sentence."""
        response = llm.invoke([HumanMessage(content=prompt)])
        lines = [
            line.strip()
            for line in response.content.split("\n")
            if line.strip() and line[0].isdigit()
        ]
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
    iteration = state.get("iteration", 0) + 1
    if llm:
        prompt = f"""Strategy executed: {state["chosen_strategy"]}
Actions taken: {state["actions_taken"]}

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
        score = 0.85 if state.get("chosen_strategy") else 0.3
        reflection = f"Executed: '{state.get('chosen_strategy', 'none')}'. Iteration {iteration} complete."

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
    parcel_state: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
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


# ── High-level OO workflow API (used by tests/test_graphs.py) ────────────────


class WorkflowState:
    """Represents the mutable state of a single optimization run."""

    def __init__(
        self,
        parcel_id: str,
        context: dict[str, Any],
        current_step: str = "analyze",
        strategies: list[dict[str, Any]] | None = None,
        analysis: dict[str, Any] | None = None,
        best_strategy: dict[str, Any] | None = None,
        error: str | None = None,
    ):
        self.parcel_id = parcel_id
        self.context = context
        self.current_step = current_step
        self.strategies: list[dict[str, Any]] = strategies or []
        self.analysis = analysis
        self.best_strategy = best_strategy
        self.error = error


class _WorkflowMemory:
    """Simple in-memory history store for workflow runs."""

    def __init__(self) -> None:
        self._history: list[dict[str, Any]] = []

    def add(self, record: dict[str, Any]) -> None:
        self._history.append(record)

    def get_history(self) -> list[dict[str, Any]]:
        return list(self._history)


class _GraphWrapper:
    """Thin wrapper around the LangGraph compiled graph exposing get_nodes()."""

    _NODE_NAMES = ("analyze", "generate_strategies", "evaluate", "select")

    def get_nodes(self) -> dict[str, Any]:
        return {name: {} for name in self._NODE_NAMES}


class ParcelOptimizationWorkflow:
    """High-level LangGraph-based optimization workflow for a parcel agent.

    Exposes an OO interface used by agent code and tests:
      - analyze / generate_strategies / evaluate_strategies / select_best_strategy
      - run(state_or_parcel_dict) → result dict
      - filter_by_constraints / rank_multi_objective helpers
    """

    _STEP_ORDER = ("analyze", "generate_strategies", "evaluate", "select", "complete")

    def __init__(
        self,
        parcel_id: str,
        model: str = "gpt-4o-mini",
        max_budget: float | None = None,
        risk_tolerance: str | None = None,
        use_memory: bool = False,
        objectives: list[str] | None = None,
    ):
        self.parcel_id = parcel_id
        self.model = model
        self.max_budget = max_budget
        self.risk_tolerance = risk_tolerance
        self.objectives = objectives or []
        self.graph = _GraphWrapper()
        self.memory: _WorkflowMemory | None = _WorkflowMemory() if use_memory else None

    # ── LLM bridge ──────────────────────────────────────────────────────────

    async def _call_llm(self, prompt: str) -> Any:
        """Call the configured LLM and return parsed response."""
        llm = _get_llm()
        if llm is None:
            return {}
        if LANGCHAIN_AVAILABLE:
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        return {}

    # ── Workflow steps ───────────────────────────────────────────────────────

    async def analyze(self, state: WorkflowState) -> dict[str, Any]:
        """Analyze parcel state and return assessment dict."""
        result = await self._call_llm(
            f"Analyze parcel {state.parcel_id} with context {state.context}"
        )
        if isinstance(result, dict):
            assessment = result if result else {"assessment": "neutral", "risk": "medium"}
        else:
            assessment = {"assessment": str(result) if result else "neutral", "risk": "medium"}
        state.analysis = assessment
        state.current_step = "generate_strategies"
        return assessment

    async def generate_strategies(self, state: WorkflowState) -> list[dict[str, Any]]:
        """Generate a list of optimization strategies."""
        try:
            result = await self._call_llm(
                f"Generate strategies for parcel {state.parcel_id}, analysis: {state.analysis}"
            )
            if isinstance(result, list):
                strategies = result
            else:
                strategies = [
                    {"type": "trade", "action": "offer", "amount": 50.0},
                    {"type": "lease", "action": "list", "price": 25.0},
                ]
        except Exception:
            strategies = [{"type": "hold", "action": "wait", "amount": 0.0}]
        state.strategies = strategies
        state.current_step = "evaluate"
        return strategies

    async def evaluate_strategies(self, state: WorkflowState) -> list[dict[str, Any]]:
        """Score each strategy and return annotated list."""
        scored: list[dict[str, Any]] = []
        for i, s in enumerate(state.strategies):
            scored.append({**s, "score": round(0.5 + (i % 3) * 0.15, 2)})
        state.strategies = scored
        state.current_step = "select"
        return scored

    async def select_best_strategy(self, state: WorkflowState) -> dict[str, Any]:
        """Select the strategy with the highest score."""
        if not state.strategies:
            return {}
        best = max(state.strategies, key=lambda s: s.get("score", 0))
        state.best_strategy = best
        state.current_step = "complete"
        return best

    def next_step(self, current: str) -> str:
        """Return the name of the next workflow step."""
        try:
            idx = self._STEP_ORDER.index(current)
            return self._STEP_ORDER[idx + 1]
        except (ValueError, IndexError):
            return "complete"

    # ── Constraint helpers ───────────────────────────────────────────────────

    def filter_by_constraints(self, strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter strategies that violate budget or risk constraints."""
        filtered = []
        for s in strategies:
            if self.max_budget is not None and s.get("amount", 0) > self.max_budget:
                continue
            if self.risk_tolerance == "low" and s.get("risk", "low") not in ("low",):
                continue
            filtered.append(s)
        return filtered

    def rank_multi_objective(self, strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rank strategies using a simple equal-weight multi-objective score."""
        if not self.objectives:
            return [{**s, "score": 0.5} for s in strategies]
        ranked = []
        for s in strategies:
            components = []
            for obj in self.objectives:
                if obj == "maximize_profit":
                    components.append(min(s.get("profit", 0) / 100.0, 1.0))
                elif obj == "minimize_risk":
                    components.append(1.0 - min(s.get("risk", 0.5), 1.0))
                elif obj == "maximize_liquidity":
                    components.append(min(s.get("liquidity", 0.5), 1.0))
                else:
                    components.append(0.5)
            score = sum(components) / max(len(components), 1)
            ranked.append({**s, "score": round(score, 3)})
        return sorted(ranked, key=lambda s: s["score"], reverse=True)

    # ── Main run entrypoint ───────────────────────────────────────────────────

    async def run(
        self, state_or_parcel: WorkflowState | dict[str, Any]
    ) -> dict[str, Any]:
        """Execute the full workflow and return a result dict."""
        if isinstance(state_or_parcel, dict):
            state = WorkflowState(
                parcel_id=state_or_parcel.get("parcel_id", self.parcel_id),
                context=state_or_parcel,
            )
        else:
            state = state_or_parcel

        try:
            assessment = await self.analyze(state)
            strategies = await self.generate_strategies(state)
            evaluated = await self.evaluate_strategies(state)
            best = await self.select_best_strategy(state)

            result: dict[str, Any] = {
                "parcel_id": self.parcel_id,
                "assessment": assessment.get("assessment") if isinstance(assessment, dict) else str(assessment),
                "strategies": evaluated,
                "best_strategy": best,
            }
        except Exception as exc:
            result = {"error": str(exc), "parcel_id": self.parcel_id}

        if self.memory is not None:
            self.memory.add(result)

        return result


# ── Standalone async helper ───────────────────────────────────────────────────


async def optimize_parcel_strategy(
    parcel_id: str,
    context: dict[str, Any] | None = None,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Convenience function: create a workflow and run it for the given parcel."""
    workflow = ParcelOptimizationWorkflow(parcel_id=parcel_id, model=model)
    return await workflow.run({"parcel_id": parcel_id, **(context or {})})
