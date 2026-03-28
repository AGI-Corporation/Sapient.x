"""Tests for the LangGraph optimization workflow (langgraph_workflow.py)."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.graphs.langgraph_workflow import (
    ParcelOptState,
    assess_node,
    plan_node,
    execute_node,
    reflect_node,
    should_continue,
    build_optimization_graph,
    run_parcel_optimization,
    LANGGRAPH_AVAILABLE,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _base_state(balance: float = 100.0) -> ParcelOptState:
    return {
        "parcel_state": {
            "parcel_id": "test-parcel-001",
            "owner": "0xOwner",
            "location": {"lat": 37.7749, "lng": -122.4194, "alt": 0.0},
            "balance_usdx": balance,
            "metadata": {},
            "active": True,
            "last_updated": "2026-01-01T00:00:00",
        },
        "context": {},
        "assessment": None,
        "strategies": [],
        "chosen_strategy": None,
        "actions_taken": [],
        "reflection": None,
        "score": 0.0,
        "iteration": 0,
    }


# ── assess_node Tests ──────────────────────────────────────────────────────────

def test_assess_node_fallback_no_llm():
    """assess_node produces a heuristic assessment when no LLM is configured."""
    state = _base_state()
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
        result = assess_node(state)
    assert result["assessment"] is not None
    assert "100.00 USDx" in result["assessment"]


def test_assess_node_with_llm():
    """assess_node uses LLM when one is configured."""
    state = _base_state()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Parcel looks promising.")
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=mock_llm):
        result = assess_node(state)
    assert result["assessment"] == "Parcel looks promising."
    mock_llm.invoke.assert_called_once()


def test_assess_node_preserves_other_state_keys():
    """assess_node does not discard unrelated state keys."""
    state = _base_state()
    state["score"] = 0.42
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
        result = assess_node(state)
    assert result["score"] == 0.42


def test_assess_node_zero_balance():
    """assess_node handles zero-balance parcel correctly."""
    state = _base_state(balance=0.0)
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
        result = assess_node(state)
    assert "0.00 USDx" in result["assessment"]


# ── plan_node Tests ────────────────────────────────────────────────────────────

def test_plan_node_fallback_no_llm():
    """plan_node generates heuristic strategies when no LLM configured."""
    state = _base_state()
    state["assessment"] = "Balance looks healthy."
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
        result = plan_node(state)
    assert len(result["strategies"]) == 3
    assert all(isinstance(s, str) for s in result["strategies"])


def test_plan_node_with_llm():
    """plan_node parses numbered list from LLM response."""
    state = _base_state()
    state["assessment"] = "Assessment text."
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="1. Trade 10 USDx\n2. Lease parcel\n3. Update metadata"
    )
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=mock_llm):
        result = plan_node(state)
    assert len(result["strategies"]) == 3
    assert "Trade 10 USDx" in result["strategies"][0]


def test_plan_node_with_llm_no_numbered_lines():
    """plan_node falls back to raw content when LLM returns no numbered lines."""
    state = _base_state()
    state["assessment"] = "A"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Just do something useful.")
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=mock_llm):
        result = plan_node(state)
    assert result["strategies"] == ["Just do something useful."]


def test_plan_node_includes_balance_in_fallback_strategy():
    """plan_node fallback strategy references the parcel's actual balance."""
    state = _base_state(balance=200.0)
    state["assessment"] = "Fine."
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
        result = plan_node(state)
    # First heuristic strategy includes 10% of balance = 20.0
    assert "20.00" in result["strategies"][0]


# ── execute_node Tests ─────────────────────────────────────────────────────────

def test_execute_node_picks_first_strategy():
    """execute_node selects the first strategy and records an action."""
    state = _base_state()
    state["strategies"] = ["Strategy A", "Strategy B"]
    result = execute_node(state)
    assert result["chosen_strategy"] == "Strategy A"
    assert len(result["actions_taken"]) == 1
    assert result["actions_taken"][0]["strategy"] == "Strategy A"
    assert result["actions_taken"][0]["status"] == "simulated"


def test_execute_node_no_strategies():
    """execute_node records 'No strategy available' when strategies is empty."""
    state = _base_state()
    state["strategies"] = []
    result = execute_node(state)
    assert result["chosen_strategy"] == "No strategy available"
    assert len(result["actions_taken"]) == 1


def test_execute_node_appends_to_existing_actions():
    """execute_node accumulates actions across iterations."""
    state = _base_state()
    state["strategies"] = ["New strategy"]
    state["actions_taken"] = [{"strategy": "Old strategy", "status": "simulated"}]
    result = execute_node(state)
    assert len(result["actions_taken"]) == 2


def test_execute_node_action_has_timestamp():
    """execute_node action dict contains an executed_at timestamp."""
    state = _base_state()
    state["strategies"] = ["Do something"]
    result = execute_node(state)
    assert "executed_at" in result["actions_taken"][0]


# ── reflect_node Tests ─────────────────────────────────────────────────────────

def test_reflect_node_fallback_with_strategy():
    """reflect_node scores 0.7 when a strategy was chosen (no LLM)."""
    state = _base_state()
    state["chosen_strategy"] = "Trade 10 USDx"
    state["actions_taken"] = [{"strategy": "Trade 10 USDx", "status": "simulated"}]
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
        result = reflect_node(state)
    assert result["score"] == 0.7
    assert result["reflection"] is not None


def test_reflect_node_fallback_no_strategy():
    """reflect_node scores 0.3 when no strategy was chosen."""
    state = _base_state()
    state["chosen_strategy"] = None
    state["actions_taken"] = []
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
        result = reflect_node(state)
    assert result["score"] == 0.3


def test_reflect_node_with_llm_parses_score():
    """reflect_node extracts score from LLM response format."""
    state = _base_state()
    state["chosen_strategy"] = "Trade"
    state["actions_taken"] = []
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="SCORE: 0.9 | REFLECTION: Great outcome achieved."
    )
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=mock_llm):
        result = reflect_node(state)
    assert result["score"] == pytest.approx(0.9)
    assert result["reflection"] == "Great outcome achieved."


def test_reflect_node_with_llm_invalid_score():
    """reflect_node falls back to 0.5 when SCORE parsing fails."""
    state = _base_state()
    state["chosen_strategy"] = "Trade"
    state["actions_taken"] = []
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="SCORE: not_a_number | REFLECTION: text")
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=mock_llm):
        result = reflect_node(state)
    assert result["score"] == 0.5


def test_reflect_node_reflection_includes_iteration():
    """reflect_node fallback reflection mentions the iteration count."""
    state = _base_state()
    state["chosen_strategy"] = "S"
    state["actions_taken"] = []
    state["iteration"] = 2
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
        result = reflect_node(state)
    assert "2" in result["reflection"]


# ── should_continue Tests ──────────────────────────────────────────────────────

def test_should_continue_high_score_stops():
    """should_continue returns END when score >= 0.8."""
    from src.graphs.langgraph_workflow import END

    state = _base_state()
    state["score"] = 0.85
    state["iteration"] = 1
    assert should_continue(state) == END


def test_should_continue_exact_threshold_stops():
    """should_continue returns END when score exactly equals 0.8."""
    from src.graphs.langgraph_workflow import END

    state = _base_state()
    state["score"] = 0.8
    state["iteration"] = 1
    assert should_continue(state) == END


def test_should_continue_max_iterations_stops():
    """should_continue returns END when iteration >= 3."""
    from src.graphs.langgraph_workflow import END

    state = _base_state()
    state["score"] = 0.2
    state["iteration"] = 3
    assert should_continue(state) == END


def test_should_continue_low_score_continues():
    """should_continue returns 'assess' when score is low and iterations remain."""
    state = _base_state()
    state["score"] = 0.4
    state["iteration"] = 1
    assert should_continue(state) == "assess"


def test_should_continue_zero_score_zero_iterations_continues():
    """should_continue continues from the initial state."""
    state = _base_state()
    state["score"] = 0.0
    state["iteration"] = 0
    assert should_continue(state) == "assess"


# ── build_optimization_graph Tests ────────────────────────────────────────────

@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
def test_build_optimization_graph_returns_compiled():
    """build_optimization_graph returns a compiled graph when LangGraph available."""
    graph = build_optimization_graph()
    assert graph is not None


def test_build_optimization_graph_without_langgraph():
    """build_optimization_graph returns None when LangGraph is unavailable."""
    with patch("src.graphs.langgraph_workflow.LANGGRAPH_AVAILABLE", False):
        graph = build_optimization_graph()
    assert graph is None


# ── run_parcel_optimization Tests ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_parcel_optimization_fallback(sample_parcel_state):
    """run_parcel_optimization falls back to direct node calls without LangGraph."""
    with (
        patch("src.graphs.langgraph_workflow._get_graph", return_value=None),
        patch("src.graphs.langgraph_workflow._get_llm", return_value=None),
    ):
        result = await run_parcel_optimization(parcel_state=sample_parcel_state)

    assert "assessment" in result
    assert "strategies" in result
    assert isinstance(result["strategies"], list)
    assert "chosen_strategy" in result
    assert "reflection" in result
    assert "score" in result


@pytest.mark.asyncio
async def test_run_parcel_optimization_with_context(sample_parcel_state):
    """run_parcel_optimization accepts and passes through context."""
    with (
        patch("src.graphs.langgraph_workflow._get_graph", return_value=None),
        patch("src.graphs.langgraph_workflow._get_llm", return_value=None),
    ):
        result = await run_parcel_optimization(
            parcel_state=sample_parcel_state,
            context={"market": "bearish"},
        )
    assert result is not None


@pytest.mark.asyncio
async def test_run_parcel_optimization_none_context(sample_parcel_state):
    """run_parcel_optimization handles context=None without error."""
    with (
        patch("src.graphs.langgraph_workflow._get_graph", return_value=None),
        patch("src.graphs.langgraph_workflow._get_llm", return_value=None),
    ):
        result = await run_parcel_optimization(parcel_state=sample_parcel_state, context=None)
    assert result is not None


@pytest.mark.asyncio
async def test_run_parcel_optimization_fallback_produces_score(sample_parcel_state):
    """run_parcel_optimization fallback path produces a numeric score."""
    with (
        patch("src.graphs.langgraph_workflow._get_graph", return_value=None),
        patch("src.graphs.langgraph_workflow._get_llm", return_value=None),
    ):
        result = await run_parcel_optimization(parcel_state=sample_parcel_state)
    assert isinstance(result["score"], float)


@pytest.mark.asyncio
@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
async def test_run_parcel_optimization_with_graph(sample_parcel_state):
    """run_parcel_optimization uses LangGraph graph when available."""
    expected = {
        **_base_state(),
        "assessment": "Good",
        "strategies": ["S1"],
        "score": 0.9,
    }
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=expected)

    with patch("src.graphs.langgraph_workflow._get_graph", return_value=mock_graph):
        result = await run_parcel_optimization(parcel_state=sample_parcel_state)

    assert result["assessment"] == "Good"
    mock_graph.ainvoke.assert_called_once()


# ── Coverage gap fixes ────────────────────────────────────────────────────────

def test_get_llm_sentient_key_branch(monkeypatch):
    """_get_llm returns a ChatOpenAI-like object for SENTIENT_API_KEY (line 62)."""
    import src.graphs.langgraph_workflow as wf
    monkeypatch.setenv("SENTIENT_API_KEY", "test-sentient-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    mock_chat_cls = MagicMock(return_value=MagicMock())
    with (
        patch.object(wf, "LANGCHAIN_AVAILABLE", True),
        patch.object(wf, "ChatOpenAI", mock_chat_cls),
    ):
        result = wf._get_llm()

    assert result is not None
    mock_chat_cls.assert_called_once()
    call_kwargs = mock_chat_cls.call_args.kwargs
    assert call_kwargs["api_key"] == "test-sentient-key"


def test_get_llm_openai_key_branch(monkeypatch):
    """_get_llm returns a ChatOpenAI-like object for OPENAI_API_KEY (line 69)."""
    import src.graphs.langgraph_workflow as wf
    monkeypatch.delenv("SENTIENT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    mock_chat_cls = MagicMock(return_value=MagicMock())
    with (
        patch.object(wf, "LANGCHAIN_AVAILABLE", True),
        patch.object(wf, "ChatOpenAI", mock_chat_cls),
    ):
        result = wf._get_llm()

    assert result is not None
    mock_chat_cls.assert_called_once_with(model="gpt-4o-mini", temperature=0.3)


def test_get_graph_function_body():
    """_get_graph() caches the graph in _GRAPH (covers lines 203-205)."""
    import src.graphs.langgraph_workflow as wf

    saved = wf._GRAPH
    try:
        # Reset so the if-branch executes
        wf._GRAPH = None
        graph1 = wf._get_graph()
        # The graph should now be cached
        assert wf._GRAPH is not None
        # Second call should return the same cached object without rebuilding
        graph2 = wf._get_graph()
        assert graph1 is graph2
    finally:
        wf._GRAPH = saved
