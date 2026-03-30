"""Unit tests for LangGraph workflow node functions."""

from unittest.mock import MagicMock, patch

import pytest

from src.graphs.langgraph_workflow import (
    END,
    ParcelOptState,
    assess_node,
    build_optimization_graph,
    execute_node,
    plan_node,
    reflect_node,
    run_parcel_optimization,
    should_continue,
)


def _make_state(**overrides) -> ParcelOptState:
    base: ParcelOptState = {
        "parcel_state": {
            "parcel_id": "p-001",
            "balance_usdx": 100.0,
            "location": {"lat": 37.7, "lng": -122.4},
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
    base.update(overrides)
    return base


# ── assess_node ────────────────────────────────────────────────────────────


class TestAssessNode:
    def test_assess_node_no_llm(self):
        state = _make_state()
        with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
            result = assess_node(state)
        assert result["assessment"] is not None
        assert "p-001" in result["assessment"]

    def test_assess_node_with_llm(self):
        state = _make_state()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="LLM assessment text")
        with patch("src.graphs.langgraph_workflow._get_llm", return_value=mock_llm):
            result = assess_node(state)
        assert result["assessment"] == "LLM assessment text"

    def test_assess_node_preserves_other_fields(self):
        state = _make_state(score=0.5, iteration=1)
        with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
            result = assess_node(state)
        assert result["score"] == 0.5
        assert result["iteration"] == 1


# ── plan_node ──────────────────────────────────────────────────────────────


class TestPlanNode:
    def test_plan_node_no_llm(self):
        state = _make_state(assessment="Test assessment")
        with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
            result = plan_node(state)
        assert isinstance(result["strategies"], list)
        assert len(result["strategies"]) == 3

    def test_plan_node_with_llm_numbered_list(self):
        state = _make_state(assessment="good")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="1. Strategy one\n2. Strategy two\n3. Strategy three"
        )
        with patch("src.graphs.langgraph_workflow._get_llm", return_value=mock_llm):
            result = plan_node(state)
        assert len(result["strategies"]) == 3
        assert "Strategy one" in result["strategies"][0]

    def test_plan_node_with_llm_no_numbered_lines(self):
        state = _make_state(assessment="good")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Just do something great")
        with patch("src.graphs.langgraph_workflow._get_llm", return_value=mock_llm):
            result = plan_node(state)
        assert len(result["strategies"]) >= 1


# ── execute_node ───────────────────────────────────────────────────────────


class TestExecuteNode:
    def test_execute_node_picks_first_strategy(self):
        state = _make_state(strategies=["Strategy A", "Strategy B"])
        result = execute_node(state)
        assert result["chosen_strategy"] == "Strategy A"
        assert len(result["actions_taken"]) == 1
        assert result["actions_taken"][0]["status"] == "simulated"

    def test_execute_node_no_strategies(self):
        state = _make_state(strategies=[])
        result = execute_node(state)
        assert result["chosen_strategy"] == "No strategy available"

    def test_execute_node_appends_to_existing_actions(self):
        existing_action = {"strategy": "Old", "status": "simulated"}
        state = _make_state(strategies=["New"], actions_taken=[existing_action])
        result = execute_node(state)
        assert len(result["actions_taken"]) == 2


# ── reflect_node ───────────────────────────────────────────────────────────


class TestReflectNode:
    def test_reflect_node_no_llm_with_strategy(self):
        state = _make_state(chosen_strategy="Lease the parcel", iteration=1)
        with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
            result = reflect_node(state)
        assert result["score"] == 0.7
        assert "Lease the parcel" in result["reflection"]

    def test_reflect_node_no_llm_no_strategy(self):
        state = _make_state(chosen_strategy=None)
        with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
            result = reflect_node(state)
        assert result["score"] == 0.3

    def test_reflect_node_with_llm_score_parsed(self):
        state = _make_state(
            chosen_strategy="Lease", actions_taken=[{"strategy": "Lease", "status": "simulated"}]
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="SCORE: 0.9 | REFLECTION: Great outcome"
        )
        with patch("src.graphs.langgraph_workflow._get_llm", return_value=mock_llm):
            result = reflect_node(state)
        assert result["score"] == 0.9
        assert result["reflection"] == "Great outcome"

    def test_reflect_node_with_llm_no_score_marker(self):
        state = _make_state(chosen_strategy="Lease", actions_taken=[])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Things went fine")
        with patch("src.graphs.langgraph_workflow._get_llm", return_value=mock_llm):
            result = reflect_node(state)
        assert result["score"] == 0.5


# ── should_continue ────────────────────────────────────────────────────────


class TestShouldContinue:
    def test_high_score_returns_end(self):
        state = _make_state(score=0.8, iteration=0)
        assert should_continue(state) == END

    def test_score_above_08_returns_end(self):
        state = _make_state(score=0.95, iteration=1)
        assert should_continue(state) == END

    def test_max_iterations_returns_end(self):
        state = _make_state(score=0.5, iteration=3)
        assert should_continue(state) == END

    def test_low_score_low_iteration_returns_assess(self):
        state = _make_state(score=0.5, iteration=0)
        assert should_continue(state) == "assess"

    def test_boundary_score_below_08_continues(self):
        state = _make_state(score=0.79, iteration=2)
        assert should_continue(state) == "assess"


# ── build_optimization_graph ───────────────────────────────────────────────


class TestBuildOptimizationGraph:
    def test_returns_compiled_graph(self):
        graph = build_optimization_graph()
        # If LangGraph is available the graph should be compiled
        from src.graphs.langgraph_workflow import LANGGRAPH_AVAILABLE

        if LANGGRAPH_AVAILABLE:
            assert graph is not None
        else:
            assert graph is None

    def test_langgraph_unavailable_returns_none(self):
        with patch("src.graphs.langgraph_workflow.LANGGRAPH_AVAILABLE", False):
            graph = build_optimization_graph()
            assert graph is None


# ── run_parcel_optimization ────────────────────────────────────────────────


class TestRunParcelOptimization:
    @pytest.mark.asyncio
    async def test_fallback_without_langgraph(self):
        """When graph is None, nodes run directly."""
        parcel_state = {
            "parcel_id": "p-001",
            "balance_usdx": 50.0,
            "location": {"lat": 0.0, "lng": 0.0},
        }
        with patch("src.graphs.langgraph_workflow._get_graph", return_value=None):
            with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
                result = await run_parcel_optimization(parcel_state, context={"env": "test"})
        assert "assessment" in result
        assert isinstance(result["strategies"], list)
        assert result["chosen_strategy"] is not None

    @pytest.mark.asyncio
    async def test_with_graph(self):
        """When graph is available, ainvoke is called."""
        parcel_state = {"parcel_id": "p-002", "balance_usdx": 0.0}
        expected = {
            "parcel_state": parcel_state,
            "assessment": "all good",
            "strategies": ["do nothing"],
            "chosen_strategy": "do nothing",
            "actions_taken": [],
            "reflection": "ok",
            "score": 0.9,
            "iteration": 0,
            "context": {},
        }

        async def fake_ainvoke(*args, **kwargs):
            return expected

        mock_graph = MagicMock()
        mock_graph.ainvoke = fake_ainvoke

        with patch("src.graphs.langgraph_workflow._get_graph", return_value=mock_graph):
            result = await run_parcel_optimization(parcel_state)
        assert result["assessment"] == "all good"
