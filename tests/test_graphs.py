"""Tests for LangGraph workflow optimization for parcel agents."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graphs.langgraph_workflow import (
    ParcelOptimizationWorkflow,
    WorkflowState,
    optimize_parcel_strategy,
)

# ── Workflow Tests ───────────────────────────────────────────────────────────


def test_workflow_state_creation():
    """Test WorkflowState initialization."""
    state = WorkflowState(
        parcel_id="test-001", context={"market": "bullish"}, current_step="analyze"
    )

    assert state.parcel_id == "test-001"
    assert state.context["market"] == "bullish"
    assert state.current_step == "analyze"
    assert state.strategies == []


def test_workflow_creation():
    """Test ParcelOptimizationWorkflow initialization."""
    workflow = ParcelOptimizationWorkflow(parcel_id="test-001", model="gpt-4")

    assert workflow.parcel_id == "test-001"
    assert workflow.model == "gpt-4"
    assert workflow.graph is not None


@pytest.mark.asyncio
async def test_workflow_analyze_step():
    """Test the analyze step of the workflow."""
    workflow = ParcelOptimizationWorkflow(parcel_id="test-001", model="gpt-4")

    state = WorkflowState(
        parcel_id="test-001",
        context={"balance": 100.0, "market": "bullish"},
        current_step="analyze",
    )

    with patch.object(workflow, "_get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="positive assessment")
        mock_get_llm.return_value = mock_llm

        result = await workflow.analyze(state)
        assert "assessment" in result
        assert result["assessment"] == "positive assessment"


@pytest.mark.asyncio
async def test_workflow_generate_strategies():
    """Test strategy generation step."""
    workflow = ParcelOptimizationWorkflow(parcel_id="test-001", model="gpt-4")

    state = WorkflowState(
        parcel_id="test-001",
        context={"market": "bullish", "balance": 100.0},
        current_step="generate_strategies",
        analysis={"assessment": "positive"},
    )

    with patch.object(workflow, "_get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="1. Trade strategy\n2. Lease strategy")
        mock_get_llm.return_value = mock_llm

        strategies = await workflow.generate_strategies(state)
        assert len(strategies) >= 1
        assert strategies[0]["type"] == "strategy"


@pytest.mark.asyncio
async def test_workflow_evaluate_strategies():
    """Test strategy evaluation step."""
    workflow = ParcelOptimizationWorkflow(parcel_id="test-001", model="gpt-4")

    strategies = [
        {"type": "trade", "action": "buy", "amount": 50.0, "risk": "medium"},
        {"type": "lease", "action": "offer", "price": 25.0, "risk": "low"},
    ]

    state = WorkflowState(
        parcel_id="test-001", context={}, current_step="evaluate", strategies=strategies
    )

    evaluated = await workflow.evaluate_strategies(state)
    assert len(evaluated) > 0
    assert all("score" in s for s in evaluated)


@pytest.mark.asyncio
async def test_workflow_select_best_strategy():
    """Test selecting the best strategy."""
    workflow = ParcelOptimizationWorkflow(parcel_id="test-001", model="gpt-4")

    strategies = [
        {"type": "trade", "score": 0.85, "risk": "medium"},
        {"type": "lease", "score": 0.92, "risk": "low"},
        {"type": "invest", "score": 0.78, "risk": "high"},
    ]

    state = WorkflowState(
        parcel_id="test-001", context={}, current_step="select", strategies=strategies
    )

    best = await workflow.select_best_strategy(state)
    assert best["type"] == "lease"
    assert best["score"] == 0.92


@pytest.mark.asyncio
async def test_workflow_full_execution():
    """Test complete workflow execution end-to-end."""
    workflow = ParcelOptimizationWorkflow(parcel_id="test-001", model="gpt-4")

    with patch.object(workflow, "_get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content="positive assessment"),
            MagicMock(content="1. Strategy A\n2. Strategy B"),
        ]
        mock_get_llm.return_value = mock_llm

        result = await workflow.run({"parcel_id": "test-001"})

        assert "assessment" in result
        assert "strategies" in result
        assert "best_strategy" in result


@pytest.mark.asyncio
async def test_optimize_parcel_strategy_function():
    """Test the main optimization function."""
    context = {
        "balance": 100.0,
        "market": "bullish",
        "location": {"lat": 37.7749, "lng": -122.4194},
    }

    # We patch the production class
    with patch("src.graphs.langgraph_workflow.ParcelOptimizationWorkflow") as MockWorkflow:
        mock_instance = MockWorkflow.return_value
        mock_instance.run = AsyncMock(
            return_value={
                "assessment": "positive",
                "strategies": [{"type": "trade"}],
                "best_strategy": {"type": "trade", "score": 0.9},
            }
        )

        result = await optimize_parcel_strategy(parcel_id="test-001", context=context)

        assert result["assessment"] == "positive"
        assert len(result["strategies"]) > 0


def test_workflow_state_transitions():
    """Test workflow state transitions."""
    workflow = ParcelOptimizationWorkflow(parcel_id="test-001", model="gpt-4")

    states = ["analyze", "generate_strategies", "evaluate", "select", "complete"]

    for i, state in enumerate(states[:-1]):
        next_state = workflow.next_step(state)
        assert next_state == states[i + 1]


@pytest.mark.asyncio
async def test_workflow_error_handling():
    """Test workflow error handling."""
    workflow = ParcelOptimizationWorkflow(parcel_id="test-001", model="gpt-4")

    state = WorkflowState(parcel_id="test-001", context={}, current_step="analyze")

    with patch.object(workflow, "_get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("LLM API error")
        mock_get_llm.return_value = mock_llm

        result = await workflow.run(state)
        assert "error" in result
        assert "LLM API error" in result["error"]


@pytest.mark.asyncio
async def test_workflow_with_constraints():
    """Test workflow with budget and risk constraints."""
    workflow = ParcelOptimizationWorkflow(
        parcel_id="test-001", model="gpt-4", max_budget=50.0, risk_tolerance="low"
    )

    strategies = [
        {"type": "trade", "amount": 60.0, "risk": "medium"},
        {"type": "lease", "amount": 40.0, "risk": "low"},
    ]

    filtered = workflow.filter_by_constraints(strategies)

    # Should filter out the trade that exceeds budget
    assert len(filtered) == 1
    assert filtered[0]["type"] == "lease"


def test_workflow_graph_structure():
    """Test the LangGraph graph structure."""
    workflow = ParcelOptimizationWorkflow(parcel_id="test-001", model="gpt-4")

    # Verify graph nodes
    nodes = workflow.graph.get_nodes()
    # In production nodes are assess, plan, execute, reflect
    assert "assess" in nodes
    assert "plan" in nodes
    assert "execute" in nodes
    assert "reflect" in nodes


@pytest.mark.asyncio
async def test_workflow_with_memory():
    """Test workflow with historical memory."""
    workflow = ParcelOptimizationWorkflow(parcel_id="test-001", model="gpt-4", use_memory=True)

    # First run
    state1 = WorkflowState(
        parcel_id="test-001", context={"market": "bullish"}, current_step="analyze"
    )

    with patch.object(workflow, "_get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="positive")
        mock_get_llm.return_value = mock_llm
        await workflow.run(state1)

    # Second run should access memory
    assert workflow.memory is not None


@pytest.mark.asyncio
async def test_multi_objective_optimization():
    """Test optimization with multiple objectives."""
    workflow = ParcelOptimizationWorkflow(
        parcel_id="test-001",
        model="gpt-4",
        objectives=["maximize_profit", "minimize_risk", "maximize_liquidity"],
    )

    strategies = [
        {"profit": 100, "risk": 0.8, "liquidity": 0.5},
        {"profit": 80, "risk": 0.3, "liquidity": 0.9},
        {"profit": 90, "risk": 0.5, "liquidity": 0.7},
    ]

    ranked = workflow.rank_multi_objective(strategies)

    # Should balance all objectives
    assert len(ranked) == 3
    assert all("score" in s for s in ranked)
