"""Unit tests for individual LangGraph node functions and workflow helpers."""

import pytest
from unittest.mock import MagicMock, patch

from src.graphs.langgraph_workflow import (
    assess_node,
    plan_node,
    execute_node,
    reflect_node,
    should_continue,
    _get_llm,
    LANGGRAPH_AVAILABLE,
    LANGCHAIN_AVAILABLE,
    END,
)


def _create_test_state(balance=50.0, score=0.0, iteration=0, strategies=None, chosen=None, actions=None):
    return {
        "parcel_state": {
            "parcel_id": "u-001",
            "owner": "0xUnit",
            "location": {"lat": 0.0, "lng": 0.0, "alt": 0.0},
            "balance_usdx": balance,
            "metadata": {},
            "active": True,
            "last_updated": "2026-01-01T00:00:00",
        },
        "context": {},
        "assessment": None,
        "strategies": strategies or [],
        "chosen_strategy": chosen,
        "actions_taken": actions or [],
        "reflection": None,
        "score": score,
        "iteration": iteration,
    }


# ── _get_llm Tests ─────────────────────────────────────────────────────────────

def test_get_llm_returns_none_when_langchain_unavailable():
    """_get_llm returns None when LANGCHAIN_AVAILABLE is False."""
    with patch("src.graphs.langgraph_workflow.LANGCHAIN_AVAILABLE", False):
        llm = _get_llm()
    assert llm is None


def test_get_llm_returns_none_when_no_api_keys(monkeypatch):
    """_get_llm returns None when neither SENTIENT_API_KEY nor OPENAI_API_KEY is set."""
    monkeypatch.delenv("SENTIENT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("src.graphs.langgraph_workflow.LANGCHAIN_AVAILABLE", True):
        llm = _get_llm()
    assert llm is None


# ── assess_node Unit Tests ─────────────────────────────────────────────────────

@pytest.mark.unit
def test_assess_node_returns_dict_with_assessment():
    s = _create_test_state(balance=75.0)
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
        out = assess_node(s)
    assert isinstance(out, dict)
    assert "assessment" in out
    assert out["assessment"] is not None


@pytest.mark.unit
def test_assess_node_includes_parcel_id_in_fallback():
    s = _create_test_state()
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
        out = assess_node(s)
    assert "u-001" in out["assessment"]


@pytest.mark.unit
def test_assess_node_includes_location_in_fallback():
    s = _create_test_state()
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
        out = assess_node(s)
    # Location dict should appear in assessment string
    assert "lat" in out["assessment"] or "Location" in out["assessment"]


@pytest.mark.unit
def test_assess_node_llm_response_used_as_assessment():
    s = _create_test_state()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Excellent opportunity.")
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=mock_llm):
        out = assess_node(s)
    assert out["assessment"] == "Excellent opportunity."


# ── plan_node Unit Tests ───────────────────────────────────────────────────────

@pytest.mark.unit
def test_plan_node_returns_three_fallback_strategies():
    s = _create_test_state()
    s["assessment"] = "ok"
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
        out = plan_node(s)
    assert len(out["strategies"]) == 3


@pytest.mark.unit
def test_plan_node_strategies_are_strings():
    s = _create_test_state()
    s["assessment"] = "ok"
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
        out = plan_node(s)
    assert all(isinstance(st, str) for st in out["strategies"])


@pytest.mark.unit
def test_plan_node_llm_strategies_parsed():
    s = _create_test_state()
    s["assessment"] = "ok"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="1. Sell land\n2. Buy more\n3. Hold position"
    )
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=mock_llm):
        out = plan_node(s)
    assert len(out["strategies"]) == 3
    assert "Sell land" in out["strategies"][0]


@pytest.mark.unit
def test_plan_node_limits_to_three_llm_strategies():
    s = _create_test_state()
    s["assessment"] = "ok"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="1. S1\n2. S2\n3. S3\n4. S4\n5. S5"
    )
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=mock_llm):
        out = plan_node(s)
    assert len(out["strategies"]) == 3


# ── execute_node Unit Tests ────────────────────────────────────────────────────

@pytest.mark.unit
def test_execute_node_action_status_is_simulated():
    s = _create_test_state(strategies=["Do A"])
    out = execute_node(s)
    assert out["actions_taken"][0]["status"] == "simulated"


@pytest.mark.unit
def test_execute_node_action_has_executed_at():
    s = _create_test_state(strategies=["Do B"])
    out = execute_node(s)
    assert "executed_at" in out["actions_taken"][0]


@pytest.mark.unit
def test_execute_node_does_not_modify_strategies():
    s = _create_test_state(strategies=["A", "B", "C"])
    out = execute_node(s)
    assert out["strategies"] == ["A", "B", "C"]


# ── reflect_node Unit Tests ────────────────────────────────────────────────────

@pytest.mark.unit
def test_reflect_node_score_is_float():
    s = _create_test_state(chosen="S", actions=[{"strategy": "S", "status": "simulated"}])
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
        out = reflect_node(s)
    assert isinstance(out["score"], float)


@pytest.mark.unit
def test_reflect_node_score_in_valid_range():
    s = _create_test_state(chosen="S", actions=[])
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
        out = reflect_node(s)
    assert 0.0 <= out["score"] <= 1.0


@pytest.mark.unit
def test_reflect_node_reflection_is_string():
    s = _create_test_state(chosen="S", actions=[])
    with patch("src.graphs.langgraph_workflow._get_llm", return_value=None):
        out = reflect_node(s)
    assert isinstance(out["reflection"], str)


# ── should_continue Unit Tests ─────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("score,iteration,expected", [
    (0.9, 0, END),
    (0.8, 0, END),
    (0.5, 3, END),
    (0.5, 4, END),
    (0.5, 0, "assess"),
    (0.79, 2, "assess"),
    (0.0, 0, "assess"),
])
def test_should_continue_parametrized(score, iteration, expected):
    s = _create_test_state(score=score, iteration=iteration)
    assert should_continue(s) == expected
