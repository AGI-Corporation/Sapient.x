"""Tests for src/main.py FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from src.main import app, PARCEL_AGENTS, TRADE_AGENTS


@pytest.fixture
def client():
    """Return a synchronous TestClient for the FastAPI app."""
    with TestClient(app) as c:
        yield c


# ── Root / health endpoints ───────────────────────────────────────────────────

def test_root_endpoint(client):
    """GET / returns service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Web4AGI"
    assert "version" in data
    assert "docs" in data


def test_health_endpoint(client):
    """GET /health returns healthy status and agent counts."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_parcels" in data
    assert "active_trade_agents" in data
    assert isinstance(data["active_parcels"], int)
    assert isinstance(data["active_trade_agents"], int)


def test_health_reflects_parcel_agent_count(client):
    """GET /health active_parcels reflects PARCEL_AGENTS dict size."""
    original = dict(PARCEL_AGENTS)
    try:
        PARCEL_AGENTS["p-test"] = object()
        response = client.get("/health")
        assert response.json()["active_parcels"] == len(PARCEL_AGENTS)
    finally:
        PARCEL_AGENTS.clear()
        PARCEL_AGENTS.update(original)


def test_health_reflects_trade_agent_count(client):
    """GET /health active_trade_agents reflects TRADE_AGENTS dict size."""
    original = dict(TRADE_AGENTS)
    try:
        TRADE_AGENTS["t-test"] = object()
        response = client.get("/health")
        assert response.json()["active_trade_agents"] == len(TRADE_AGENTS)
    finally:
        TRADE_AGENTS.clear()
        TRADE_AGENTS.update(original)


# ── Router mount verification ─────────────────────────────────────────────────

def test_parcels_router_mounted(client):
    """Parcels router is reachable at /api/v1/parcels/."""
    response = client.get("/api/v1/parcels/")
    assert response.status_code == 200


def test_trades_router_mounted(client):
    """Trades router is reachable at /api/v1/trades/."""
    response = client.get("/api/v1/trades/")
    assert response.status_code == 200


def test_contracts_router_mounted(client):
    """Contracts router is reachable at /api/v1/contracts/."""
    response = client.get("/api/v1/contracts/")
    assert response.status_code == 200


def test_payments_router_mounted(client):
    """Payments router is reachable at /api/v1/payments/."""
    response = client.get("/api/v1/payments/")
    assert response.status_code == 200


def test_mcp_router_mounted(client):
    """MCP router is reachable at /api/v1/mcp/tools."""
    response = client.get("/api/v1/mcp/tools")
    assert response.status_code == 200


# ── App metadata ──────────────────────────────────────────────────────────────

def test_app_title():
    """FastAPI app has the expected title."""
    assert app.title == "Web4AGI API"


def test_app_version_matches_package():
    """FastAPI app version matches the package __version__."""
    from src import __version__
    assert app.version == __version__


# ── Lifespan (startup/shutdown) ───────────────────────────────────────────────

def test_lifespan_startup_and_shutdown(capsys):
    """Lifespan context manager prints startup and shutdown messages."""
    with TestClient(app):
        out = capsys.readouterr().out
        assert "Starting up" in out
    out = capsys.readouterr().out
    assert "Shutting down" in out


# ── Individual resource endpoints ─────────────────────────────────────────────

def test_get_parcel_by_id(client):
    """GET /api/v1/parcels/{parcel_id} returns the parcel_id in response."""
    response = client.get("/api/v1/parcels/p-001")
    assert response.status_code == 200
    assert response.json()["parcel_id"] == "p-001"


def test_get_trade_by_id(client):
    """GET /api/v1/trades/{trade_id} returns the trade_id in response."""
    response = client.get("/api/v1/trades/t-001")
    assert response.status_code == 200
    assert response.json()["trade_id"] == "t-001"


def test_get_contract_by_id(client):
    """GET /api/v1/contracts/{contract_id} returns the contract_id in response."""
    response = client.get("/api/v1/contracts/c-001")
    assert response.status_code == 200
    assert response.json()["contract_id"] == "c-001"
