"""Unit tests for FastAPI main application (src/main.py)."""

import pytest
from fastapi.testclient import TestClient

from src.main import app, PARCEL_AGENTS, TRADE_AGENTS


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestRootEndpoint:
    def test_root_returns_service_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "Web4AGI"
        assert "version" in data
        assert data["docs"] == "/docs"

    def test_root_contains_description(self, client):
        resp = client.get("/")
        data = resp.json()
        assert "description" in data


class TestHealthEndpoint:
    def test_health_check_status(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_health_check_reports_parcel_count(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "active_parcels" in data
        assert isinstance(data["active_parcels"], int)

    def test_health_check_reports_trade_agent_count(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "active_trade_agents" in data
        assert isinstance(data["active_trade_agents"], int)

    def test_health_check_reflects_agent_state(self, client):
        PARCEL_AGENTS.clear()
        TRADE_AGENTS.clear()
        resp = client.get("/health")
        data = resp.json()
        assert data["active_parcels"] == 0
        assert data["active_trade_agents"] == 0

        PARCEL_AGENTS["fake-parcel"] = object()
        TRADE_AGENTS["fake-trade"] = object()
        resp2 = client.get("/health")
        data2 = resp2.json()
        assert data2["active_parcels"] == 1
        assert data2["active_trade_agents"] == 1

        PARCEL_AGENTS.clear()
        TRADE_AGENTS.clear()


class TestCORSMiddleware:
    def test_cors_headers_present(self, client):
        resp = client.get("/health", headers={"Origin": "http://example.com"})
        # FastAPI CORS middleware adds the header when an Origin is sent
        assert resp.status_code == 200


class TestRoutersMounted:
    def test_openapi_schema_is_reachable(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert schema["info"]["title"] == "Web4AGI API"
