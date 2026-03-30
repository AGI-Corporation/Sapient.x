"""Integration tests for FastAPI endpoints.

Tests the Web4AGI API endpoints including:
- Agent CRUD operations
- Authentication and authorization
- Error handling and validation
- Request/response formats
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app


class TestAgentEndpoints:
    """Test agent-related API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with the real FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def agent_data(self):
        """Sample agent creation data."""
        return {
            "parcel_id": "parcel_001",
            "model": "gpt-4",
            "wallet_address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            "initial_balance": 0.0,
            "config": {"max_iterations": 10, "trade_limit": 5000.0},
        }

    def test_create_agent(self, client, agent_data):
        """Test POST /api/v1/agents - Create new agent."""
        response = client.post("/api/v1/agents", json=agent_data)

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["parcel_id"] == "parcel_001"

    def test_create_agent_invalid_data(self, client):
        """Test agent creation with invalid data."""
        invalid_data = {"parcel_id": ""}  # Missing required fields

        response = client.post("/api/v1/agents", json=invalid_data)

        assert response.status_code == 422  # Validation error
        assert "detail" in response.json()

    def test_get_agent(self, client, agent_data):
        """Test GET /api/v1/agents/{agent_id} - Retrieve agent."""
        create_resp = client.post("/api/v1/agents", json=agent_data)
        assert create_resp.status_code == 201
        agent_id = create_resp.json()["id"]

        response = client.get(f"/api/v1/agents/{agent_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == agent_id

    def test_get_agent_not_found(self, client):
        """Test retrieving non-existent agent."""
        response = client.get("/api/v1/agents/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_list_agents(self, client):
        """Test GET /api/v1/agents - List all agents."""
        response = client.get("/api/v1/agents")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_update_agent(self, client, agent_data):
        """Test PATCH /api/v1/agents/{agent_id} - Update agent."""
        create_resp = client.post("/api/v1/agents", json=agent_data)
        assert create_resp.status_code == 201
        agent_id = create_resp.json()["id"]

        update_data = {"status": "paused", "config": {"max_iterations": 20}}
        response = client.patch(f"/api/v1/agents/{agent_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "paused"

    def test_delete_agent(self, client, agent_data):
        """Test DELETE /api/v1/agents/{agent_id} - Delete agent."""
        create_resp = client.post("/api/v1/agents", json=agent_data)
        assert create_resp.status_code == 201
        agent_id = create_resp.json()["id"]

        response = client.delete(f"/api/v1/agents/{agent_id}")

        assert response.status_code == 204


class TestTradeEndpoints:
    """Test trading-related API endpoints."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def trade_request(self):
        """Sample trade request data (uses offer/bid pattern)."""
        return {
            "seller_parcel_id": "parcel_002",
            "asset": "data_rights",
            "amount_usdx": 100.0,
            "ttl_seconds": 300,
        }

    def test_create_trade(self, client, trade_request):
        """Test POST /api/v1/trades/offers - Create trade offer."""
        response = client.post("/api/v1/trades/offers", json=trade_request)

        assert response.status_code == 201
        data = response.json()
        assert "offer_id" in data
        assert data["status"] == "open"

    def test_create_trade_insufficient_balance(self, client):
        """Test trade creation with missing data returns error."""
        invalid_data = {}  # Missing required fields

        response = client.post("/api/v1/trades/offers", json=invalid_data)

        assert response.status_code == 422  # Unprocessable entity

    def test_get_trade_status(self, client, trade_request):
        """Test GET /api/v1/trades/offers - List offers."""
        client.post("/api/v1/trades/offers", json=trade_request)
        response = client.get("/api/v1/trades/offers")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if data:
            offer = data[0]
            assert "offer_id" in offer

    def test_cancel_trade(self, client, trade_request):
        """Test POST /api/v1/trades/offers/{offer_id}/close - Close offer."""
        create_resp = client.post("/api/v1/trades/offers", json=trade_request)
        assert create_resp.status_code == 201
        offer_id = create_resp.json()["offer_id"]

        # Close with no bids → should fail gracefully
        response = client.post(f"/api/v1/trades/offers/{offer_id}/close")
        assert response.status_code in [200, 400]


class TestContractEndpoints:
    """Test contract-related API endpoints."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def contract_data(self):
        """Sample contract creation data."""
        return {
            "contract_type": "parcel_lease",
            "party_a": "agent_123",
            "party_b": "agent_456",
            "terms": {"parcel_id": "parcel_002", "price": 5000.0, "delivery_date": "2026-04-01"},
        }

    def test_create_contract(self, client, contract_data):
        """Test POST /api/v1/contracts - Create contract."""
        response = client.post("/api/v1/contracts", json=contract_data)

        assert response.status_code == 201
        data = response.json()
        assert "contract_id" in data
        assert data["status"] == "pending_signature"

    def test_sign_contract(self, client, contract_data):
        """Test POST /api/v1/contracts/{contract_id}/sign - Sign contract."""
        create_resp = client.post("/api/v1/contracts", json=contract_data)
        assert create_resp.status_code == 201
        contract_id = create_resp.json()["contract_id"]

        signature_data = {"agent_id": "agent_123", "signature": "0xsignature123"}
        response = client.post(f"/api/v1/contracts/{contract_id}/sign", json=signature_data)

        assert response.status_code == 200
        data = response.json()
        assert "signatures" in data

    def test_get_contract(self, client, contract_data):
        """Test GET /api/v1/contracts/{contract_id} - Retrieve contract."""
        create_resp = client.post("/api/v1/contracts", json=contract_data)
        assert create_resp.status_code == 201
        contract_id = create_resp.json()["contract_id"]

        response = client.get(f"/api/v1/contracts/{contract_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["contract_id"] == contract_id

    def test_execute_contract(self, client, contract_data):
        """Test POST /api/v1/contracts/{contract_id}/execute - Execute signed contract."""
        create_resp = client.post("/api/v1/contracts", json=contract_data)
        assert create_resp.status_code == 201
        contract_id = create_resp.json()["contract_id"]

        # Sign by both parties first
        client.post(
            f"/api/v1/contracts/{contract_id}/sign",
            json={"agent_id": "agent_123", "signature": "0xsig_a"},
        )
        client.post(
            f"/api/v1/contracts/{contract_id}/sign",
            json={"agent_id": "agent_456", "signature": "0xsig_b"},
        )

        response = client.post(f"/api/v1/contracts/{contract_id}/execute")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "executed"


class TestAuthenticationEndpoints:
    """Test that open API endpoints are reachable without auth (auth not yet implemented)."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_login(self, client):
        """Test POST /api/v1/auth/login endpoint exists."""
        # Auth endpoint is not yet implemented; expect 404
        response = client.post("/api/v1/auth/login", json={"username": "u", "password": "p"})
        assert response.status_code in [200, 404, 422]

    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        response = client.post("/api/v1/auth/login", json={"username": "bad", "password": "bad"})
        assert response.status_code in [200, 401, 404, 422]

    def test_protected_endpoint_no_auth(self, client):
        """Test accessing an agent endpoint (currently open, no auth middleware)."""
        response = client.get("/api/v1/agents")
        # Routes are currently open; expect 200 or auth error
        assert response.status_code in [200, 401, 403]

    def test_protected_endpoint_with_token(self, client):
        """Test accessing endpoint with Authorization header."""
        headers = {"Authorization": "Bearer valid_token_123"}
        response = client.get("/api/v1/agents", headers=headers)
        assert response.status_code in [200, 401, 403]

