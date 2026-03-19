"""Integration tests for FastAPI endpoints.

Tests the Web4AGI API endpoints including:
- Agent CRUD operations
- Authentication and authorization
- Error handling and validation
- Request/response formats
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

# Assuming FastAPI app is in src/api/app.py
# from src.api.app import app


class TestAgentEndpoints:
    """Test agent-related API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        # Mock the FastAPI app
        app_mock = Mock()
        return TestClient(app_mock)

    @pytest.fixture
    def agent_data(self):
        """Sample agent creation data."""
        return {
            "parcel_id": "parcel_001",
            "model": "gpt-4",
            "wallet_address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            "initial_balance": 1000.0,
            "config": {
                "max_iterations": 10,
                "trade_limit": 5000.0
            }
        }

    @patch('src.agents.parcel_agent.ParcelAgent')
    def test_create_agent(self, mock_agent_class, client, agent_data):
        """Test POST /api/agents - Create new agent."""
        mock_agent = Mock()
        mock_agent.id = "agent_123"
        mock_agent.parcel_id = agent_data["parcel_id"]
        mock_agent_class.return_value = mock_agent

        response = client.post("/api/agents", json=agent_data)

        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "agent_123"
        assert data["parcel_id"] == "parcel_001"

    def test_create_agent_invalid_data(self, client):
        """Test agent creation with invalid data."""
        invalid_data = {"parcel_id": ""} # Missing required fields

        response = client.post("/api/agents", json=invalid_data)

        assert response.status_code == 422  # Validation error
        assert "detail" in response.json()

    @patch('src.agents.parcel_agent.ParcelAgent')
    def test_get_agent(self, mock_agent_class, client):
        """Test GET /api/agents/{agent_id} - Retrieve agent."""
        mock_agent = Mock()
        mock_agent.id = "agent_123"
        mock_agent.status = "active"
        mock_agent.balance = 1000.0

        response = client.get("/api/agents/agent_123")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "agent_123"

    def test_get_agent_not_found(self, client):
        """Test retrieving non-existent agent."""
        response = client.get("/api/agents/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @patch('src.agents.parcel_agent.ParcelAgent')
    def test_list_agents(self, mock_agent_class, client):
        """Test GET /api/agents - List all agents."""
        response = client.get("/api/agents")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @patch('src.agents.parcel_agent.ParcelAgent')
    def test_update_agent(self, mock_agent_class, client):
        """Test PATCH /api/agents/{agent_id} - Update agent."""
        update_data = {
            "status": "paused",
            "config": {"max_iterations": 20}
        }

        response = client.patch("/api/agents/agent_123", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "paused"

    @patch('src.agents.parcel_agent.ParcelAgent')
    def test_delete_agent(self, mock_agent_class, client):
        """Test DELETE /api/agents/{agent_id} - Delete agent."""
        response = client.delete("/api/agents/agent_123")

        assert response.status_code == 204


class TestTradeEndpoints:
    """Test trading-related API endpoints."""

    @pytest.fixture
    def client(self):
        app_mock = Mock()
        return TestClient(app_mock)

    @pytest.fixture
    def trade_request(self):
        """Sample trade request data."""
        return {
            "agent_id": "agent_123",
            "action": "buy",
            "parcel_id": "parcel_002",
            "amount": 100.0,
            "price": 50.0
        }

    @patch('src.agents.trade_agent.TradeAgent')
    def test_create_trade(self, mock_trade_class, client, trade_request):
        """Test POST /api/trades - Create trade order."""
        mock_trade = Mock()
        mock_trade.id = "trade_456"
        mock_trade.status = "pending"
        mock_trade_class.return_value = mock_trade

        response = client.post("/api/trades", json=trade_request)

        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "trade_456"
        assert data["status"] == "pending"

    def test_create_trade_insufficient_balance(self, client, trade_request):
        """Test trade creation with insufficient balance."""
        trade_request["amount"] = 999999.0  # Unrealistic amount

        response = client.post("/api/trades", json=trade_request)

        assert response.status_code == 400
        assert "insufficient balance" in response.json()["detail"].lower()

    @patch('src.agents.trade_agent.TradeAgent')
    def test_get_trade_status(self, mock_trade_class, client):
        """Test GET /api/trades/{trade_id} - Get trade status."""
        response = client.get("/api/trades/trade_456")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["pending", "completed", "failed", "cancelled"]

    @patch('src.agents.trade_agent.TradeAgent')
    def test_cancel_trade(self, mock_trade_class, client):
        """Test POST /api/trades/{trade_id}/cancel - Cancel trade."""
        response = client.post("/api/trades/trade_456/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"


class TestContractEndpoints:
    """Test contract-related API endpoints."""

    @pytest.fixture
    def client(self):
        app_mock = Mock()
        return TestClient(app_mock)

    @pytest.fixture
    def contract_data(self):
        """Sample contract creation data."""
        return {
            "agent_id": "agent_123",
            "counterparty_id": "agent_456",
            "terms": {
                "parcel_id": "parcel_002",
                "price": 5000.0,
                "delivery_date": "2026-04-01"
            },
            "type": "sale_agreement"
        }

    @patch('src.contracts.manager.ContractManager')
    def test_create_contract(self, mock_contract_class, client, contract_data):
        """Test POST /api/contracts - Create contract."""
        mock_contract = Mock()
        mock_contract.id = "contract_789"
        mock_contract.status = "pending"
        mock_contract_class.return_value = mock_contract

        response = client.post("/api/contracts", json=contract_data)

        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "contract_789"
        assert data["status"] == "pending"

    @patch('src.contracts.manager.ContractManager')
    def test_sign_contract(self, mock_contract_class, client):
        """Test POST /api/contracts/{contract_id}/sign - Sign contract."""
        signature_data = {
            "agent_id": "agent_123",
            "signature": "0xsignature123"
        }

        response = client.post(
            "/api/contracts/contract_789/sign",
            json=signature_data
        )

        assert response.status_code == 200
        data = response.json()
        assert "signatures" in data

    @patch('src.contracts.manager.ContractManager')
    def test_get_contract(self, mock_contract_class, client):
        """Test GET /api/contracts/{contract_id} - Retrieve contract."""
        response = client.get("/api/contracts/contract_789")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "contract_789"

    @patch('src.contracts.manager.ContractManager')
    def test_execute_contract(self, mock_contract_class, client):
        """Test POST /api/contracts/{contract_id}/execute - Execute signed contract."""
        response = client.post("/api/contracts/contract_789/execute")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "executed"


class TestAuthenticationEndpoints:
    """Test authentication and authorization."""

    @pytest.fixture
    def client(self):
        app_mock = Mock()
        return TestClient(app_mock)

    def test_login(self, client):
        """Test POST /api/auth/login - User authentication."""
        credentials = {
            "username": "testuser",
            "password": "testpass123"
        }

        response = client.post("/api/auth/login", json=credentials)

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data

    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        credentials = {
            "username": "wronguser",
            "password": "wrongpass"
        }

        response = client.post("/api/auth/login", json=credentials)

        assert response.status_code == 401

    def test_protected_endpoint_no_auth(self, client):
        """Test accessing protected endpoint without authentication."""
        response = client.get("/api/agents")

        # Should require authentication
        assert response.status_code in [401, 403]

    def test_protected_endpoint_with_token(self, client):
        """Test accessing protected endpoint with valid token."""
        headers = {
            "Authorization": "Bearer valid_token_123"
        }

        response = client.get("/api/agents", headers=headers)

        assert response.status_code == 200
