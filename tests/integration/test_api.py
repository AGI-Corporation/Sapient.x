"""Integration tests for FastAPI endpoints.

Tests the Web4AGI API endpoints including:
- Parcel CRUD operations
- Trading operations
- Contract lifecycle
- MCP tool and messaging endpoints
- Error handling and validation
- Request/response formats
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestParcelEndpoints:
    """Test parcel-related API endpoints."""

    def test_create_parcel(self):
        """Test POST /api/v1/parcels - Create new parcel agent."""
        parcel_data = {
            "owner_address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            "location": {"lat": 37.7749, "lng": -122.4194, "alt": 0.0},
            "metadata": {"zone": "sf-downtown"},
        }

        response = client.post("/api/v1/parcels/", json=parcel_data)

        assert response.status_code == 201
        data = response.json()
        assert "parcel_id" in data
        assert data["owner"] == parcel_data["owner_address"].lower()

    def test_create_parcel_invalid_address(self):
        """Test parcel creation with an invalid wallet address."""
        invalid_data = {
            "owner_address": "not_a_wallet",
            "location": {"lat": 37.7, "lng": -122.4, "alt": 0.0},
        }

        response = client.post("/api/v1/parcels/", json=invalid_data)

        assert response.status_code == 422
        assert "detail" in response.json()

    def test_list_parcels(self):
        """Test GET /api/v1/parcels - List all parcels."""
        response = client.get("/api/v1/parcels/")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_parcel(self):
        """Test GET /api/v1/parcels/{parcel_id} - Retrieve parcel."""
        # Create a parcel first
        create_resp = client.post(
            "/api/v1/parcels/",
            json={
                "owner_address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
                "location": {"lat": 37.7, "lng": -122.4, "alt": 0.0},
            },
        )
        assert create_resp.status_code == 201
        parcel_id = create_resp.json()["parcel_id"]

        response = client.get(f"/api/v1/parcels/{parcel_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["parcel_id"] == parcel_id

    def test_get_parcel_not_found(self):
        """Test retrieving a non-existent parcel."""
        response = client.get("/api/v1/parcels/nonexistent-parcel-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_update_parcel(self):
        """Test PATCH /api/v1/parcels/{parcel_id} - Update parcel."""
        create_resp = client.post(
            "/api/v1/parcels/",
            json={
                "owner_address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
                "location": {"lat": 37.7, "lng": -122.4, "alt": 0.0},
            },
        )
        parcel_id = create_resp.json()["parcel_id"]

        update_data = {"metadata": {"status": "leased", "tenant": "parcel-002"}}
        response = client.patch(f"/api/v1/parcels/{parcel_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["status"] == "leased"

    def test_delete_parcel(self):
        """Test DELETE /api/v1/parcels/{parcel_id} - Delete parcel."""
        create_resp = client.post(
            "/api/v1/parcels/",
            json={
                "owner_address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
                "location": {"lat": 37.7, "lng": -122.4, "alt": 0.0},
            },
        )
        parcel_id = create_resp.json()["parcel_id"]

        response = client.delete(f"/api/v1/parcels/{parcel_id}")
        assert response.status_code == 204

        # Verify it's gone
        get_resp = client.get(f"/api/v1/parcels/{parcel_id}")
        assert get_resp.status_code == 404


class TestTradeEndpoints:
    """Test trading-related API endpoints."""

    def _create_parcel(self, owner: str = "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb") -> str:
        """Helper: create a parcel and return its ID."""
        resp = client.post(
            "/api/v1/parcels/",
            json={
                "owner_address": owner,
                "location": {"lat": 37.7, "lng": -122.4, "alt": 0.0},
            },
        )
        assert resp.status_code == 201
        return resp.json()["parcel_id"]

    def test_create_offer(self):
        """Test POST /api/v1/trades/offers - Create a trade offer."""
        seller_id = self._create_parcel()
        offer_data = {
            "seller_parcel_id": seller_id,
            "asset": "bandwidth",
            "amount_usdx": 25.0,
            "ttl_seconds": 300,
        }

        response = client.post("/api/v1/trades/offers", json=offer_data)

        assert response.status_code == 201
        data = response.json()
        assert "offer_id" in data
        assert data["asset"] == "bandwidth"

    def test_place_bid(self):
        """Test POST /api/v1/trades/offers/{offer_id}/bid - Place a bid."""
        seller_id = self._create_parcel()
        offer_resp = client.post(
            "/api/v1/trades/offers",
            json={
                "seller_parcel_id": seller_id,
                "asset": "compute",
                "amount_usdx": 10.0,
                "ttl_seconds": 300,
            },
        )
        offer_id = offer_resp.json()["offer_id"]

        bid_data = {"offer_id": offer_id, "bidder_parcel_id": "parcel-bidder", "bid_amount_usdx": 12.0}
        response = client.post(f"/api/v1/trades/offers/{offer_id}/bid", json=bid_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_close_offer(self):
        """Test POST /api/v1/trades/offers/{offer_id}/close - Close an offer."""
        seller_id = self._create_parcel()
        offer_resp = client.post(
            "/api/v1/trades/offers",
            json={
                "seller_parcel_id": seller_id,
                "asset": "storage",
                "amount_usdx": 5.0,
                "ttl_seconds": 300,
            },
        )
        offer_id = offer_resp.json()["offer_id"]
        # Place a bid first so close has a winner
        client.post(
            f"/api/v1/trades/offers/{offer_id}/bid",
            json={"offer_id": offer_id, "bidder_parcel_id": "bidder-99", "bid_amount_usdx": 6.0},
        )

        response = client.post(f"/api/v1/trades/offers/{offer_id}/close")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_trade_history(self):
        """Test GET /api/v1/trades/history - Get trade history."""
        response = client.get("/api/v1/trades/history")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestContractEndpoints:
    """Test contract-related API endpoints."""

    def test_create_contract(self):
        """Test POST /api/v1/contracts - Create contract."""
        contract_data = {
            "contract_type": "parcel_lease",
            "party_a": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            "party_b": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",
            "terms": {"monthly_rent_usdx": 50.0, "duration_months": 12},
        }

        response = client.post("/api/v1/contracts/", json=contract_data)

        assert response.status_code == 201
        data = response.json()
        assert "contract_id" in data
        assert data["status"] == "pending_signature"

    def test_get_contract(self):
        """Test GET /api/v1/contracts/{contract_id} - Retrieve contract."""
        create_resp = client.post(
            "/api/v1/contracts/",
            json={
                "contract_type": "data_access",
                "party_a": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
                "party_b": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",
                "terms": {"dataset": "sf-parcels", "price_usdx": 100.0},
            },
        )
        assert create_resp.status_code == 201
        contract_id = create_resp.json()["contract_id"]

        response = client.get(f"/api/v1/contracts/{contract_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["contract_id"] == contract_id

    def test_get_contract_not_found(self):
        """Test retrieving a non-existent contract."""
        response = client.get("/api/v1/contracts/nonexistent-contract")

        assert response.status_code == 404

    def test_sign_contract(self):
        """Test POST /api/v1/contracts/{contract_id}/sign - Sign contract."""
        party_a = "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb"
        party_b = "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063"
        create_resp = client.post(
            "/api/v1/contracts/",
            json={
                "contract_type": "parcel_lease",
                "party_a": party_a,
                "party_b": party_b,
                "terms": {"rent": 100.0},
            },
        )
        contract_id = create_resp.json()["contract_id"]

        response = client.post(
            f"/api/v1/contracts/{contract_id}/sign",
            json={"agent_id": party_a, "signature": "0xsig_party_a"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "signatures" in data

    def test_execute_contract(self):
        """Test POST /api/v1/contracts/{contract_id}/execute - Execute signed contract."""
        party_a = "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb"
        party_b = "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063"
        create_resp = client.post(
            "/api/v1/contracts/",
            json={
                "contract_type": "parcel_lease",
                "party_a": party_a,
                "party_b": party_b,
                "terms": {"rent": 50.0},
            },
        )
        contract_id = create_resp.json()["contract_id"]

        response = client.post(f"/api/v1/contracts/{contract_id}/execute")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "executed"


class TestMCPEndpoints:
    """Test MCP tool and messaging endpoints."""

    def test_list_tools(self):
        """Test GET /api/v1/mcp/tools - List available tools."""
        response = client.get("/api/v1/mcp/tools")

        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert isinstance(data["tools"], list)
        assert data["count"] >= 0

    def test_call_tool(self):
        """Test POST /api/v1/mcp/tools/call - Call an MCP tool."""
        payload = {"tool_name": "parcel.get_state", "arguments": {"parcel_id": "test-001"}}

        response = client.post("/api/v1/mcp/tools/call", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_send_message(self):
        """Test POST /api/v1/mcp/messages - Send MCP message."""
        msg = {
            "from_parcel_id": "parcel-001",
            "to_parcel_id": "parcel-002",
            "msg_type": "trade_request",
            "payload": {"amount": 10.0},
        }

        response = client.post("/api/v1/mcp/messages", json=msg)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

