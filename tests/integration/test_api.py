"""Integration tests for FastAPI endpoints.

Tests the Web4AGI API endpoints including:
- Parcel CRUD operations
- Trade offers and bidding
- Contract lifecycle
- Payment operations
- MCP tool invocation
- Error handling and validation
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

# ── Parcel Endpoints ──────────────────────────────────────────────────────────


class TestParcelEndpoints:
    """Test parcel-related API endpoints."""

    def test_list_parcels_empty(self):
        """GET /api/v1/parcels returns a list (possibly empty)."""
        response = client.get("/api/v1/parcels/")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_create_parcel(self):
        """POST /api/v1/parcels/ creates a new parcel agent."""
        payload = {
            "owner_address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            "location": {"lat": 37.7749, "lng": -122.4194, "alt": 0.0},
            "metadata": {"zone": "sf-downtown"},
        }
        response = client.post("/api/v1/parcels/", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert "parcel_id" in data
        assert data["owner"] == payload["owner_address"].lower()

    def test_create_parcel_invalid_address(self):
        """POST /api/v1/parcels/ with bad address returns 422."""
        payload = {
            "owner_address": "not_an_address",
            "location": {"lat": 37.7749, "lng": -122.4194},
        }
        response = client.post("/api/v1/parcels/", json=payload)
        assert response.status_code == 422

    def test_get_parcel(self):
        """GET /api/v1/parcels/{id} returns the parcel state."""
        # Create first
        create_resp = client.post(
            "/api/v1/parcels/",
            json={
                "owner_address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
                "location": {"lat": 1.0, "lng": 2.0},
            },
        )
        parcel_id = create_resp.json()["parcel_id"]

        response = client.get(f"/api/v1/parcels/{parcel_id}")
        assert response.status_code == 200
        assert response.json()["parcel_id"] == parcel_id

    def test_get_parcel_not_found(self):
        """GET /api/v1/parcels/nonexistent returns 404."""
        response = client.get("/api/v1/parcels/nonexistent-id")
        assert response.status_code == 404

    def test_update_parcel_metadata(self):
        """PATCH /api/v1/parcels/{id} updates metadata."""
        create_resp = client.post(
            "/api/v1/parcels/",
            json={
                "owner_address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
                "location": {"lat": 3.0, "lng": 4.0},
            },
        )
        parcel_id = create_resp.json()["parcel_id"]

        response = client.patch(
            f"/api/v1/parcels/{parcel_id}",
            json={"metadata": {"tier": "premium"}},
        )
        assert response.status_code == 200
        assert response.json()["metadata"]["tier"] == "premium"

    def test_delete_parcel(self):
        """DELETE /api/v1/parcels/{id} removes the parcel."""
        create_resp = client.post(
            "/api/v1/parcels/",
            json={
                "owner_address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
                "location": {"lat": 5.0, "lng": 6.0},
            },
        )
        parcel_id = create_resp.json()["parcel_id"]

        del_resp = client.delete(f"/api/v1/parcels/{parcel_id}")
        assert del_resp.status_code == 204

        get_resp = client.get(f"/api/v1/parcels/{parcel_id}")
        assert get_resp.status_code == 404


# ── Trade Endpoints ──────────────────────────────────────────────────────────


class TestTradeEndpoints:
    """Test trade offer and bidding endpoints."""

    def test_list_offers(self):
        """GET /api/v1/trades/offers returns a list."""
        response = client.get("/api/v1/trades/offers")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_create_offer(self):
        """POST /api/v1/trades/offers creates an offer."""
        payload = {
            "seller_parcel_id": "seller-001",
            "asset": "data_rights",
            "amount_usdx": 100.0,
            "ttl_seconds": 300,
        }
        response = client.post("/api/v1/trades/offers", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert "offer_id" in data
        assert data["asset"] == "data_rights"

    def test_place_bid(self):
        """POST /api/v1/trades/offers/{id}/bid places a bid."""
        offer_resp = client.post(
            "/api/v1/trades/offers",
            json={
                "seller_parcel_id": "seller-002",
                "asset": "parcel_lease",
                "amount_usdx": 200.0,
                "ttl_seconds": 300,
            },
        )
        offer_id = offer_resp.json()["offer_id"]

        bid_resp = client.post(
            f"/api/v1/trades/offers/{offer_id}/bid",
            json={
                "offer_id": offer_id,
                "bidder_parcel_id": "buyer-001",
                "bid_amount_usdx": 210.0,
            },
        )
        assert bid_resp.status_code == 200
        assert bid_resp.json()["success"] is True

    def test_close_offer(self):
        """POST /api/v1/trades/offers/{id}/close selects winner."""
        offer_resp = client.post(
            "/api/v1/trades/offers",
            json={
                "seller_parcel_id": "seller-003",
                "asset": "computing_rights",
                "amount_usdx": 50.0,
                "ttl_seconds": 300,
            },
        )
        offer_id = offer_resp.json()["offer_id"]
        client.post(
            f"/api/v1/trades/offers/{offer_id}/bid",
            json={
                "offer_id": offer_id,
                "bidder_parcel_id": "buyer-002",
                "bid_amount_usdx": 60.0,
            },
        )

        close_resp = client.post(f"/api/v1/trades/offers/{offer_id}/close")
        assert close_resp.status_code == 200
        assert close_resp.json()["success"] is True
        assert close_resp.json()["winner"] == "buyer-002"

    def test_close_offer_no_bids(self):
        """Closing an offer with no bids returns 400."""
        offer_resp = client.post(
            "/api/v1/trades/offers",
            json={
                "seller_parcel_id": "seller-004",
                "asset": "unused",
                "amount_usdx": 10.0,
                "ttl_seconds": 300,
            },
        )
        offer_id = offer_resp.json()["offer_id"]
        close_resp = client.post(f"/api/v1/trades/offers/{offer_id}/close")
        assert close_resp.status_code == 400

    def test_get_trade_history(self):
        """GET /api/v1/trades/history returns a list."""
        response = client.get("/api/v1/trades/history")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_volume(self):
        """GET /api/v1/trades/volume returns a numeric value."""
        response = client.get("/api/v1/trades/volume")
        assert response.status_code == 200
        assert "volume_usdx" in response.json()


# ── Contract Endpoints ────────────────────────────────────────────────────────


class TestContractEndpoints:
    """Test contract lifecycle API endpoints."""

    def test_list_contracts(self):
        """GET /api/v1/contracts/ returns a list."""
        response = client.get("/api/v1/contracts/")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_create_contract(self):
        """POST /api/v1/contracts/ creates a contract proposal."""
        payload = {
            "contract_type": "parcel_lease",
            "party_a": "agent-001",
            "party_b": "agent-002",
            "terms": {"monthly_rent": 50.0, "duration_months": 6},
        }
        response = client.post("/api/v1/contracts/", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert "contract_id" in data
        assert data["status"] == "pending_signature"

    def test_get_contract(self):
        """GET /api/v1/contracts/{id} returns the contract."""
        create_resp = client.post(
            "/api/v1/contracts/",
            json={
                "contract_type": "data_access",
                "party_a": "provider-001",
                "party_b": "consumer-001",
                "terms": {"price": 25.0},
            },
        )
        contract_id = create_resp.json()["contract_id"]

        response = client.get(f"/api/v1/contracts/{contract_id}")
        assert response.status_code == 200
        assert response.json()["contract_id"] == contract_id

    def test_get_contract_not_found(self):
        """GET /api/v1/contracts/nonexistent returns 404."""
        response = client.get("/api/v1/contracts/nonexistent")
        assert response.status_code == 404

    def test_sign_and_execute_contract(self):
        """Full flow: create → sign (both) → execute."""
        create_resp = client.post(
            "/api/v1/contracts/",
            json={
                "contract_type": "parcel_lease",
                "party_a": "lessor-001",
                "party_b": "lessee-001",
                "terms": {"monthly_rent": 100.0},
            },
        )
        contract_id = create_resp.json()["contract_id"]

        sign1 = client.post(
            f"/api/v1/contracts/{contract_id}/sign",
            params={"signer_id": "lessor-001", "signature": "0xsig1"},
        )
        assert sign1.status_code == 200

        sign2 = client.post(
            f"/api/v1/contracts/{contract_id}/sign",
            params={"signer_id": "lessee-001", "signature": "0xsig2"},
        )
        assert sign2.status_code == 200
        assert sign2.json()["status"] == "fully_signed"

        execute_resp = client.post(f"/api/v1/contracts/{contract_id}/execute")
        assert execute_resp.status_code == 200
        assert execute_resp.json()["status"] == "executed"

    def test_reject_contract(self):
        """POST /api/v1/contracts/{id}/reject marks as rejected."""
        create_resp = client.post(
            "/api/v1/contracts/",
            json={
                "contract_type": "custom",
                "party_a": "a-001",
                "party_b": "b-001",
                "terms": {"price": 999.0},
            },
        )
        contract_id = create_resp.json()["contract_id"]

        rej = client.post(
            f"/api/v1/contracts/{contract_id}/reject",
            params={"rejector_id": "b-001", "reason": "Too expensive"},
        )
        assert rej.status_code == 200
        assert rej.json()["status"] == "rejected"


# ── Payment Endpoints ─────────────────────────────────────────────────────────


class TestPaymentEndpoints:
    """Test payment-related API endpoints."""

    def test_get_wallet_address(self):
        """GET /api/v1/payments/address returns a 0x address."""
        response = client.get("/api/v1/payments/address")
        assert response.status_code == 200
        assert response.json()["address"].startswith("0x")

    def test_get_balance(self):
        """GET /api/v1/payments/balance/{address} returns a balance."""
        address = "0x742d35Cc6634C0532925a3b8440000000000000f"  # valid 0x + 40 hex chars
        response = client.get(f"/api/v1/payments/balance/{address}")
        assert response.status_code == 200
        assert "balance_usdx" in response.json()

    def test_get_balance_invalid_address(self):
        """GET /api/v1/payments/balance with bad address returns 400."""
        response = client.get("/api/v1/payments/balance/not_an_address")
        assert response.status_code == 400

    def test_transfer(self):
        """POST /api/v1/payments/transfer succeeds in simulation mode."""
        response = client.post(
            "/api/v1/payments/transfer",
            params={
                "to_address": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",
                "amount_usdx": 10.0,
                "memo": "test transfer",
            },
        )
        assert response.status_code == 200
        assert response.json().get("success") is True


# ── MCP Endpoints ─────────────────────────────────────────────────────────────


class TestMCPEndpoints:
    """Test MCP tool and messaging endpoints."""

    def test_list_tools(self):
        """GET /api/v1/mcp/tools returns the list of tools."""
        response = client.get("/api/v1/mcp/tools")
        assert response.status_code == 200
        tools = response.json()
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert any(t["name"] == "parcel.get_state" for t in tools)

    def test_call_tool(self):
        """POST /api/v1/mcp/tools/call calls a registered tool."""
        payload = {"tool_name": "parcel.get_state", "arguments": {"parcel_id": "test-001"}}
        response = client.post("/api/v1/mcp/tools/call", json=payload)
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_call_unknown_tool(self):
        """POST /api/v1/mcp/tools/call with unknown tool returns 400."""
        payload = {"tool_name": "no_such_tool", "arguments": {}}
        response = client.post("/api/v1/mcp/tools/call", json=payload)
        assert response.status_code == 400

    def test_send_message(self):
        """POST /api/v1/mcp/messages sends an MCP message."""
        payload = {
            "from_parcel_id": "agent-001",
            "to_parcel_id": "agent-002",
            "msg_type": "greeting",
            "payload": {"text": "hello"},
        }
        response = client.post("/api/v1/mcp/messages", json=payload)
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_mcp_status(self):
        """GET /api/v1/mcp/status returns connection info."""
        response = client.get("/api/v1/mcp/status")
        assert response.status_code == 200
        data = response.json()
        assert "connected" in data


# ── Root / Health ─────────────────────────────────────────────────────────────


class TestRootEndpoints:
    """Test root and health check endpoints."""

    def test_root(self):
        """GET / returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data

    def test_health(self):
        """GET /health returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

