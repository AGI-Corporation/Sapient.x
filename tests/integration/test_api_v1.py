"""Integration tests for Web4AGI API v1."""

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root_endpoint(client):
    """Test the root API endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Web4AGI"


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_parcel_lifecycle(client):
    """Test creating, getting, and updating a parcel agent via API."""
    # 1. Create
    parcel_data = {
        "owner_address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
        "location": {"lat": 37.7749, "lng": -122.4194},
        "metadata": {"type": "residential"},
    }
    response = client.post("/api/v1/parcels/", json=parcel_data)
    assert response.status_code == 201
    parcel = response.json()
    parcel_id = parcel["parcel_id"]
    assert parcel["owner"] == parcel_data["owner_address"].lower()

    # 2. Get
    response = client.get(f"/api/v1/parcels/{parcel_id}")
    assert response.status_code == 200
    assert response.json()["metadata"]["type"] == "residential"

    # 3. Update
    update_data = {"metadata": {"zone": "sf-mission"}, "active": False}
    response = client.patch(f"/api/v1/parcels/{parcel_id}", json=update_data)
    assert response.status_code == 200
    updated = response.json()
    assert updated["metadata"]["zone"] == "sf-mission"
    assert updated["active"] is False

    # 4. Delete
    response = client.delete(f"/api/v1/parcels/{parcel_id}")
    assert response.status_code == 200


def test_trade_flow(client):
    """Test creating an offer and placing a bid via API."""
    # Setup: create two parcels
    p1 = client.post(
        "/api/v1/parcels/",
        json={
            "owner_address": "0x1111111111111111111111111111111111111111",
            "location": {"lat": 0, "lng": 0},
        },
    ).json()
    p2 = client.post(
        "/api/v1/parcels/",
        json={
            "owner_address": "0x2222222222222222222222222222222222222222",
            "location": {"lat": 1, "lng": 1},
        },
    ).json()

    # 1. Create Offer
    offer_data = {
        "seller_parcel_id": p1["parcel_id"],
        "asset": "data_lease_001",
        "amount_usdx": 100.0,
    }
    response = client.post("/api/v1/trades/offers", json=offer_data)
    assert response.status_code == 201
    offer = response.json()
    offer_id = offer["offer_id"]

    # 2. Place Bid
    bid_data = {
        "offer_id": offer_id,
        "bidder_parcel_id": p2["parcel_id"],
        "bid_amount_usdx": 110.0,
    }
    response = client.post("/api/v1/trades/bids", json=bid_data)
    assert response.status_code == 200
    assert response.json()["success"] is True

    # 3. Close Offer
    response = client.post(f"/api/v1/trades/close/{offer_id}")
    assert response.status_code == 200
    result = response.json()
    assert result["winner"] == p2["parcel_id"]
    assert result["amount"] == 110.0


def test_mcp_discovery(client):
    """Test listing MCP tools via API."""
    response = client.get("/api/v1/mcp/tools")
    assert response.status_code == 200
    tools = response.json()["tools"]
    assert len(tools) > 0
    # Check for a known tool stub
    tool_names = [t["name"] for t in tools]
    assert "parcel_get_place_hierarchy" in tool_names


def test_system_visibility(client):
    """Test OS-level system visibility endpoints."""
    # 1. Status
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    data = response.json()
    assert "agents_online" in data
    assert "subsystems" in data

    # 2. Map
    response = client.get("/api/v1/system/map")
    assert response.status_code == 200
    geojson = response.json()
    assert geojson["type"] == "FeatureCollection"


def test_nanda_registry_flow(client):
    """Test registering and discovering agents in NANDA registry."""
    # 1. Register
    agent_id = "test-agent-001"
    fact_data = {
        "agent_id": agent_id,
        "capabilities": ["compute", "storage"],
        "metadata": {"version": "1.0.0"},
        "owner_address": "0x1234567890123456789012345678901234567890",
    }
    response = client.post("/api/v1/registry/register", json=fact_data)
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["agent"]["agent_id"] == agent_id

    # 2. Get Fact
    response = client.get(f"/api/v1/registry/{agent_id}")
    assert response.status_code == 200
    assert response.json()["capabilities"] == ["compute", "storage"]

    # 3. Discover (all)
    response = client.get("/api/v1/registry/discover")
    assert response.status_code == 200
    assert len(response.json()) >= 1

    # 4. Discover (by capability)
    response = client.get("/api/v1/registry/discover?capability=compute")
    assert response.status_code == 200
    assert any(a["agent_id"] == agent_id for a in response.json())

    # 5. Discover (non-existent capability)
    response = client.get("/api/v1/registry/discover?capability=magic")
    assert response.status_code == 200
    assert not any(a["agent_id"] == agent_id for a in response.json())


def test_incentive_flow(client):
    """Test executing a USDx incentive via API."""
    # 1. Setup agents
    p1 = client.post(
        "/api/v1/parcels/",
        json={
            "owner_address": "0x1111111111111111111111111111111111111111",
            "location": {"lat": 0, "lng": 0},
        },
    ).json()
    p2 = client.post(
        "/api/v1/parcels/",
        json={
            "owner_address": "0x2222222222222222222222222222222222222222",
            "location": {"lat": 1, "lng": 1},
        },
    ).json()

    # 2. Deposit funds to sponsor
    client.post(f"/api/v1/payments/deposit", json={"parcel_id": p1["parcel_id"], "amount_usdx": 50.0})

    # 3. Execute incentive
    incentive_data = {
        "parcel_id": p1["parcel_id"],
        "target_parcel_id": p2["parcel_id"],
        "amount_usdx": 10.0,
        "incentive_type": "check_in",
        "metadata": {"reason": "visit_loyalty"},
    }
    response = client.post("/api/v1/trades/incentive", json=incentive_data)
    assert response.status_code == 200
    assert response.json()["success"] is True

    # 4. Verify balances
    b1 = client.get(f"/api/v1/parcels/{p1['parcel_id']}").json()["balance_usdx"]
    b2 = client.get(f"/api/v1/parcels/{p2['parcel_id']}").json()["balance_usdx"]
    assert b1 == 40.0
    assert b2 == 10.0
