"""Pytest configuration and fixtures for Web4AGI tests."""

import asyncio
from typing import Any

import pytest

from src.agents.parcel_agent import ParcelAgent
from src.agents.trade_agent import TradeAgent
from src.mcp.mcp_tools import MCPToolkit
from src.payments.x402_client import X402Client

# ── Event loop fixture for async tests ─────────────────────────────────────────


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ── Test data fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def sample_location() -> dict[str, float]:
    """Sample SF location for testing."""
    return {"lat": 37.7749, "lng": -122.4194, "alt": 0.0}


@pytest.fixture
def test_wallet_address() -> str:
    """Test wallet address."""
    return "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb"


@pytest.fixture
def test_private_key() -> str:
    """Test x402 private key (INSECURE - for testing only!)."""
    return "test_key_abc123_do_not_use_in_production"


# ── Agent fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def parcel_agent(sample_location, test_wallet_address, test_private_key) -> ParcelAgent:
    """Create a test ParcelAgent instance."""
    return ParcelAgent(
        parcel_id="test-parcel-001",
        owner_address=test_wallet_address,
        location=sample_location,
        wallet_private_key=test_private_key,
    )


@pytest.fixture
def trade_agent() -> TradeAgent:
    """Create a test TradeAgent instance."""
    return TradeAgent(agent_id="test-trade-agent-001")


@pytest.fixture
def x402_client(test_private_key) -> X402Client:
    """Create a test X402Client instance."""
    return X402Client(private_key=test_private_key)


@pytest.fixture
def mcp_toolkit() -> MCPToolkit:
    """Create a test MCPToolkit instance in local_only mode."""
    return MCPToolkit(agent_id="test-mcp-001", local_only=True)


# ── Mock API fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def sample_parcel_state() -> dict[str, Any]:
    """Sample parcel state for testing optimization workflows."""
    return {
        "parcel_id": "test-parcel-001",
        "owner": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
        "location": {"lat": 37.7749, "lng": -122.4194, "alt": 0.0},
        "balance_usdx": 100.0,
        "metadata": {"type": "test", "zone": "sf-downtown"},
        "active": True,
        "last_updated": "2026-03-07T19:00:00Z",
    }


@pytest.fixture
def sample_trade_request() -> dict[str, Any]:
    """Sample trade request payload."""
    return {
        "from_parcel_id": "test-parcel-001",
        "to_parcel_id": "test-parcel-002",
        "amount_usdx": 10.5,
        "trade_type": "transfer",
        "contract_terms": None,
    }


@pytest.fixture
def sample_contract() -> dict[str, Any]:
    """Sample parcel lease contract."""
    return {
        "type": "parcel_lease",
        "version": "1.0",
        "parties": {
            "lessor": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            "lessee": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",
        },
        "parcel_id": "test-parcel-001",
        "terms": {
            "monthly_rent_usdx": 50.0,
            "duration_months": 12,
            "total_usdx": 600.0,
        },
        "created_at": "2026-03-07T19:00:00Z",
        "status": "pending_signatures",
    }
