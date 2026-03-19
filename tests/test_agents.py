"""Tests for Web4AGI agent classes."""

import pytest
from src.agents.trade_agent import TradeAgent

# ── ParcelAgent Tests ───────────────────────────────────────────────────────

def test_parcel_agent_creation(parcel_agent):
    """Test ParcelAgent initialization."""
    assert parcel_agent.parcel_id == "test-parcel-001"
    assert parcel_agent.owner_address.startswith("0x")
    assert parcel_agent.location["lat"] == 37.7749
    assert parcel_agent.state.active is True


def test_parcel_agent_state(parcel_agent):
    """Test ParcelAgent state retrieval."""
    state = parcel_agent.get_state()
    assert "parcel_id" in state
    assert "owner" in state
    assert "location" in state
    assert "balance_usdx" in state
    assert state["balance_usdx"] == 0.0


def test_parcel_agent_metadata_update(parcel_agent):
    """Test updating parcel metadata."""
    parcel_agent.update_metadata("zone", "sf-downtown")
    state = parcel_agent.get_state()
    assert state["metadata"]["zone"] == "sf-downtown"


@pytest.mark.asyncio
async def test_parcel_agent_deposit(parcel_agent):
    """Test USDx deposit into parcel wallet."""
    result = await parcel_agent.deposit(amount_usdx=50.0)
    assert result["success"] is True
    # In simulation mode, balance updates locally
    assert parcel_agent.state.balance_usdx == 50.0


@pytest.mark.asyncio
async def test_parcel_agent_trade(parcel_agent):
    """Test USDx trade between parcels."""
    # Deposit first
    await parcel_agent.deposit(amount_usdx=100.0)

    # Execute trade
    result = await parcel_agent.trade(
        counterparty_id="test-parcel-002",
        amount_usdx=25.0,
        trade_type="transfer",
    )

    assert result["success"] is True
    assert parcel_agent.state.balance_usdx == 75.0


@pytest.mark.asyncio
async def test_parcel_agent_insufficient_balance(parcel_agent):
    """Test trade with insufficient balance fails gracefully."""
    result = await parcel_agent.trade(
        counterparty_id="test-parcel-002",
        amount_usdx=999.0,
        trade_type="transfer",
    )
    assert result["success"] is False
    assert "insufficient" in result["error"].lower()


@pytest.mark.asyncio
async def test_parcel_agent_sign_contract(parcel_agent, sample_contract):
    """Test contract signing."""
    result = await parcel_agent.sign_contract(
        counterparty_id="test-parcel-002",
        contract=sample_contract,
    )
    assert result["success"] is True


@pytest.mark.asyncio
async def test_parcel_agent_send_message(parcel_agent):
    """Test MCP message sending."""
    result = await parcel_agent.send_message(
        target_parcel_id="test-parcel-002",
        content={"type": "trade_request", "amount": 10.0},
    )
    assert result["success"] is True


@pytest.mark.asyncio
async def test_parcel_agent_optimize(parcel_agent):
    """Test LangGraph optimization workflow."""
    result = await parcel_agent.optimize(context={"market": "bullish"})
    assert "assessment" in result
    assert "strategies" in result
    assert isinstance(result["strategies"], list)


# ── TradeAgent Tests ─────────────────────────────────────────────────────────

def test_trade_agent_creation(trade_agent):
    """Test TradeAgent initialization."""
    assert trade_agent.agent_id == "test-trade-agent-001"
    assert len(trade_agent.offers) == 0
    assert len(trade_agent.trade_history) == 0


def test_trade_agent_create_offer(trade_agent):
    """Test creating a trade offer."""
    offer = trade_agent.create_offer(
        seller_id="test-parcel-001",
        asset="premium_location_rights",
        amount_usdx=100.0,
        ttl_seconds=600,
    )
    assert offer.seller_id == "test-parcel-001"
    assert offer.amount_usdx == 100.0
    assert not offer.is_expired()
    assert len(trade_agent.offers) == 1


def test_trade_agent_place_bid(trade_agent):
    """Test placing a bid on an offer."""
    offer = trade_agent.create_offer(
        seller_id="test-parcel-001",
        asset="data_access",
        amount_usdx=50.0,
    )

    result = trade_agent.place_bid(
        offer_id=offer.offer_id,
        bidder_id="test-parcel-002",
        bid_amount=55.0,
    )

    assert result["success"] is True
    assert len(offer.bids) == 1
    assert offer.best_bid()["amount"] == 55.0


def test_trade_agent_close_offer(trade_agent):
    """Test closing an offer and selecting winner."""
    offer = trade_agent.create_offer(
        seller_id="test-parcel-001",
        asset="parcel_lease",
        amount_usdx=200.0,
    )

    trade_agent.place_bid(offer.offer_id, "bidder-A", 210.0)
    trade_agent.place_bid(offer.offer_id, "bidder-B", 220.0)
    trade_agent.place_bid(offer.offer_id, "bidder-C", 215.0)

    result = trade_agent.close_offer(offer.offer_id)

    assert result["success"] is True
    assert result["winner"] == "bidder-B"
    assert result["amount"] == 220.0
    assert len(trade_agent.trade_history) == 1


def test_trade_agent_contract_templates(trade_agent):
    """Test contract template generation."""
    lease = TradeAgent.parcel_lease_contract(
        lessor_id="0xLessor",
        lessee_id="0xLessee",
        parcel_id="parcel-123",
        monthly_usdx=100.0,
        duration_months=6,
    )

    assert lease["type"] == "parcel_lease"
    assert lease["terms"]["total_usdx"] == 600.0

    data_access = TradeAgent.data_access_contract(
        provider_id="0xProvider",
        consumer_id="0xConsumer",
        dataset="parcel_analytics",
        price_usdx=25.0,
    )

    assert data_access["type"] == "data_access"
    assert data_access["terms"]["price_usdx"] == 25.0


def test_trade_agent_volume_calculation(trade_agent):
    """Test trade volume calculation."""
    offer1 = trade_agent.create_offer("seller-1", "asset-1", 100.0)
    trade_agent.place_bid(offer1.offer_id, "buyer-1", 105.0)
    trade_agent.close_offer(offer1.offer_id)

    offer2 = trade_agent.create_offer("seller-2", "asset-2", 50.0)
    trade_agent.place_bid(offer2.offer_id, "buyer-2", 52.0)
    trade_agent.close_offer(offer2.offer_id)

    total_volume = trade_agent.volume_usdx()
    assert total_volume == 157.0  # 105 + 52
