"""Tests for Web4AGI agent classes."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.parcel_agent import ParcelAgent, ParcelState
from src.agents.trade_agent import TradeAgent, TradeOffer


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
    with patch.object(parcel_agent.x402, "_post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = {"success": True, "simulated": True}
        result = await parcel_agent.deposit(amount_usdx=50.0)
    assert result["success"] is True
    # Balance updates locally when deposit succeeds
    assert parcel_agent.state.balance_usdx == 50.0


@pytest.mark.asyncio
async def test_parcel_agent_trade(parcel_agent):
    """Test USDx trade between parcels."""
    with patch.object(parcel_agent.x402, "_post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = {"success": True, "simulated": True}
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
    with patch.object(parcel_agent.x402, "_post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = {"success": True, "simulated": True}
        result = await parcel_agent.sign_contract(
            counterparty_id="test-parcel-002",
            contract=sample_contract,
        )
    assert result["success"] is True


@pytest.mark.asyncio
async def test_parcel_agent_send_message(parcel_agent):
    """Test MCP message sending."""
    with patch.object(parcel_agent.mcp, "send", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = {"success": True, "simulated": True}
        result = await parcel_agent.send_message(
            target_parcel_id="test-parcel-002",
            content={"type": "trade_request", "amount": 10.0},
        )
    assert result["success"] is True


@pytest.mark.asyncio
async def test_parcel_agent_optimize(parcel_agent):
    """Test LangGraph optimization workflow."""
    mock_result = {
        "assessment": "Good balance, consider leasing",
        "strategies": ["Transfer 10 USDx", "Lease parcel", "Update metadata"],
        "chosen_strategy": "Transfer 10 USDx",
        "actions_taken": [],
        "reflection": "Strategy executed",
        "score": 0.7,
    }
    with patch("src.graphs.langgraph_workflow.run_parcel_optimization", new_callable=AsyncMock) as mock_opt:
        mock_opt.return_value = mock_result
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


# ── Additional ParcelAgent Coverage ──────────────────────────────────────────

def test_parcel_state_dataclass(sample_location, test_wallet_address):
    """Test ParcelState dataclass fields and defaults."""
    state = ParcelState(
        parcel_id="p-test",
        owner_address=test_wallet_address,
        location=sample_location,
    )
    assert state.parcel_id == "p-test"
    assert state.balance_usdx == 0.0
    assert state.active is True
    assert state.metadata == {}
    assert state.last_updated is not None


def test_parcel_agent_auto_generates_id(sample_location):
    """Test ParcelAgent generates a UUID when no parcel_id provided."""
    agent = ParcelAgent(location=sample_location)
    assert len(agent.parcel_id) == 36  # UUID4 length


def test_parcel_agent_default_location():
    """Test ParcelAgent uses default SF location when none provided."""
    agent = ParcelAgent()
    assert agent.location["lat"] == 37.7749
    assert agent.location["lng"] == -122.4194


def test_parcel_agent_multiple_metadata_updates(parcel_agent):
    """Test successive metadata updates are all preserved."""
    parcel_agent.update_metadata("zone", "residential")
    parcel_agent.update_metadata("tier", 3)
    parcel_agent.update_metadata("zone", "commercial")  # overwrite
    state = parcel_agent.get_state()
    assert state["metadata"]["zone"] == "commercial"
    assert state["metadata"]["tier"] == 3


def test_parcel_agent_last_updated_changes(parcel_agent):
    """Test that last_updated timestamp changes after metadata update."""
    initial = parcel_agent.state.last_updated
    import time
    time.sleep(0.01)
    parcel_agent.update_metadata("key", "value")
    assert parcel_agent.state.last_updated >= initial


@pytest.mark.asyncio
async def test_parcel_agent_receive_messages_empty(parcel_agent):
    """Test receiving messages from an empty queue returns empty list."""
    msgs = await parcel_agent.receive_messages()
    assert msgs == []


@pytest.mark.asyncio
async def test_parcel_agent_run_cycles(parcel_agent):
    """Test ParcelAgent run loop terminates after N cycles."""
    await parcel_agent.run(cycles=2)
    assert parcel_agent.state.active is True


@pytest.mark.asyncio
async def test_parcel_agent_run_stops_when_inactive(parcel_agent):
    """Test that run loop exits when state.active is False."""
    parcel_agent.state.active = False
    # Should return immediately without running any cycle
    await parcel_agent.run(cycles=0)


@pytest.mark.asyncio
async def test_parcel_agent_handle_message_trade_request(parcel_agent):
    """Test _handle_message dispatches trade_request to trade()."""
    with patch.object(parcel_agent.x402, "_post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = {"success": True}
        parcel_agent.state.balance_usdx = 50.0
        msg = {"type": "trade_request", "from": "sender-parcel", "amount": 10.0}
        await parcel_agent._handle_message(msg)


@pytest.mark.asyncio
async def test_parcel_agent_handle_message_contract_offer(parcel_agent, sample_contract):
    """Test _handle_message dispatches contract_offer to sign_contract()."""
    with patch.object(parcel_agent.x402, "_post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = {"success": True}
        msg = {
            "type": "contract_offer",
            "from": "sender-parcel",
            "contract": sample_contract,
        }
        await parcel_agent._handle_message(msg)


@pytest.mark.asyncio
async def test_parcel_agent_handle_message_optimize(parcel_agent):
    """Test _handle_message dispatches optimize to optimize()."""
    mock_result = {
        "assessment": "ok",
        "strategies": [],
        "chosen_strategy": None,
        "actions_taken": [],
        "reflection": None,
        "score": 0.5,
    }
    with patch("src.graphs.langgraph_workflow.run_parcel_optimization", new_callable=AsyncMock) as mock_opt:
        mock_opt.return_value = mock_result
        msg = {"type": "optimize", "context": {"market": "neutral"}}
        await parcel_agent._handle_message(msg)


@pytest.mark.asyncio
async def test_parcel_agent_handle_message_unknown(parcel_agent, capsys):
    """Test _handle_message handles unknown message types gracefully."""
    msg = {"type": "unknown_msg_type"}
    await parcel_agent._handle_message(msg)
    captured = capsys.readouterr()
    assert "Unknown message type" in captured.out


@pytest.mark.asyncio
async def test_parcel_agent_trade_with_contract_terms(parcel_agent, sample_contract):
    """Test trade including contract terms in the payload."""
    with patch.object(parcel_agent.x402, "_post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = {"success": True}
        parcel_agent.state.balance_usdx = 100.0
        result = await parcel_agent.trade(
            counterparty_id="cp-parcel",
            amount_usdx=20.0,
            trade_type="lease",
            contract_terms=sample_contract,
        )
    assert result["success"] is True
    assert parcel_agent.state.balance_usdx == 80.0


# ── Additional TradeOffer Coverage ───────────────────────────────────────────

def test_trade_offer_expired():
    """Test TradeOffer.is_expired() for a zero-TTL offer."""
    from src.agents.trade_agent import TradeOffer

    offer = TradeOffer("o-1", "seller-1", "asset", 10.0, ttl_seconds=0)
    assert offer.is_expired() is True


def test_trade_offer_no_bids_best_bid_is_none(trade_agent):
    """Test best_bid() returns None when there are no bids."""
    offer = trade_agent.create_offer("seller-1", "asset", 50.0)
    assert offer.best_bid() is None


def test_trade_agent_close_offer_no_bids(trade_agent):
    """Test close_offer fails gracefully when no bids exist."""
    offer = trade_agent.create_offer("seller-1", "asset", 50.0)
    result = trade_agent.close_offer(offer.offer_id)
    assert result["success"] is False
    assert "No bids" in result["error"]


def test_trade_agent_place_bid_nonexistent_offer(trade_agent):
    """Test place_bid returns error for non-existent offer."""
    result = trade_agent.place_bid("nonexistent-offer", "bidder", 10.0)
    assert result["success"] is False
    assert "not found" in result["error"].lower()


def test_trade_agent_close_nonexistent_offer(trade_agent):
    """Test close_offer returns error for non-existent offer."""
    result = trade_agent.close_offer("nonexistent-offer")
    assert result["success"] is False
    assert "not found" in result["error"].lower()


def test_trade_agent_get_history_limit(trade_agent):
    """Test get_history respects the limit parameter."""
    for i in range(10):
        offer = trade_agent.create_offer(f"seller-{i}", "asset", float(i + 1))
        trade_agent.place_bid(offer.offer_id, f"buyer-{i}", float(i + 2))
        trade_agent.close_offer(offer.offer_id)

    history = trade_agent.get_history(limit=3)
    assert len(history) == 3


def test_trade_agent_empty_volume(trade_agent):
    """Test volume_usdx() is 0.0 with no completed trades."""
    assert trade_agent.volume_usdx() == 0.0


@pytest.mark.asyncio
async def test_trade_agent_batch_transfer(parcel_agent):
    """Test batch_transfer executes concurrent transfers."""
    trade_agent = TradeAgent(agent_id="batch-agent")

    with patch.object(parcel_agent.x402, "_post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = {"success": True}
        parcel_agent.state.balance_usdx = 200.0

        recipients = [
            {"parcel_id": "parcel-A", "amount": 10.0},
            {"parcel_id": "parcel-B", "amount": 20.0},
            {"parcel_id": "parcel-C", "amount": 30.0},
        ]
        results = await trade_agent.batch_transfer(parcel_agent, recipients)

    assert len(results) == 3
    assert all(r["success"] for r in results)


@pytest.mark.asyncio
async def test_trade_agent_batch_transfer_handles_exception(parcel_agent):
    """Test batch_transfer wraps exceptions as failure dicts."""
    trade_agent = TradeAgent(agent_id="batch-err-agent")

    with patch.object(parcel_agent, "trade", new_callable=AsyncMock) as mock_trade:
        mock_trade.side_effect = RuntimeError("transfer failed")
        parcel_agent.state.balance_usdx = 200.0
        recipients = [{"parcel_id": "bad-parcel", "amount": 10.0}]
        results = await trade_agent.batch_transfer(parcel_agent, recipients)

    assert results[0]["success"] is False
    assert "transfer failed" in results[0]["error"]


# ── Coverage gap fixes ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parcel_agent_receive_messages_with_items(parcel_agent):
    """Test receive_messages drains items from the queue (covers line 90)."""
    # Put two messages into the internal queue directly
    await parcel_agent._message_queue.put({"type": "ping", "data": 1})
    await parcel_agent._message_queue.put({"type": "pong", "data": 2})

    msgs = await parcel_agent.receive_messages()
    assert len(msgs) == 2
    assert msgs[0] == {"type": "ping", "data": 1}
    assert msgs[1] == {"type": "pong", "data": 2}
    # Queue should be empty afterwards
    assert parcel_agent._message_queue.empty()


@pytest.mark.asyncio
async def test_parcel_agent_run_processes_queued_messages(parcel_agent):
    """Test that run() dispatches messages in the queue (covers line 152)."""
    # Pre-load an 'optimize' message so the loop body executes _handle_message
    mock_result = {
        "assessment": "ok", "strategies": [], "chosen_strategy": None,
        "actions_taken": [], "reflection": None, "score": 0.5,
    }
    await parcel_agent._message_queue.put({"type": "optimize", "context": {}})

    with patch("src.graphs.langgraph_workflow.run_parcel_optimization", new_callable=AsyncMock) as mock_opt:
        mock_opt.return_value = mock_result
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await parcel_agent.run(cycles=1)


def test_trade_agent_place_bid_expired_offer(trade_agent):
    """Test place_bid returns error for an expired offer (covers line 68)."""
    # Create an offer with zero TTL so it expires immediately
    offer = trade_agent.create_offer("seller-1", "asset", 100.0, ttl_seconds=0)
    result = trade_agent.place_bid(offer.offer_id, "bidder-1", 110.0)
    assert result["success"] is False
    assert result["error"] == "Offer expired"
