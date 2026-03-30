"""Unit tests for TradeAgent edge cases and uncovered paths."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.trade_agent import TradeAgent, TradeOffer


@pytest.fixture
def agent():
    return TradeAgent(agent_id="ext-trade-agent")


# ── TradeOffer ─────────────────────────────────────────────────────────────


def test_trade_offer_is_expired_when_ttl_zero():
    """Offer with ttl_seconds=0 expires immediately."""
    offer = TradeOffer("o-1", "seller-1", "asset-x", 100.0, ttl_seconds=0)
    assert offer.is_expired() is True


def test_trade_offer_is_not_expired_with_long_ttl():
    offer = TradeOffer("o-2", "seller-2", "asset-y", 50.0, ttl_seconds=9999)
    assert offer.is_expired() is False


def test_trade_offer_best_bid_none_when_no_bids():
    offer = TradeOffer("o-3", "seller-3", "asset-z", 10.0)
    assert offer.best_bid() is None


def test_trade_offer_best_bid_returns_maximum():
    offer = TradeOffer("o-4", "seller-4", "asset-w", 10.0)
    offer.add_bid("bidder-a", 10.0)
    offer.add_bid("bidder-b", 20.0)
    offer.add_bid("bidder-c", 15.0)
    best = offer.best_bid()
    assert best["bidder"] == "bidder-b"
    assert best["amount"] == 20.0


def test_trade_offer_add_bid_stores_timestamp():
    offer = TradeOffer("o-5", "seller-5", "asset", 10.0)
    offer.add_bid("b1", 11.0)
    assert "ts" in offer.bids[0]


# ── place_bid edge cases ───────────────────────────────────────────────────


def test_place_bid_offer_not_found(agent):
    result = agent.place_bid(offer_id="nonexistent", bidder_id="b1", bid_amount=50.0)
    assert result["success"] is False
    assert "not found" in result["error"].lower()


def test_place_bid_expired_offer(agent):
    offer = agent.create_offer("seller-1", "asset", 100.0, ttl_seconds=0)
    result = agent.place_bid(offer_id=offer.offer_id, bidder_id="b1", bid_amount=105.0)
    assert result["success"] is False
    assert "expired" in result["error"].lower()


# ── close_offer edge cases ─────────────────────────────────────────────────


def test_close_offer_not_found(agent):
    result = agent.close_offer(offer_id="no-such-offer")
    assert result["success"] is False
    assert "not found" in result["error"].lower()


def test_close_offer_no_bids(agent):
    offer = agent.create_offer("seller-1", "asset", 100.0)
    result = agent.close_offer(offer_id=offer.offer_id)
    assert result["success"] is False
    assert "no bids" in result["error"].lower()


def test_close_offer_records_winner(agent):
    offer = agent.create_offer("seller", "asset", 100.0)
    agent.place_bid(offer.offer_id, "winner", 150.0)
    result = agent.close_offer(offer.offer_id)
    assert result["success"] is True
    assert offer.accepted is not None
    assert offer.accepted["bidder"] == "winner"


# ── get_history ────────────────────────────────────────────────────────────


def test_get_history_with_limit(agent):
    """get_history respects the limit parameter."""
    for i in range(10):
        offer = agent.create_offer(f"seller-{i}", f"asset-{i}", float(i + 10))
        agent.place_bid(offer.offer_id, f"buyer-{i}", float(i + 11))
        agent.close_offer(offer.offer_id)

    history = agent.get_history(limit=3)
    assert len(history) == 3


def test_get_history_default_limit(agent):
    """Default limit is 50."""
    for i in range(60):
        offer = agent.create_offer(f"s-{i}", f"a-{i}", float(i + 1))
        agent.place_bid(offer.offer_id, f"b-{i}", float(i + 2))
        agent.close_offer(offer.offer_id)

    history = agent.get_history()
    assert len(history) == 50


def test_get_history_empty(agent):
    assert agent.get_history() == []


# ── volume_usdx ────────────────────────────────────────────────────────────


def test_volume_usdx_empty(agent):
    assert agent.volume_usdx() == 0.0


# ── batch_transfer ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_batch_transfer_success(agent):
    """batch_transfer sends transfers concurrently and collects results."""
    sender = AsyncMock()
    sender.trade.return_value = {"success": True, "amount": 10.0}

    recipients = [
        {"parcel_id": "p1", "amount": 10.0},
        {"parcel_id": "p2", "amount": 20.0},
        {"parcel_id": "p3", "amount": 30.0},
    ]
    results = await agent.batch_transfer(sender, recipients)
    assert len(results) == 3
    assert all(r["success"] for r in results)
    assert sender.trade.call_count == 3


@pytest.mark.asyncio
async def test_batch_transfer_handles_exceptions(agent):
    """Exceptions in individual transfers are captured as error dicts."""
    sender = AsyncMock()
    sender.trade.side_effect = [
        {"success": True},
        Exception("Network error"),
        {"success": True},
    ]

    recipients = [
        {"parcel_id": "p1", "amount": 5.0},
        {"parcel_id": "p2", "amount": 10.0},
        {"parcel_id": "p3", "amount": 15.0},
    ]
    results = await agent.batch_transfer(sender, recipients)
    assert len(results) == 3
    assert results[0]["success"] is True
    assert results[1]["success"] is False
    assert "Network error" in results[1]["error"]


@pytest.mark.asyncio
async def test_batch_transfer_empty_recipients(agent):
    sender = AsyncMock()
    results = await agent.batch_transfer(sender, [])
    assert results == []
    sender.trade.assert_not_called()


# ── contract templates ─────────────────────────────────────────────────────


def test_parcel_lease_contract_total_calculation(agent):
    contract = TradeAgent.parcel_lease_contract(
        lessor_id="0xL",
        lessee_id="0xE",
        parcel_id="p-1",
        monthly_usdx=150.0,
        duration_months=12,
    )
    assert contract["terms"]["total_usdx"] == 1800.0
    assert contract["status"] == "pending_signatures"
    assert contract["version"] == "1.0"


def test_data_access_contract_structure(agent):
    contract = TradeAgent.data_access_contract(
        provider_id="0xP",
        consumer_id="0xC",
        dataset="analytics",
        price_usdx=75.0,
    )
    assert contract["type"] == "data_access"
    assert contract["terms"]["access_type"] == "read"
    assert contract["terms"]["duration"] == "perpetual"
