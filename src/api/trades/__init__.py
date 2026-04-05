"""Trades API router — Web4AGI."""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.agents.parcel_agent import ParcelAgent
from src.agents.trade_agent import TradeAgent
from src.api.parcels import _AGENTS
from src.models.parcel_models import (
    BidRequest,
    OfferCreate,
    TradeRequest,
    TradeResponse,
)

router = APIRouter()

# Singleton TradeAgent for the platform
_TRADE_AGENT = TradeAgent(agent_id="platform-trade-agent")


@router.post("/", response_model=TradeResponse, status_code=201)
async def create_trade(request: TradeRequest) -> Any:
    """Execute a USDx transfer between two parcel agents."""
    from_agent: ParcelAgent | None = _AGENTS.get(request.from_parcel_id)
    if from_agent is None:
        raise HTTPException(
            status_code=404, detail=f"Source parcel '{request.from_parcel_id}' not found"
        )

    result = await from_agent.trade(
        counterparty_id=request.to_parcel_id,
        amount_usdx=request.amount_usdx,
        trade_type=request.trade_type,
        contract_terms=request.contract_terms,
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Trade failed"))

    return TradeResponse(
        success=True,
        tx_id=result.get("transaction_id"),
        amount_usdx=request.amount_usdx,
        from_parcel_id=request.from_parcel_id,
        to_parcel_id=request.to_parcel_id,
    )


@router.post("/offers", status_code=201)
async def create_offer(data: OfferCreate) -> Any:
    """Post a new trade offer (auction)."""
    offer = _TRADE_AGENT.create_offer(
        seller_id=data.seller_parcel_id,
        asset=data.asset,
        amount_usdx=data.amount_usdx,
        ttl_seconds=data.ttl_seconds,
    )
    return {
        "offer_id": offer.offer_id,
        "seller_parcel_id": offer.seller_id,
        "asset": offer.asset,
        "amount_usdx": offer.amount_usdx,
        "expires_at": offer.expires_at,
    }


@router.post("/offers/{offer_id}/bid")
async def place_bid(offer_id: str, data: BidRequest) -> Any:
    """Place a bid on an open trade offer."""
    result = _TRADE_AGENT.place_bid(
        offer_id=offer_id,
        bidder_id=data.bidder_parcel_id,
        bid_amount=data.bid_amount_usdx,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Bid failed"))
    return result


@router.post("/offers/{offer_id}/close")
async def close_offer(offer_id: str) -> Any:
    """Close a trade offer and settle with the highest bidder."""
    result = _TRADE_AGENT.close_offer(offer_id=offer_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Close failed"))
    return result


@router.get("/history")
async def trade_history(limit: int = 50) -> Any:
    """Return recent trade history."""
    return _TRADE_AGENT.get_history(limit=limit)
