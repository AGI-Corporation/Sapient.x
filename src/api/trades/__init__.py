"""Trades API router — Web4AGI."""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.agents.trade_agent import TradeAgent
from src.models.parcel_models import BidRequest, ErrorResponse, OfferCreate, SuccessResponse

router = APIRouter()

_TRADE_AGENT = TradeAgent(agent_id="platform-trade-agent")


@router.post("/offers", status_code=201)
async def create_offer(body: OfferCreate) -> dict[str, Any]:
    """Create a new trade offer."""
    offer = _TRADE_AGENT.create_offer(
        seller_id=body.seller_parcel_id,
        asset=body.asset,
        amount_usdx=body.amount_usdx,
        ttl_seconds=body.ttl_seconds,
    )
    return {
        "offer_id": offer.offer_id,
        "seller_id": offer.seller_id,
        "asset": offer.asset,
        "amount_usdx": offer.amount_usdx,
        "expires_at": offer.expires_at,
    }


@router.get("/offers")
async def list_offers() -> list[dict[str, Any]]:
    """List all active (non-expired) trade offers."""
    return [
        {
            "offer_id": o.offer_id,
            "seller_id": o.seller_id,
            "asset": o.asset,
            "amount_usdx": o.amount_usdx,
            "bid_count": len(o.bids),
            "expired": o.is_expired(),
        }
        for o in _TRADE_AGENT.offers.values()
    ]


@router.post("/offers/{offer_id}/bid")
async def place_bid(offer_id: str, body: BidRequest) -> dict[str, Any]:
    """Place a bid on an offer."""
    result = _TRADE_AGENT.place_bid(
        offer_id=offer_id,
        bidder_id=body.bidder_parcel_id,
        bid_amount=body.bid_amount_usdx,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Bid failed"))
    return result


@router.post("/offers/{offer_id}/close")
async def close_offer(offer_id: str) -> dict[str, Any]:
    """Close an offer and select the winner."""
    result = _TRADE_AGENT.close_offer(offer_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Close failed"))
    return result


@router.get("/history")
async def get_trade_history(limit: int = 50) -> list[dict[str, Any]]:
    """Get the trade history."""
    return _TRADE_AGENT.get_history(limit=limit)


@router.get("/volume")
async def get_volume() -> dict[str, Any]:
    """Get total trade volume in USDx."""
    return {"volume_usdx": _TRADE_AGENT.volume_usdx()}
