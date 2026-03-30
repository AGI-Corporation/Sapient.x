"""Trades API router — Web4AGI."""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.agents.trade_agent import TradeAgent
from src.models.parcel_models import (
    BidRequest,
    ErrorResponse,
    OfferCreate,
    SuccessResponse,
    TradeRequest,
    TradeResponse,
)

router = APIRouter()

_trade_agent = TradeAgent(agent_id="global-trade-agent")

# In-memory parcel store (shared with parcels router in real app; mocked here)
_parcel_agents: dict[str, Any] = {}


def _register_parcel(agent: Any) -> None:
    _parcel_agents[agent.parcel_id] = agent


@router.post("", response_model=TradeResponse, status_code=201)
async def create_trade(body: TradeRequest) -> dict[str, Any]:
    from src.api.parcels import _parcels  # lazy import to avoid circular deps

    sender = _parcels.get(body.from_parcel_id)
    if sender is None:
        raise HTTPException(status_code=404, detail=f"Parcel '{body.from_parcel_id}' not found")

    result = await sender.trade(
        counterparty_id=body.to_parcel_id,
        amount_usdx=body.amount_usdx,
        trade_type=body.trade_type,
        contract_terms=body.contract_terms,
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Trade failed"))

    return {
        "success": True,
        "tx_id": result.get("transaction_id"),
        "amount_usdx": body.amount_usdx,
        "from_parcel_id": body.from_parcel_id,
        "to_parcel_id": body.to_parcel_id,
    }


# ── Offer / Auction endpoints ─────────────────────────────────────────────────


@router.post("/offers", status_code=201)
async def create_offer(body: OfferCreate) -> dict[str, Any]:
    offer = _trade_agent.create_offer(
        seller_id=body.seller_parcel_id,
        asset=body.asset,
        amount_usdx=body.amount_usdx,
        ttl_seconds=body.ttl_seconds,
    )
    return {
        "offer_id": offer.offer_id,
        "seller_parcel_id": offer.seller_id,
        "asset": offer.asset,
        "amount_usdx": offer.amount_usdx,
        "expires_at": offer.expires_at,
        "status": "open",
    }


@router.get("/offers")
async def list_offers() -> list[dict[str, Any]]:
    return [
        {
            "offer_id": o.offer_id,
            "seller_parcel_id": o.seller_id,
            "asset": o.asset,
            "amount_usdx": o.amount_usdx,
            "bids": len(o.bids),
            "expired": o.is_expired(),
        }
        for o in _trade_agent.offers.values()
    ]


@router.post("/offers/{offer_id}/bid")
async def place_bid(offer_id: str, body: BidRequest) -> dict[str, Any]:
    result = _trade_agent.place_bid(offer_id, body.bidder_parcel_id, body.bid_amount_usdx)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Bid failed"))
    return result


@router.post("/offers/{offer_id}/close")
async def close_offer(offer_id: str) -> dict[str, Any]:
    result = _trade_agent.close_offer(offer_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Close failed"))
    return result


@router.get("/history")
async def trade_history(limit: int = 50) -> dict[str, Any]:
    return {
        "history": _trade_agent.get_history(limit),
        "total_volume_usdx": _trade_agent.volume_usdx(),
    }
