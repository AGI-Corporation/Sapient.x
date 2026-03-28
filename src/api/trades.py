"""Trades API router — Web4AGI."""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException

from src.agents.trade_agent import TradeAgent
from src.models.parcel_models import (
    OfferCreate,
    BidRequest,
    TradeRequest,
    TradeResponse,
)

router = APIRouter()

_TRADE_AGENTS: Dict[str, TradeAgent] = {}
_DEFAULT_AGENT_ID = "global-trade-agent"


def _get_agent() -> TradeAgent:
    if _DEFAULT_AGENT_ID not in _TRADE_AGENTS:
        _TRADE_AGENTS[_DEFAULT_AGENT_ID] = TradeAgent(agent_id=_DEFAULT_AGENT_ID)
    return _TRADE_AGENTS[_DEFAULT_AGENT_ID]


@router.get("")
async def list_trades() -> Any:
    return _get_agent().get_history()


@router.post("/offers", status_code=201)
async def create_offer(body: OfferCreate) -> Any:
    agent = _get_agent()
    offer = agent.create_offer(
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
        "bids": offer.bids,
    }


@router.post("/bids")
async def place_bid(body: BidRequest) -> Any:
    result = _get_agent().place_bid(
        offer_id=body.offer_id,
        bidder_id=body.bidder_parcel_id,
        bid_amount=body.bid_amount_usdx,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Bid failed"))
    return result


@router.post("/offers/{offer_id}/close")
async def close_offer(offer_id: str) -> Any:
    result = _get_agent().close_offer(offer_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Close failed"))
    return result


@router.get("/history")
async def trade_history(limit: int = 50) -> Any:
    return _get_agent().get_history(limit=limit)
