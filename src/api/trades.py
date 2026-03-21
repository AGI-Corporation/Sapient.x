"""Trades API Router — Web4AGI."""

from fastapi import APIRouter, HTTPException

from src.agents.trade_agent import TradeAgent
from src.models.parcel_models import (
    BidRequest,
    IncentiveRequest,
    OfferCreate,
    SuccessResponse,
    TradeRequest,
)

router = APIRouter()


@router.post("/offers", status_code=201)
async def create_offer(offer: OfferCreate):
    """Create a new trade offer via TradeAgent."""
    from src.main import PARCEL_AGENTS, TRADE_AGENTS

    if offer.seller_parcel_id not in PARCEL_AGENTS:
        raise HTTPException(status_code=404, detail="Seller parcel agent not found")

    # Use a global trade agent or create one if not exists
    if "global" not in TRADE_AGENTS:
        TRADE_AGENTS["global"] = TradeAgent(agent_id="global-trade-orchestrator")

    trade_agent = TRADE_AGENTS["global"]
    new_offer = trade_agent.create_offer(
        seller_id=offer.seller_parcel_id,
        asset=offer.asset,
        amount_usdx=offer.amount_usdx,
        ttl_seconds=offer.ttl_seconds,
    )

    return {
        "offer_id": new_offer.offer_id,
        "seller_id": new_offer.seller_id,
        "asset": new_offer.asset,
        "amount": new_offer.amount_usdx,
        "expires_at": new_offer.expires_at,
    }


@router.post("/bids")
async def place_bid(bid: BidRequest):
    """Place a bid on an active offer."""
    from src.main import PARCEL_AGENTS, TRADE_AGENTS

    if bid.bidder_parcel_id not in PARCEL_AGENTS:
        raise HTTPException(status_code=404, detail="Bidder parcel agent not found")

    trade_agent = TRADE_AGENTS.get("global")
    if not trade_agent:
        raise HTTPException(status_code=404, detail="Trade orchestrator not initialized")

    result = trade_agent.place_bid(
        offer_id=bid.offer_id, bidder_id=bid.bidder_parcel_id, bid_amount=bid.bid_amount_usdx
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/close/{offer_id}")
async def close_offer(offer_id: str):
    """Close an offer and select winner."""
    from src.main import TRADE_AGENTS

    trade_agent = TRADE_AGENTS.get("global")
    if not trade_agent:
        raise HTTPException(status_code=404, detail="Trade orchestrator not initialized")

    result = await trade_agent.close_offer(offer_id)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/settle/{offer_id}")
async def settle_trade(offer_id: str):
    """Execute settlement for a closed trade."""
    from src.main import TRADE_AGENTS

    trade_agent = TRADE_AGENTS.get("global")
    if not trade_agent:
        raise HTTPException(status_code=404, detail="Trade orchestrator not initialized")

    result = await trade_agent.settle_trade(offer_id)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/incentive", response_model=SuccessResponse)
async def execute_incentive(request: IncentiveRequest):
    """Execute a USDx incentive payment (e.g., for loyalty or community check-in)."""
    from src.main import PARCEL_AGENTS

    if request.parcel_id not in PARCEL_AGENTS:
        raise HTTPException(status_code=404, detail="Sponsoring parcel agent not found")

    sponsor = PARCEL_AGENTS[request.parcel_id]

    # Standard trade wrapping the incentive
    result = await sponsor.trade(
        counterparty_id=request.target_parcel_id,
        amount_usdx=request.amount_usdx,
        trade_type=f"incentive:{request.incentive_type}",
        contract_terms=request.metadata,
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Incentive failed"))

    return SuccessResponse(message=f"Incentive {request.incentive_type} executed", data=result)


@router.post("/direct", response_model=SuccessResponse)
async def direct_trade(request: TradeRequest):
    """Execute a direct trade between two parcels."""
    from src.main import PARCEL_AGENTS

    if request.from_parcel_id not in PARCEL_AGENTS:
        raise HTTPException(status_code=404, detail="Sender parcel not found")

    sender = PARCEL_AGENTS[request.from_parcel_id]
    result = await sender.trade(
        counterparty_id=request.to_parcel_id,
        amount_usdx=request.amount_usdx,
        trade_type=request.trade_type,
        contract_terms=request.contract_terms,
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Trade failed"))

    return SuccessResponse(message="Direct trade executed successfully", data=result)
