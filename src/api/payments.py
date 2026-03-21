"""Payments API Router — Web4AGI."""

from fastapi import APIRouter, HTTPException

from src.models.parcel_models import DepositRequest, SuccessResponse, TradeRequest

router = APIRouter()


@router.get("/balance/{parcel_id}")
async def get_balance(parcel_id: str):
    """Get the USDx balance for a parcel agent."""
    from src.core.state import PARCEL_AGENTS

    if parcel_id not in PARCEL_AGENTS:
        raise HTTPException(status_code=404, detail="Parcel agent not found")

    agent = PARCEL_AGENTS[parcel_id]
    return {
        "parcel_id": parcel_id,
        "balance_usdx": agent.state.balance_usdx,
        "owner_address": agent.owner_address,
    }


@router.post("/deposit", response_model=SuccessResponse)
async def deposit(request: DepositRequest):
    """Deposit USDx into a parcel wallet."""
    from src.core.state import PARCEL_AGENTS

    if request.parcel_id not in PARCEL_AGENTS:
        raise HTTPException(status_code=404, detail="Parcel agent not found")

    agent = PARCEL_AGENTS[request.parcel_id]
    result = await agent.deposit(amount_usdx=request.amount_usdx)

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Deposit failed"))

    return SuccessResponse(message="Deposit successful", data=result)


@router.post("/transfer", response_model=SuccessResponse)
async def transfer(request: TradeRequest):
    """Transfer USDx between parcel agents."""
    from src.core.state import PARCEL_AGENTS

    if request.from_parcel_id not in PARCEL_AGENTS:
        raise HTTPException(status_code=404, detail="Sender parcel not found")

    sender = PARCEL_AGENTS[request.from_parcel_id]
    result = await sender.trade(
        counterparty_id=request.to_parcel_id,
        amount_usdx=request.amount_usdx,
        trade_type="transfer",
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Transfer failed"))

    return SuccessResponse(message="Transfer successful", data=result)
