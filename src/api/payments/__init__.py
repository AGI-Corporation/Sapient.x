"""Payments API router — Web4AGI."""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.parcels import _AGENTS
from src.models.parcel_models import DepositRequest, PaymentStreamRequest
from src.payments.x402_client import X402Client

router = APIRouter()

# Singleton x402 client for platform-level operations
_CLIENT = X402Client(local_only=True)


@router.post("/deposit")
async def deposit(data: DepositRequest) -> Any:
    """Deposit USDx into a parcel agent's wallet."""
    agent = _AGENTS.get(data.parcel_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Parcel '{data.parcel_id}' not found")

    result = await agent.deposit(amount_usdx=data.amount_usdx)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Deposit failed"))

    return {
        "parcel_id": data.parcel_id,
        "deposited_usdx": data.amount_usdx,
        "new_balance_usdx": agent.state.balance_usdx,
        "tx": result,
    }


@router.post("/stream")
async def open_payment_stream(data: PaymentStreamRequest) -> Any:
    """Open a real-time USDx payment stream between two parcels."""
    from_agent = _AGENTS.get(data.from_parcel_id)
    if from_agent is None:
        raise HTTPException(
            status_code=404, detail=f"Source parcel '{data.from_parcel_id}' not found"
        )

    result = await from_agent.x402.stream_payments(
        to_address=data.to_parcel_id,
        rate_usdx_per_second=data.rate_usdx_per_second,
        duration_seconds=data.duration_seconds,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Stream failed"))

    return {
        "from_parcel_id": data.from_parcel_id,
        "to_parcel_id": data.to_parcel_id,
        "rate_usdx_per_second": data.rate_usdx_per_second,
        "duration_seconds": data.duration_seconds,
        "total_usdx": data.rate_usdx_per_second * data.duration_seconds,
        "tx": result,
    }


@router.get("/balance/{address}")
async def get_balance(address: str) -> Any:
    """Query the USDx balance of a wallet address."""
    balance = await _CLIENT.get_balance(address)
    return {"address": address, "balance_usdx": balance}
