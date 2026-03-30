"""Payments API router — Web4AGI."""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.models.parcel_models import DepositRequest, PaymentStreamRequest
from src.payments.x402_client import X402Client

router = APIRouter()

_x402 = X402Client()


@router.post("/deposit")
async def deposit(body: DepositRequest) -> dict[str, Any]:
    from src.api.parcels import _parcels  # lazy import

    agent = _parcels.get(body.parcel_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Parcel '{body.parcel_id}' not found")
    result = await agent.deposit(amount_usdx=body.amount_usdx)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Deposit failed"))
    return {
        "success": True,
        "parcel_id": body.parcel_id,
        "amount_usdx": body.amount_usdx,
        "new_balance_usdx": agent.state.balance_usdx,
    }


@router.get("/balance/{address}")
async def get_balance(address: str) -> dict[str, Any]:
    balance = await _x402.get_balance(address)
    return {"address": address, "balance_usdx": balance}


@router.post("/stream")
async def open_payment_stream(body: PaymentStreamRequest) -> dict[str, Any]:
    from src.api.parcels import _parcels  # lazy import

    sender = _parcels.get(body.from_parcel_id)
    if sender is None:
        raise HTTPException(status_code=404, detail=f"Parcel '{body.from_parcel_id}' not found")

    result = await sender.x402.stream_payments(
        to_address=body.to_parcel_id,
        rate_usdx_per_second=body.rate_usdx_per_second,
        duration_seconds=body.duration_seconds,
    )
    return {
        "success": result.get("success", True),
        "from_parcel_id": body.from_parcel_id,
        "to_parcel_id": body.to_parcel_id,
        "rate_usdx_per_second": body.rate_usdx_per_second,
        "duration_seconds": body.duration_seconds,
        "total_usdx": body.rate_usdx_per_second * body.duration_seconds,
    }


@router.get("/history/{address}")
async def payment_history(address: str) -> dict[str, Any]:
    history = await _x402.get_transaction_history(address)
    return {"address": address, "history": history}
