"""Payments API router — Web4AGI."""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException

from src.payments.x402_client import X402Client, make_x402_client
from src.models.parcel_models import DepositRequest, PaymentStreamRequest

router = APIRouter()


def _client() -> X402Client:
    return make_x402_client()


@router.post("/deposit")
async def deposit(body: DepositRequest) -> Any:
    client = _client()
    result = await client.deposit(amount=body.amount_usdx, source=body.source)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Deposit failed"))
    return result


@router.post("/transfer")
async def transfer(body: Dict[str, Any]) -> Any:
    client = _client()
    result = await client.transfer(
        to_address=body.get("to_address", ""),
        amount=float(body.get("amount_usdx", 0)),
        memo=body.get("memo", ""),
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Transfer failed"))
    return result


@router.post("/stream")
async def stream_payment(body: PaymentStreamRequest) -> Any:
    client = _client()
    result = await client.stream_payments(
        to_address=body.to_parcel_id,
        rate_usdx_per_second=body.rate_usdx_per_second,
        duration_seconds=body.duration_seconds,
    )
    return result


@router.get("/balance/{address}")
async def get_balance(address: str) -> Any:
    client = _client()
    try:
        client.validate_address(address)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    balance = await client.get_balance(address)
    return {"address": address, "balance_usdx": balance}
