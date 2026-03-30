"""Payments API router — Web4AGI."""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.models.parcel_models import DepositRequest, PaymentStreamRequest, SuccessResponse
from src.payments.x402_client import X402Client

router = APIRouter()

_CLIENT = X402Client()


@router.get("/balance/{address}")
async def get_balance(address: str) -> dict[str, Any]:
    """Get the USDx balance for a wallet address."""
    try:
        _CLIENT.validate_address(address)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    balance = await _CLIENT.get_balance(address)
    return {"address": address, "balance_usdx": balance}


@router.post("/transfer")
async def transfer(
    to_address: str,
    amount_usdx: float,
    memo: str = "",
) -> dict[str, Any]:
    """Transfer USDx to an address via the x402 protocol."""
    if amount_usdx <= 0:
        raise HTTPException(status_code=400, detail="Transfer amount must be positive")
    result = await _CLIENT.transfer(to_address=to_address, amount=amount_usdx, memo=memo)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Transfer failed"))
    return result


@router.post("/stream")
async def open_payment_stream(body: PaymentStreamRequest) -> dict[str, Any]:
    """Open a USDx payment stream (real-time parcel rent)."""
    result = await _CLIENT.stream_payments(
        to_address=body.to_parcel_id,
        rate_usdx_per_second=body.rate_usdx_per_second,
        duration_seconds=body.duration_seconds,
    )
    return result


@router.get("/address")
async def get_wallet_address() -> dict[str, Any]:
    """Return the platform wallet address."""
    return {"address": _CLIENT.get_address()}
