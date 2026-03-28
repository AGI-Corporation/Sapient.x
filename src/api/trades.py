"""Trades router — Web4AGI API."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_trades():
    return {"trades": []}


@router.get("/{trade_id}")
async def get_trade(trade_id: str):
    return {"trade_id": trade_id}
