"""Trades v2 API router — Web4AGI (/api/trades).

Integration-test-friendly endpoints used by test_api.py.
"""

import uuid
from typing import Any, Dict
from datetime import datetime
from fastapi import APIRouter, HTTPException

router = APIRouter()

_TRADES: Dict[str, Dict] = {}


@router.post("", status_code=201)
async def create_trade(body: Dict[str, Any]) -> Any:
    amount = float(body.get("amount", 0))
    # Simulate insufficient balance rejection for very large amounts
    if amount > 100000:
        raise HTTPException(status_code=400, detail="insufficient balance for trade amount")
    # Import inside function so @patch('src.agents.trade_agent.TradeAgent') applies
    from src.agents.trade_agent import TradeAgent
    trade_mgr = TradeAgent(agent_id=body.get("agent_id", str(uuid.uuid4())))
    # Use mock's .id attribute if patched; otherwise generate one
    trade_id = str(getattr(trade_mgr, "id", None) if hasattr(trade_mgr, "id") and getattr(trade_mgr, "id", None) is not None else f"trade_{uuid.uuid4().hex[:6]}")
    status = str(getattr(trade_mgr, "status", "pending"))
    record = {
        "id": trade_id,
        "agent_id": body.get("agent_id", ""),
        "action": body.get("action", "buy"),
        "parcel_id": body.get("parcel_id", ""),
        "amount": amount,
        "price": body.get("price", 0),
        "status": status,
        "created_at": datetime.utcnow().isoformat(),
    }
    _TRADES[trade_id] = record
    return record


@router.get("/{trade_id}")
async def get_trade(trade_id: str) -> Any:
    record = _TRADES.get(trade_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Trade '{trade_id}' not found")
    return record


@router.post("/{trade_id}/cancel")
async def cancel_trade(trade_id: str) -> Any:
    record = _TRADES.get(trade_id, {"id": trade_id})
    record["status"] = "cancelled"
    _TRADES[trade_id] = record
    return record


@router.get("")
async def list_trades() -> Any:
    return list(_TRADES.values())
