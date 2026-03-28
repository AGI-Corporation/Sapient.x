"""Agents API router — Web4AGI (/api/agents)."""

import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from datetime import datetime

router = APIRouter()

# Simple in-memory store
_AGENTS: Dict[str, Dict] = {}


@router.post("", status_code=201)
async def create_agent(body: Dict[str, Any]) -> Any:
    parcel_id = body.get("parcel_id", "")
    if not parcel_id:
        raise HTTPException(status_code=422, detail="parcel_id is required")
    # Import inside function so @patch('src.agents.parcel_agent.ParcelAgent') applies
    from src.agents.parcel_agent import ParcelAgent
    agent = ParcelAgent(
        parcel_id=parcel_id,
        owner_address=body.get("wallet_address", "0x0000000000000000000000000000000000000000"),
    )
    # Prefer mock's .id attribute; fall back to real agent's parcel_id
    agent_id = str(getattr(agent, "id", None) or getattr(agent, "parcel_id", parcel_id))
    record = {
        "id": agent_id,
        "parcel_id": str(getattr(agent, "parcel_id", parcel_id)),
        "model": body.get("model", "gpt-4"),
        "status": "active",
        "balance": body.get("initial_balance", 0.0),
        "config": body.get("config", {}),
        "created_at": datetime.utcnow().isoformat(),
    }
    _AGENTS[agent_id] = record
    return record


@router.get("")
async def list_agents() -> Any:
    return list(_AGENTS.values())


@router.get("/{agent_id}")
async def get_agent(agent_id: str) -> Any:
    record = _AGENTS.get(agent_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return record


@router.patch("/{agent_id}")
async def update_agent(agent_id: str, body: Dict[str, Any]) -> Any:
    record = _AGENTS.get(agent_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    record.update({k: v for k, v in body.items() if k in ("status", "config", "model")})
    return record


@router.delete("/{agent_id}", status_code=204)
async def delete_agent(agent_id: str) -> None:
    if agent_id not in _AGENTS:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    del _AGENTS[agent_id]
