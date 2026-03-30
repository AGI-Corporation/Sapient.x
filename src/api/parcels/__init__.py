"""Parcels API router — Web4AGI."""

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

from src.agents.parcel_agent import ParcelAgent
from src.models.parcel_models import (
    ErrorResponse,
    OptimizeRequest,
    OptimizeResponse,
    ParcelCreate,
    ParcelRead,
    ParcelUpdate,
    SuccessResponse,
    TradeRequest,
    TradeResponse,
)

router = APIRouter()

# In-memory store for this demo; production would use a DB.
_AGENTS: dict[str, ParcelAgent] = {}


def _get_agent(parcel_id: str) -> ParcelAgent:
    agent = _AGENTS.get(parcel_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Parcel '{parcel_id}' not found")
    return agent


def _agent_to_read(agent: ParcelAgent) -> dict[str, Any]:
    return agent.get_state()


@router.post("/", response_model=ParcelRead, status_code=201)
async def create_parcel(body: ParcelCreate) -> dict[str, Any]:
    """Create a new parcel agent."""
    parcel_id = str(uuid.uuid4())
    agent = ParcelAgent(
        parcel_id=parcel_id,
        owner_address=body.owner_address,
        location=body.location.model_dump(),
    )
    if body.metadata:
        for key, value in body.metadata.items():
            agent.update_metadata(key, value)
    _AGENTS[parcel_id] = agent
    return _agent_to_read(agent)


@router.get("/", response_model=list[ParcelRead])
async def list_parcels() -> list[dict[str, Any]]:
    """List all active parcel agents."""
    return [_agent_to_read(a) for a in _AGENTS.values()]


@router.get("/{parcel_id}", response_model=ParcelRead)
async def get_parcel(parcel_id: str) -> dict[str, Any]:
    """Get state for a specific parcel."""
    return _agent_to_read(_get_agent(parcel_id))


@router.patch("/{parcel_id}", response_model=ParcelRead)
async def update_parcel(parcel_id: str, body: ParcelUpdate) -> dict[str, Any]:
    """Update parcel metadata or active status."""
    agent = _get_agent(parcel_id)
    if body.metadata is not None:
        for key, value in body.metadata.items():
            agent.update_metadata(key, value)
    if body.active is not None:
        agent.state.active = body.active
    return _agent_to_read(agent)


@router.delete("/{parcel_id}", status_code=204)
async def delete_parcel(parcel_id: str) -> None:
    """Remove a parcel agent."""
    _get_agent(parcel_id)
    del _AGENTS[parcel_id]


@router.post("/{parcel_id}/deposit", response_model=SuccessResponse)
async def deposit(parcel_id: str, amount_usdx: float) -> dict[str, Any]:
    """Deposit USDx into a parcel wallet."""
    agent = _get_agent(parcel_id)
    result = await agent.deposit(amount_usdx=amount_usdx)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Deposit failed"))
    return {"success": True, "message": "Deposit successful", "data": result}


@router.post("/{parcel_id}/trade", response_model=TradeResponse)
async def trade(parcel_id: str, body: TradeRequest) -> dict[str, Any]:
    """Execute a USDx trade from this parcel to another."""
    agent = _get_agent(parcel_id)
    result = await agent.trade(
        counterparty_id=body.to_parcel_id,
        amount_usdx=body.amount_usdx,
        trade_type=body.trade_type,
        contract_terms=body.contract_terms,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Trade failed"))
    return {
        "success": True,
        "tx_id": result.get("transaction_id"),
        "amount_usdx": body.amount_usdx,
        "from_parcel_id": parcel_id,
        "to_parcel_id": body.to_parcel_id,
    }


@router.post("/{parcel_id}/optimize", response_model=OptimizeResponse)
async def optimize(parcel_id: str, body: OptimizeRequest) -> dict[str, Any]:
    """Run LangGraph optimization workflow for a parcel."""
    agent = _get_agent(parcel_id)
    result = await agent.optimize(context=body.context)
    return {
        "parcel_id": parcel_id,
        "assessment": result.get("assessment"),
        "strategies": result.get("strategies", []),
        "chosen_strategy": result.get("chosen_strategy"),
        "actions_taken": result.get("actions_taken", []),
        "reflection": result.get("reflection"),
        "score": result.get("score", 0.0),
    }


@router.post("/{parcel_id}/message", response_model=SuccessResponse)
async def send_message(parcel_id: str, to_parcel_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send an MCP message from this parcel to another."""
    agent = _get_agent(parcel_id)
    result = await agent.send_message(target_parcel_id=to_parcel_id, content=payload)
    return {"success": result.get("success", False), "message": "Message sent", "data": result}
