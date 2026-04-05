"""Parcels API router — Web4AGI."""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.agents.parcel_agent import ParcelAgent
from src.models.parcel_models import (
    OptimizeRequest,
    OptimizeResponse,
    ParcelCreate,
    ParcelRead,
    ParcelUpdate,
)

router = APIRouter()

# Module-level in-memory store: parcel_id -> ParcelAgent
_AGENTS: dict[str, ParcelAgent] = {}


def _agent_to_read(agent: ParcelAgent) -> ParcelRead:
    state = agent.get_state()
    return ParcelRead(
        parcel_id=state["parcel_id"],
        owner=state["owner"],
        location=state["location"],
        balance_usdx=state["balance_usdx"],
        metadata=state["metadata"],
        active=state["active"],
        last_updated=state["last_updated"],
    )


@router.post("/", response_model=ParcelRead, status_code=201)
async def create_parcel(data: ParcelCreate) -> Any:
    """Register a new parcel agent."""
    agent = ParcelAgent(
        owner_address=data.owner_address,
        location=data.location.model_dump(),
        wallet_private_key=None,
    )
    for key, value in data.metadata.items():
        agent.update_metadata(key, value)
    _AGENTS[agent.parcel_id] = agent
    return _agent_to_read(agent)


@router.get("/", response_model=list[ParcelRead])
async def list_parcels() -> Any:
    """List all registered parcels."""
    return [_agent_to_read(a) for a in _AGENTS.values()]


@router.get("/{parcel_id}", response_model=ParcelRead)
async def get_parcel(parcel_id: str) -> Any:
    """Retrieve a parcel by ID."""
    agent = _AGENTS.get(parcel_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Parcel '{parcel_id}' not found")
    return _agent_to_read(agent)


@router.patch("/{parcel_id}", response_model=ParcelRead)
async def update_parcel(parcel_id: str, data: ParcelUpdate) -> Any:
    """Update parcel metadata or active flag."""
    agent = _AGENTS.get(parcel_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Parcel '{parcel_id}' not found")
    if data.metadata is not None:
        for key, value in data.metadata.items():
            agent.update_metadata(key, value)
    if data.active is not None:
        agent.state.active = data.active
    return _agent_to_read(agent)


@router.delete("/{parcel_id}", status_code=204)
async def delete_parcel(parcel_id: str) -> None:
    """Deregister a parcel agent."""
    if parcel_id not in _AGENTS:
        raise HTTPException(status_code=404, detail=f"Parcel '{parcel_id}' not found")
    del _AGENTS[parcel_id]


@router.post("/{parcel_id}/optimize", response_model=OptimizeResponse)
async def optimize_parcel(parcel_id: str, request: OptimizeRequest) -> Any:
    """Run the LangGraph optimization workflow for a parcel."""
    agent = _AGENTS.get(parcel_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Parcel '{parcel_id}' not found")
    result = await agent.optimize(context=request.context)
    return OptimizeResponse(
        parcel_id=parcel_id,
        assessment=result.get("assessment"),
        strategies=result.get("strategies", []),
        chosen_strategy=result.get("chosen_strategy"),
        actions_taken=result.get("actions_taken", []),
        reflection=result.get("reflection"),
        score=result.get("score", 0.0),
    )
