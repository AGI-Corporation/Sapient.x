"""Parcels API router — Web4AGI."""

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
)

router = APIRouter()

# In-memory store; replace with DB in production
_parcels: dict[str, ParcelAgent] = {}


def _agent_to_read(agent: ParcelAgent) -> dict[str, Any]:
    s = agent.get_state()
    return {
        "parcel_id": s["parcel_id"],
        "owner": s["owner"],
        "location": s["location"],
        "balance_usdx": s["balance_usdx"],
        "metadata": s["metadata"],
        "active": s["active"],
        "last_updated": s["last_updated"],
    }


@router.post("", response_model=ParcelRead, status_code=201)
async def create_parcel(body: ParcelCreate) -> dict[str, Any]:
    agent = ParcelAgent(
        owner_address=body.owner_address,
        location=body.location.model_dump(),
        wallet_private_key=None,
    )
    agent.state.metadata = body.metadata
    _parcels[agent.parcel_id] = agent
    return _agent_to_read(agent)


@router.get("", response_model=list[ParcelRead])
async def list_parcels() -> list[dict[str, Any]]:
    return [_agent_to_read(a) for a in _parcels.values()]


@router.get("/{parcel_id}", response_model=ParcelRead)
async def get_parcel(parcel_id: str) -> dict[str, Any]:
    agent = _parcels.get(parcel_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Parcel '{parcel_id}' not found")
    return _agent_to_read(agent)


@router.patch("/{parcel_id}", response_model=ParcelRead)
async def update_parcel(parcel_id: str, body: ParcelUpdate) -> dict[str, Any]:
    agent = _parcels.get(parcel_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Parcel '{parcel_id}' not found")
    if body.metadata is not None:
        agent.state.metadata.update(body.metadata)
    if body.active is not None:
        agent.state.active = body.active
    return _agent_to_read(agent)


@router.delete("/{parcel_id}", status_code=204)
async def delete_parcel(parcel_id: str) -> None:
    if parcel_id not in _parcels:
        raise HTTPException(status_code=404, detail=f"Parcel '{parcel_id}' not found")
    del _parcels[parcel_id]


@router.post("/{parcel_id}/optimize", response_model=OptimizeResponse)
async def optimize_parcel(parcel_id: str, body: OptimizeRequest) -> dict[str, Any]:
    agent = _parcels.get(parcel_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Parcel '{parcel_id}' not found")
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
