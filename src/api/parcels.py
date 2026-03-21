"""Parcels API Router — Web4AGI."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from src.agents.parcel_agent import ParcelAgent
from src.models.parcel_models import (
    OptimizeRequest,
    OptimizeResponse,
    ParcelCreate,
    ParcelRead,
    ParcelUpdate,
    SuccessResponse,
)

router = APIRouter()


@router.post("/", response_model=ParcelRead, status_code=201)
async def create_parcel(parcel: ParcelCreate):
    """Register a new parcel agent."""
    from src.core.state import PARCEL_AGENTS

    parcel_id = f"parcel-{uuid.uuid4().hex[:8]}"

    # Initialize agent
    agent = ParcelAgent(
        parcel_id=parcel_id,
        owner_address=parcel.owner_address,
        location=parcel.location.model_dump(),
    )
    agent.state.metadata = parcel.metadata

    PARCEL_AGENTS[parcel_id] = agent

    return ParcelRead(
        parcel_id=parcel_id,
        owner=agent.owner_address,
        location=parcel.location,
        balance_usdx=agent.state.balance_usdx,
        metadata=agent.state.metadata,
        active=agent.state.active,
        last_updated=agent.state.last_updated,
    )


@router.get("/{parcel_id}", response_model=ParcelRead)
async def get_parcel(parcel_id: str):
    """Retrieve parcel agent state."""
    from src.core.state import PARCEL_AGENTS

    if parcel_id not in PARCEL_AGENTS:
        raise HTTPException(status_code=404, detail="Parcel agent not found")

    agent = PARCEL_AGENTS[parcel_id]
    state = agent.get_state()

    return ParcelRead(
        parcel_id=parcel_id,
        owner=state["owner"],
        location=state["location"],
        balance_usdx=state["balance_usdx"],
        metadata=state["metadata"],
        active=state["active"],
        last_updated=state["last_updated"],
    )


@router.patch("/{parcel_id}", response_model=ParcelRead)
async def update_parcel(parcel_id: str, update: ParcelUpdate):
    """Update parcel metadata or status."""
    from src.core.state import PARCEL_AGENTS

    if parcel_id not in PARCEL_AGENTS:
        raise HTTPException(status_code=404, detail="Parcel agent not found")

    agent = PARCEL_AGENTS[parcel_id]

    if update.metadata is not None:
        agent.state.metadata.update(update.metadata)
    if update.active is not None:
        agent.state.active = update.active

    agent.state.last_updated = datetime.now(timezone.utc).isoformat()

    state = agent.get_state()
    return ParcelRead(
        parcel_id=parcel_id,
        owner=state["owner"],
        location=state["location"],
        balance_usdx=state["balance_usdx"],
        metadata=state["metadata"],
        active=state["active"],
        last_updated=state["last_updated"],
    )


@router.post("/optimize", response_model=OptimizeResponse)
async def trigger_optimization(request: OptimizeRequest):
    """Run LangGraph optimization for a parcel."""
    from src.core.state import PARCEL_AGENTS

    if request.parcel_id not in PARCEL_AGENTS:
        raise HTTPException(status_code=404, detail="Parcel agent not found")

    agent = PARCEL_AGENTS[request.parcel_id]
    result = await agent.optimize(context=request.context)

    return OptimizeResponse(
        parcel_id=request.parcel_id,
        assessment=result.get("assessment"),
        strategies=result.get("strategies", []),
        chosen_strategy=result.get("chosen_strategy"),
        actions_taken=result.get("actions_taken", []),
        reflection=result.get("reflection"),
        score=result.get("score", 0.0),
    )


@router.delete("/{parcel_id}", response_model=SuccessResponse)
async def delete_parcel(parcel_id: str):
    """Deactivate and remove a parcel agent."""
    from src.core.state import PARCEL_AGENTS

    if parcel_id not in PARCEL_AGENTS:
        raise HTTPException(status_code=404, detail="Parcel agent not found")

    del PARCEL_AGENTS[parcel_id]
    return SuccessResponse(message=f"Parcel agent {parcel_id} removed")
