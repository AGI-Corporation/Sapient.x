"""Parcel API router — Web4AGI."""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException

from src.agents.parcel_agent import ParcelAgent
from src.models.parcel_models import ParcelCreate, ParcelRead, ParcelUpdate, SuccessResponse

router = APIRouter()

# In-memory store for demo purposes; replace with a proper DB in production.
_PARCELS: Dict[str, ParcelAgent] = {}


@router.post("", response_model=ParcelRead, status_code=201)
async def create_parcel(body: ParcelCreate) -> Any:
    agent = ParcelAgent(
        owner_address=body.owner_address,
        location=body.location.model_dump(),
    )
    agent.update_metadata("metadata", body.metadata)
    _PARCELS[agent.parcel_id] = agent
    return _agent_to_read(agent)


@router.get("/{parcel_id}", response_model=ParcelRead)
async def get_parcel(parcel_id: str) -> Any:
    agent = _get_or_404(parcel_id)
    return _agent_to_read(agent)


@router.get("", response_model=list)
async def list_parcels() -> Any:
    return [_agent_to_read(a) for a in _PARCELS.values()]


@router.patch("/{parcel_id}", response_model=ParcelRead)
async def update_parcel(parcel_id: str, body: ParcelUpdate) -> Any:
    agent = _get_or_404(parcel_id)
    if body.metadata is not None:
        for k, v in body.metadata.items():
            agent.update_metadata(k, v)
    if body.active is not None:
        agent.state.active = body.active
    return _agent_to_read(agent)


@router.delete("/{parcel_id}", status_code=204)
async def delete_parcel(parcel_id: str) -> None:
    _get_or_404(parcel_id)
    del _PARCELS[parcel_id]


@router.post("/{parcel_id}/optimize")
async def optimize_parcel(parcel_id: str) -> Any:
    agent = _get_or_404(parcel_id)
    result = await agent.optimize()
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_or_404(parcel_id: str) -> ParcelAgent:
    agent = _PARCELS.get(parcel_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Parcel '{parcel_id}' not found")
    return agent


def _agent_to_read(agent: ParcelAgent) -> Dict:
    state = agent.get_state()
    return {
        "parcel_id": state["parcel_id"],
        "owner": state["owner"],
        "location": state["location"],
        "balance_usdx": state["balance_usdx"],
        "metadata": state["metadata"],
        "active": state["active"],
        "last_updated": state["last_updated"],
    }
