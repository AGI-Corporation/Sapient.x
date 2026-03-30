"""Agents API router — Web4AGI.

CRUD operations for parcel agents exposed at /api/v1/agents.
"""

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

from src.agents.parcel_agent import ParcelAgent

router = APIRouter()

# In-memory registry of running agents; keyed by agent_id (≠ parcel_id)
_agents: dict[str, dict[str, Any]] = {}


def _make_record(agent_id: str, parcel: ParcelAgent, config: dict[str, Any]) -> dict[str, Any]:
    state = parcel.get_state()
    return {
        "id": agent_id,
        "parcel_id": state["parcel_id"],
        "status": "active" if state["active"] else "paused",
        "balance": state["balance_usdx"],
        "location": state["location"],
        "owner": state["owner"],
        "metadata": state["metadata"],
        "config": config,
    }


@router.post("", status_code=201)
async def create_agent(body: dict[str, Any]) -> dict[str, Any]:
    parcel_id = body.get("parcel_id", "")
    owner = body.get("wallet_address", "0x0000000000000000000000000000000000000000")
    if not parcel_id:
        raise HTTPException(status_code=422, detail=[{"msg": "parcel_id is required"}])

    agent_id = f"agent-{uuid.uuid4().hex[:8]}"
    parcel = ParcelAgent(
        parcel_id=parcel_id,
        owner_address=owner,
        location=body.get("location", {"lat": 37.7749, "lng": -122.4194, "alt": 0.0}),
    )
    config = body.get("config", {})
    if body.get("initial_balance", 0) > 0:
        await parcel.deposit(amount_usdx=float(body["initial_balance"]))

    record = _make_record(agent_id, parcel, config)
    _agents[agent_id] = {"agent": parcel, "config": config}
    return record


@router.get("")
async def list_agents() -> list[dict[str, Any]]:
    return [_make_record(aid, v["agent"], v["config"]) for aid, v in _agents.items()]


@router.get("/{agent_id}")
async def get_agent(agent_id: str) -> dict[str, Any]:
    entry = _agents.get(agent_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return _make_record(agent_id, entry["agent"], entry["config"])


@router.patch("/{agent_id}")
async def update_agent(agent_id: str, body: dict[str, Any]) -> dict[str, Any]:
    entry = _agents.get(agent_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    parcel: ParcelAgent = entry["agent"]
    new_status = body.get("status")
    if new_status == "paused":
        parcel.state.active = False
    elif new_status == "active":
        parcel.state.active = True
    if "config" in body:
        entry["config"].update(body["config"])
    if "metadata" in body:
        parcel.state.metadata.update(body["metadata"])
    return _make_record(agent_id, parcel, entry["config"])


@router.delete("/{agent_id}", status_code=204)
async def delete_agent(agent_id: str) -> None:
    if agent_id not in _agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    del _agents[agent_id]


@router.post("/{agent_id}/optimize")
async def optimize_agent(agent_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    entry = _agents.get(agent_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    result = await entry["agent"].optimize(context=body or {})
    return result
