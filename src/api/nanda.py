"""NANDA API router — Web4AGI."""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.nanda.registry import NANDARegistry

router = APIRouter()

_registry = NANDARegistry(local_only=True)


@router.post("/register", status_code=201)
async def register_agent(body: dict[str, Any]) -> dict[str, Any]:
    fact = await _registry.register(
        parcel_id=body.get("parcel_id", ""),
        owner_address=body.get("owner_address", ""),
        capabilities=body.get("capabilities", []),
        location=body.get("location", {}),
        region_tag=body.get("region_tag", ""),
        protocols=body.get("protocols"),
        mcp_endpoint=body.get("mcp_endpoint", ""),
        metadata=body.get("metadata"),
    )
    return fact.to_dict()


@router.delete("/{agent_id}", status_code=204)
async def deregister_agent(agent_id: str) -> None:
    removed = await _registry.deregister(agent_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found in NANDA registry")


@router.get("/find")
async def find_agents(
    capability: str | None = None,
    region_tag: str | None = None,
    protocol: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    agents = await _registry.find(
        capability=capability,
        region_tag=region_tag,
        protocol=protocol,
        limit=limit,
    )
    return {"agents": [a.to_dict() for a in agents], "total": len(agents)}


@router.get("/{agent_id}")
async def get_agent_fact(agent_id: str) -> dict[str, Any]:
    fact = await _registry.get(agent_id)
    if fact is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return fact.to_dict()


@router.get("")
async def list_agents() -> dict[str, Any]:
    facts = _registry.list_all()
    return {"agents": [f.to_dict() for f in facts], "total": _registry.agent_count}
