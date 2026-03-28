"""NANDA Agent Registry API Router — Web4AGI."""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.models.parcel_models import AgentFact, RegistryResponse

router = APIRouter()

# ── Registry Store (In-memory for roadmap) ────────────────────────────────────

# agent_id -> registry details
AGENT_REGISTRY: dict[str, dict[str, Any]] = {}


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.post("/register", response_model=RegistryResponse)
async def register_agent(fact: AgentFact):
    """Register an agent's facts and capabilities into NANDA."""
    AGENT_REGISTRY[fact.agent_id] = fact.model_dump()
    return RegistryResponse(
        success=True, message=f"Agent {fact.agent_id} registered in NANDA", agent=fact
    )


@router.get("/discover", response_model=list[AgentFact])
async def discover_agents(capability: str | None = None):
    """Discover agents by capability."""
    results = []
    for fact in AGENT_REGISTRY.values():
        if capability is None or capability in fact.get("capabilities", []):
            results.append(AgentFact(**fact))
    return results


@router.get("/{agent_id}", response_model=AgentFact)
async def get_agent_fact(agent_id: str):
    """Retrieve verified facts about an agent."""
    if agent_id not in AGENT_REGISTRY:
        raise HTTPException(status_code=404, detail="Agent not found in NANDA registry")
    return AgentFact(**AGENT_REGISTRY[agent_id])
