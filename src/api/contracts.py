"""Contracts API Router — Web4AGI."""

from fastapi import APIRouter, HTTPException

from src.models.parcel_models import ContractRequest, ContractResponse, SuccessResponse

router = APIRouter()


@router.post("/", response_model=ContractResponse)
async def propose_contract(request: ContractRequest):
    """Propose a new contract between two agents."""
    from src.core.state import PARCEL_AGENTS

    if request.party_a not in PARCEL_AGENTS:
        raise HTTPException(status_code=404, detail=f"Agent {request.party_a} not found")

    # In a real system, this would store the contract in a DB
    # Here we simulate the proposal flow
    import uuid
    from datetime import datetime, timezone

    contract_id = f"con-{uuid.uuid4().hex[:8]}"

    return ContractResponse(
        contract_id=contract_id,
        contract_type=request.contract_type,
        status="proposed",
        parties={"party_a": request.party_a, "party_b": request.party_b},
        terms=request.terms,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/{contract_id}/sign", response_model=SuccessResponse)
async def sign_contract(contract_id: str, agent_id: str):
    """Sign an existing contract."""
    from src.core.state import PARCEL_AGENTS

    if agent_id not in PARCEL_AGENTS:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = PARCEL_AGENTS[agent_id]
    # Simulate contract fetching
    contract = {"contract_id": contract_id, "type": "parcel_lease"}

    result = await agent.sign_contract(counterparty_id="other-agent", contract=contract)

    if not result.get("success"):
        raise HTTPException(status_code=400, detail="Signing failed")

    return SuccessResponse(message=f"Contract {contract_id} signed by {agent_id}", data=result)
