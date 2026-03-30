"""Contracts API router — Web4AGI."""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.contracts.manager import ContractManager
from src.models.parcel_models import ContractRequest, ContractResponse, SuccessResponse

router = APIRouter()

_CONTRACT_MANAGER = ContractManager()


@router.post("/", response_model=dict, status_code=201)
async def create_contract(body: ContractRequest) -> dict[str, Any]:
    """Create a new contract proposal."""
    try:
        contract_id = _CONTRACT_MANAGER.propose(
            party_a=body.party_a,
            party_b=body.party_b,
            contract_data={"terms": body.terms},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _CONTRACT_MANAGER.get(contract_id)  # type: ignore[return-value]


@router.get("/{contract_id}")
async def get_contract(contract_id: str) -> dict[str, Any]:
    """Get a contract by ID."""
    contract = _CONTRACT_MANAGER.get(contract_id)
    if contract is None:
        raise HTTPException(status_code=404, detail=f"Contract '{contract_id}' not found")
    return contract


@router.post("/{contract_id}/sign")
async def sign_contract(contract_id: str, signer_id: str, signature: str) -> dict[str, Any]:
    """Record a signature on a contract."""
    success = _CONTRACT_MANAGER.sign(contract_id, signer_id, signature)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot sign contract in current state")
    contract = _CONTRACT_MANAGER.get(contract_id)
    return {"success": True, "signatures": contract["signatures"], "status": contract["status"]}


@router.post("/{contract_id}/reject")
async def reject_contract(
    contract_id: str, rejector_id: str, reason: str = ""
) -> dict[str, Any]:
    """Reject a contract."""
    success = _CONTRACT_MANAGER.reject(contract_id, rejector_id, reason)
    if not success:
        raise HTTPException(status_code=404, detail=f"Contract '{contract_id}' not found")
    return {"success": True, "status": "rejected"}


@router.post("/{contract_id}/execute")
async def execute_contract(contract_id: str) -> dict[str, Any]:
    """Execute a fully-signed contract."""
    result = await _CONTRACT_MANAGER.execute(contract_id)
    if not result.get("status") == "executed":
        raise HTTPException(status_code=400, detail=result.get("error", "Execution failed"))
    return result


@router.get("/")
async def list_contracts(party: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
    """List all contracts, optionally filtered."""
    return [c for c in _CONTRACT_MANAGER.list_contracts(party=party, status=status)]
