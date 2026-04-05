"""Contracts API router — Web4AGI."""

from typing import Any

from fastapi import APIRouter, Body, HTTPException

from src.contracts.manager import ContractManager
from src.models.parcel_models import ContractRequest, ContractResponse

router = APIRouter()

# Module-level singleton
_CONTRACT_MANAGER = ContractManager()


@router.post("/", response_model=ContractResponse, status_code=201)
async def create_contract(data: ContractRequest) -> Any:
    """Propose a new service contract between two parties."""
    try:
        contract_id = _CONTRACT_MANAGER.propose(
            party_a=data.party_a,
            party_b=data.party_b,
            terms=data.terms,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    contract = _CONTRACT_MANAGER.get(contract_id)
    return ContractResponse(
        contract_id=contract_id,
        contract_type=data.contract_type,
        status=contract["status"],
        parties={"party_a": data.party_a, "party_b": data.party_b},
        terms=data.terms,
        created_at=contract["created_at"],
        tx_hash=None,
    )


@router.get("/{contract_id}", response_model=ContractResponse)
async def get_contract(contract_id: str) -> Any:
    """Retrieve a contract by ID."""
    try:
        contract = _CONTRACT_MANAGER.get(contract_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ContractResponse(
        contract_id=contract_id,
        contract_type=contract["terms"].get("type", "custom"),
        status=contract["status"],
        parties={"party_a": contract["party_a"], "party_b": contract["party_b"]},
        terms=contract["terms"],
        created_at=contract["created_at"],
        tx_hash=contract.get("tx_hash"),
    )


@router.post("/{contract_id}/sign")
async def sign_contract(
    contract_id: str,
    agent_id: str = Body(...),
    signature: str = Body(...),
) -> Any:
    """Record a party's digital signature on a contract."""
    try:
        signed = _CONTRACT_MANAGER.sign(contract_id, agent_id, signature)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not signed:
        raise HTTPException(status_code=400, detail="Contract cannot be signed in its current state")

    contract = _CONTRACT_MANAGER.get(contract_id)
    return {
        "contract_id": contract_id,
        "signatures": contract["signatures"],
        "fully_signed": _CONTRACT_MANAGER.is_fully_signed(contract_id),
    }


@router.post("/{contract_id}/execute")
async def execute_contract(contract_id: str) -> Any:
    """Execute a fully-signed contract and settle escrow."""
    try:
        result = await _CONTRACT_MANAGER.execute(contract_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result


@router.post("/{contract_id}/cancel")
async def cancel_contract(contract_id: str) -> Any:
    """Cancel a pending contract."""
    try:
        cancelled = _CONTRACT_MANAGER.cancel(contract_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not cancelled:
        raise HTTPException(status_code=400, detail="Contract cannot be cancelled in its current state")
    return {"contract_id": contract_id, "status": "cancelled"}
