"""Contracts API router — Web4AGI."""

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

from src.contracts.manager import ContractError, ContractManager
from src.models.parcel_models import ContractRequest, ContractResponse

router = APIRouter()

_manager = ContractManager()


@router.post("", response_model=ContractResponse, status_code=201)
async def create_contract(body: ContractRequest) -> dict[str, Any]:
    try:
        contract_id = _manager.propose(
            proposer_id=body.party_a,
            counterparty_id=body.party_b,
            terms={"contract_type": body.contract_type, **body.terms},
        )
    except ContractError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    contract = _manager.get(contract_id)
    return {
        "contract_id": contract["contract_id"],
        "contract_type": body.contract_type,
        "status": contract["status"],
        "parties": {"party_a": body.party_a, "party_b": body.party_b},
        "terms": body.terms,
        "created_at": contract["created_at"],
        "tx_hash": contract["tx_hash"],
    }


@router.get("/{contract_id}", response_model=ContractResponse)
async def get_contract(contract_id: str) -> dict[str, Any]:
    try:
        contract = _manager.get(contract_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "contract_id": contract["contract_id"],
        "contract_type": contract["terms"].get("contract_type", "custom"),
        "status": contract["status"],
        "parties": {
            "party_a": contract["proposer"],
            "party_b": contract["counterparty"],
        },
        "terms": contract["terms"],
        "created_at": contract["created_at"],
        "tx_hash": contract["tx_hash"],
    }


@router.post("/{contract_id}/sign")
async def sign_contract(contract_id: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        _manager.sign(
            contract_id=contract_id,
            signer_id=body.get("agent_id", ""),
            signature=body.get("signature", ""),
        )
        contract = _manager.get(contract_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ContractError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "contract_id": contract_id,
        "status": contract["status"],
        "signatures": contract["signatures"],
        "fully_signed": _manager.is_fully_signed(contract_id),
    }


@router.post("/{contract_id}/execute")
async def execute_contract(contract_id: str) -> dict[str, Any]:
    try:
        result = await _manager.execute(contract_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


@router.post("/{contract_id}/reject")
async def reject_contract(contract_id: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        _manager.reject(
            contract_id=contract_id,
            rejector_id=body.get("agent_id", ""),
            reason=body.get("reason", ""),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"contract_id": contract_id, "status": "rejected"}


@router.get("")
async def list_contracts(party_id: str | None = None, status: str | None = None) -> dict[str, Any]:
    contracts = _manager.list_contracts(party_id=party_id, status=status)
    return {"contracts": contracts, "total": len(contracts)}
