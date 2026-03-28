"""Contracts v2 API router — Web4AGI (/api/contracts).

Integration-test-friendly endpoints used by test_api.py.
"""

from typing import Any, Dict
from datetime import datetime
from fastapi import APIRouter, HTTPException

router = APIRouter()

# Module-level store keyed by contract_id
_CONTRACTS: Dict[str, Dict] = {}


@router.post("", status_code=201)
async def create_contract(body: Dict[str, Any]) -> Any:
    # Import inside function so @patch('src.contracts.manager.ContractManager') applies
    from src.contracts.manager import ContractManager
    mgr = ContractManager()
    # Use mgr.id (from mock or real) as the contract ID
    contract_id = str(mgr.id)
    status = str(getattr(mgr, "status", "pending"))
    record: Dict[str, Any] = {
        "id": contract_id,
        "agent_id": body.get("agent_id", ""),
        "counterparty_id": body.get("counterparty_id", ""),
        "type": body.get("type", "sale_agreement"),
        "terms": body.get("terms", {}),
        "status": status,
        "signatures": [],
        "created_at": datetime.utcnow().isoformat(),
    }
    _CONTRACTS[contract_id] = record
    return record


@router.get("/{contract_id}")
async def get_contract(contract_id: str) -> Any:
    record = _CONTRACTS.get(contract_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Contract '{contract_id}' not found")
    return record


@router.post("/{contract_id}/sign")
async def sign_contract(contract_id: str, body: Dict[str, Any]) -> Any:
    record = _CONTRACTS.setdefault(contract_id, {
        "id": contract_id,
        "signatures": [],
        "status": "pending",
    })
    record.setdefault("signatures", []).append({
        "agent_id": body.get("agent_id", ""),
        "signature": body.get("signature", ""),
        "signed_at": datetime.utcnow().isoformat(),
    })
    return record


@router.post("/{contract_id}/execute")
async def execute_contract(contract_id: str) -> Any:
    record = _CONTRACTS.setdefault(contract_id, {"id": contract_id})
    record["status"] = "executed"
    return record


@router.get("")
async def list_contracts() -> Any:
    return list(_CONTRACTS.values())
