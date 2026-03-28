"""Contracts API router — Web4AGI."""

import uuid
from typing import Any, Dict
from datetime import datetime
from fastapi import APIRouter, HTTPException

from src.models.parcel_models import ContractRequest, ContractResponse

router = APIRouter()

_CONTRACTS: Dict[str, Dict] = {}


@router.post("", response_model=ContractResponse, status_code=201)
async def create_contract(body: ContractRequest) -> Any:
    contract_id = f"contract-{uuid.uuid4().hex[:8]}"
    record: Dict[str, Any] = {
        "contract_id": contract_id,
        "contract_type": body.contract_type,
        "status": "pending_signatures",
        "parties": {"party_a": body.party_a, "party_b": body.party_b},
        "terms": body.terms,
        "created_at": datetime.utcnow().isoformat(),
        "tx_hash": None,
        "signatures": [],
    }
    _CONTRACTS[contract_id] = record
    return record


@router.get("/{contract_id}", response_model=ContractResponse)
async def get_contract(contract_id: str) -> Any:
    return _get_or_404(contract_id)


@router.post("/{contract_id}/sign")
async def sign_contract(contract_id: str, body: Dict[str, Any]) -> Any:
    record = _get_or_404(contract_id)
    record.setdefault("signatures", []).append({
        "signer": body.get("signer", body.get("agent_id", "unknown")),
        "signature": body.get("signature", ""),
        "signed_at": datetime.utcnow().isoformat(),
    })
    if len(record["signatures"]) >= 2:
        record["status"] = "signed"
    return record


@router.post("/{contract_id}/execute")
async def execute_contract(contract_id: str) -> Any:
    record = _get_or_404(contract_id)
    record["status"] = "executed"
    record["tx_hash"] = f"0x{uuid.uuid4().hex}"
    return record


@router.get("")
async def list_contracts() -> Any:
    return list(_CONTRACTS.values())


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_or_404(contract_id: str) -> Dict:
    record = _CONTRACTS.get(contract_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Contract '{contract_id}' not found")
    return record
