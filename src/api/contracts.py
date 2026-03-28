"""Contracts router — Web4AGI API."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_contracts():
    return {"contracts": []}


@router.get("/{contract_id}")
async def get_contract(contract_id: str):
    return {"contract_id": contract_id}
