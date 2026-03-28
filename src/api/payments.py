"""Payments router — Web4AGI API."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_payments():
    return {"payments": []}
