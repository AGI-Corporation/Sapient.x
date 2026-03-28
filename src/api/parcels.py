"""Parcels router — Web4AGI API."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_parcels():
    return {"parcels": []}


@router.get("/{parcel_id}")
async def get_parcel(parcel_id: str):
    return {"parcel_id": parcel_id}
