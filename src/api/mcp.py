"""MCP router — Web4AGI API."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/tools")
async def list_tools():
    return {"tools": []}
