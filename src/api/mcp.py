"""MCP API router — Web4AGI."""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException

from src.mcp.mcp_tools import MCPToolkit
from src.models.parcel_models import MCPMessage, MCPToolCall

router = APIRouter()

_TOOLKIT = MCPToolkit(agent_id="api-gateway", local_only=False)


@router.get("/tools")
async def list_tools() -> Any:
    return _TOOLKIT.list_tools()


@router.post("/tools/call")
async def call_tool(body: MCPToolCall) -> Any:
    result = await _TOOLKIT.call_tool(tool_name=body.tool_name, parameters=body.arguments)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Tool call failed"))
    return result


@router.post("/messages")
async def send_message(body: MCPMessage) -> Any:
    result = await _TOOLKIT.send_message(
        target_id=body.to_parcel_id,
        content={"from": body.from_parcel_id, "type": body.msg_type, **body.payload},
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Send failed"))
    return result


@router.get("/status")
async def connection_status() -> Any:
    return await _TOOLKIT.get_connection_status()
