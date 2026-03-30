"""MCP API router — Web4AGI."""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.mcp.mcp_tools import MCPToolkit
from src.models.parcel_models import MCPMessage, MCPToolCall, SuccessResponse

router = APIRouter()

_PLATFORM_MCP = MCPToolkit(agent_id="platform", local_only=True)


@router.get("/tools")
async def list_tools() -> list[dict[str, Any]]:
    """List all available MCP tools."""
    return await _PLATFORM_MCP.list_tools()


@router.post("/tools/call")
async def call_tool(body: MCPToolCall) -> dict[str, Any]:
    """Call an MCP tool by name with given arguments."""
    result = await _PLATFORM_MCP.call_tool(tool_name=body.tool_name, parameters=body.arguments)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Tool call failed"))
    return result


@router.post("/messages")
async def send_message(body: MCPMessage) -> dict[str, Any]:
    """Send an MCP message from one parcel agent to another."""
    toolkit = MCPToolkit(agent_id=body.from_parcel_id, local_only=True)
    result = await toolkit.send_message(
        target_id=body.to_parcel_id,
        content={"type": body.msg_type, **body.payload},
    )
    return result


@router.get("/status")
async def connection_status() -> dict[str, Any]:
    """Check the MCP connection status."""
    return await _PLATFORM_MCP.get_connection_status()
