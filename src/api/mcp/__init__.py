"""MCP API router — Web4AGI.

Exposes the Route.X MCP orchestration layer as REST endpoints so that
external clients and agents can call tools and exchange messages via HTTP.
"""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.mcp.mcp_tools import MCPToolkit
from src.models.parcel_models import MCPMessage, MCPToolCall

router = APIRouter()

# Platform-level MCP toolkit (local-only, Route.X forwarding disabled by default)
_PLATFORM_MCP = MCPToolkit(agent_id="platform-mcp", local_only=True)


@router.get("/tools")
async def list_tools() -> Any:
    """List all available MCP tools (local + Route.X)."""
    tools = await _PLATFORM_MCP.list_tools()
    return {"tools": tools, "count": len(tools)}


@router.post("/tools/call")
async def call_tool(call: MCPToolCall) -> Any:
    """Invoke an MCP tool by name with the supplied arguments."""
    result = await _PLATFORM_MCP.call_tool(call.tool_name, call.arguments)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail="Tool call failed")
    return result


@router.post("/messages")
async def send_message(msg: MCPMessage) -> Any:
    """Send an MCP message from one parcel agent to another."""
    result = await _PLATFORM_MCP.send(
        to=msg.to_parcel_id,
        payload={
            "from": msg.from_parcel_id,
            "type": msg.msg_type,
            **msg.payload,
        },
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail="Message send failed")
    return result


@router.get("/messages/{agent_id}")
async def receive_messages(agent_id: str) -> Any:
    """Poll for messages addressed to an agent."""
    toolkit = MCPToolkit(agent_id=agent_id, local_only=True)
    messages = await toolkit.receive_messages()
    return {"agent_id": agent_id, "messages": messages, "count": len(messages)}
