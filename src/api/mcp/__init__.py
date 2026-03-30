"""MCP API router — Web4AGI.

Exposes a JSON-RPC 2.0 compatible endpoint backed by the local MCPToolkit,
mirroring the Route.X MCP server interface for local testing.
"""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.mcp.mcp_tools import MCPToolkit
from src.models.parcel_models import MCPMessage, MCPToolCall

router = APIRouter()

_toolkit = MCPToolkit(agent_id="api-gateway", local_only=True)


@router.post("")
async def mcp_jsonrpc(body: dict[str, Any]) -> dict[str, Any]:
    """JSON-RPC 2.0 dispatch for MCP tool calls and messages."""
    method = body.get("method", "")
    params = body.get("params", {})
    req_id = body.get("id", "unknown")

    if method == "tools/list":
        tools = await _toolkit.list_tools()
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        result = await _toolkit.call_tool(tool_name, arguments)
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    if method == "messages/send":
        result = await _toolkit.send(to=params.get("to", ""), payload=params.get("payload", {}))
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


@router.get("/tools")
async def list_tools() -> dict[str, Any]:
    tools = await _toolkit.list_tools()
    return {"tools": tools}


@router.post("/tools/call")
async def call_tool(body: MCPToolCall) -> dict[str, Any]:
    result = await _toolkit.call_tool(body.tool_name, body.arguments)
    return result


@router.post("/messages")
async def send_message(body: MCPMessage) -> dict[str, Any]:
    result = await _toolkit.send(
        to=body.to_parcel_id,
        payload={
            "from": body.from_parcel_id,
            "type": body.msg_type,
            **body.payload,
        },
    )
    return result


@router.get("/messages/{agent_id}")
async def receive_messages(agent_id: str) -> dict[str, Any]:
    tk = MCPToolkit(agent_id=agent_id, local_only=True)
    messages = await tk.receive()
    return {"agent_id": agent_id, "messages": messages}


@router.get("/status")
async def mcp_status() -> dict[str, Any]:
    status = await _toolkit.get_connection_status()
    return status
