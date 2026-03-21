"""MCP API Router — Web4AGI."""

from fastapi import APIRouter, HTTPException

from src.models.parcel_models import MCPToolCall, SuccessResponse

router = APIRouter()


@router.get("/tools")
async def list_tools(agent_id: str = "default"):
    """List available MCP tools for an agent."""
    from src.agents.parcel_agent import ParcelAgent
    from src.core.state import PARCEL_AGENTS

    if agent_id in PARCEL_AGENTS:
        agent = PARCEL_AGENTS[agent_id]
    else:
        # Use a temporary agent for discovery
        agent = ParcelAgent(parcel_id="discovery-agent")

    tools = await agent.mcp.list_tools()
    return {"tools": tools}


@router.post("/call", response_model=SuccessResponse)
async def call_tool(request: MCPToolCall, agent_id: str = "default"):
    """Execute an MCP tool call."""
    from src.agents.parcel_agent import ParcelAgent
    from src.core.state import PARCEL_AGENTS

    if agent_id in PARCEL_AGENTS:
        agent = PARCEL_AGENTS[agent_id]
    else:
        agent = ParcelAgent(parcel_id="proxy-agent")

    result = await agent.mcp.call_tool(tool_name=request.tool_name, parameters=request.arguments)

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Tool call failed"))

    return SuccessResponse(message=f"Tool {request.tool_name} executed", data=result)
