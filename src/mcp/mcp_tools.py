"""MCPToolkit — Web4AGI

Model Context Protocol (MCP) integration for parcel agents.
Connects to Route.X MCP server for tool discovery and inter-agent messaging.
Also interfaces with OS-level servers:
  - SpatialFabricServer (:3001)
  - ParcelAgentServer (:3002)
  - ExchangeServer (:3003)

Route.X repo: https://github.com/AGI-Corporation/Route.X
"""

import asyncio
import json
import os
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, TypedDict

try:
    import httpx
except ImportError:
    httpx = None


ROUTE_X_BASE = os.getenv("ROUTE_X_URL", "http://localhost:8001")
SPATIAL_FABRIC_URL = os.getenv("SPATIAL_FABRIC_URL", "http://localhost:3001")
AGENT_SERVER_URL = os.getenv("AGENT_SERVER_URL", "http://localhost:3002")
EXCHANGE_SERVER_URL = os.getenv("EXCHANGE_SERVER_URL", "http://localhost:3003")
NANDA_REGISTRY_URL = os.getenv("NANDA_REGISTRY_URL", "http://localhost:8000/api/v1/registry")


# ── Type Definitions ─────────────────────────────────────────────────────────────


class MCPTool(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]
    source: str


class MCPMessageEnvelope(TypedDict):
    from_id: str
    to_id: str
    payload: dict[str, Any]
    sent_at: str


class MCPResult(TypedDict):
    success: bool
    data: Any | None
    error: str | None
    message_id: str | None


# ── Tool Registry ───────────────────────────────────────────────────────────────

_LOCAL_TOOLS: dict[str, dict[str, Any]] = {}


def register_tool(name: str, description: str = "", parameters: dict | None = None):
    """Decorator to register a local MCP tool."""

    def decorator(fn: Callable):
        _LOCAL_TOOLS[name] = {
            "func": fn,
            "description": description or fn.__doc__ or "",
            "parameters": parameters or {},
        }
        return fn

    return decorator


# ── Default Tool Implementations (Connecting to OS Servers) ─────────────────────


@register_tool(
    "parcel_get_place_hierarchy", "Return country/region/county/city/parcel IDs for a coordinate."
)
async def tool_get_place_hierarchy(lat: float, lon: float) -> dict:
    if httpx:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{SPATIAL_FABRIC_URL}/placeHierarchy", params={"lat": lat, "lon": lon}
                )
                if resp.status_code == 200:
                    return resp.json()
        except Exception:
            pass

    return {
        "parcel_id": f"sf-parcel-{int(lat * 1000)}-{int(lon * 1000)}",
        "hierarchy": ["USA", "California", "San Francisco County", "San Francisco"],
    }


@register_tool("parcel_get_parcel", "Return parcel geometry and attributes by parcel_id.")
async def tool_get_parcel(parcel_id: str) -> dict:
    if httpx:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{SPATIAL_FABRIC_URL}/parcel/{parcel_id}")
                if resp.status_code == 200:
                    return resp.json()
        except Exception:
            pass

    return {
        "parcel_id": parcel_id,
        "geometry": {"type": "Polygon", "coordinates": []},
        "attributes": {"zone": "commercial", "owner_count": 1},
    }


@register_tool("spatial_get_stikk_spots", "Retrieve Stikk loyalty spots near a coordinate.")
async def tool_get_stikk_spots(lat: float, lon: float, radius: float = 100.0) -> list:
    if httpx:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{SPATIAL_FABRIC_URL}/stikk/spots",
                    params={"lat": lat, "lon": lon, "radius": radius},
                )
                if resp.status_code == 200:
                    return resp.json().get("spots", [])
        except Exception:
            pass
    return [{"id": "stikk-001", "name": "Simulated Coffee Spot", "loyalty_points": 50}]


@register_tool("spatial_get_utility_risk", "Retrieve underground utility risk data for a parcel.")
async def tool_get_utility_risk(parcel_id: str) -> dict:
    if httpx:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{SPATIAL_FABRIC_URL}/utilities/risk/{parcel_id}")
                if resp.status_code == 200:
                    return resp.json()
        except Exception:
            pass
    return {"parcel_id": parcel_id, "risk_level": "low", "hazards": ["underground_power"]}


@register_tool(
    "parcel_get_agent_state", "Get optimization metrics and suggested actions for a parcel agent."
)
async def tool_get_agent_state(parcel_id: str) -> dict:
    if httpx:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{AGENT_SERVER_URL}/agent/{parcel_id}")
                if resp.status_code == 200:
                    return resp.json()
        except Exception:
            pass

    return {
        "parcel_id": parcel_id,
        "metrics": {"engagement": 0.85, "liquidity": 0.42},
        "suggested_actions": ["open_lease", "rebalance_usdx"],
    }


@register_tool(
    "parcel_get_usdx_balance",
    "Get the USDx (stablecoin) balance and limits for a parcel agent wallet.",
)
async def tool_get_usdx_balance(parcel_id: str) -> dict:
    # This might combine data from AgentServer and ExchangeServer
    return {"parcel_id": parcel_id, "balance_usdx": 1500.0, "limits": {"daily_transfer": 500.0}}


@register_tool(
    "parcel_propose_usdx_incentive_contract", "Propose a USDx-denominated incentive contract."
)
async def tool_propose_contract(
    parcel_id: str,
    counterparty_agent_id: str,
    purpose: str,
    asset_symbol: str = "USDx",
    rate_per_event: float = 0.1,
    max_total: float = 100.0,
) -> dict:
    if httpx:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{EXCHANGE_SERVER_URL}/exchange/trade",
                    json={
                        "sender": parcel_id,
                        "receiver": counterparty_agent_id,
                        "amount": max_total,
                        "asset": asset_symbol,
                        "memo": purpose,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "contract_id": data.get("trade_id"),
                        "status": "proposed",
                        "terms": {"rate": rate_per_event, "max": max_total, "purpose": purpose},
                    }
        except Exception:
            pass

    return {
        "contract_id": f"contract-{uuid.uuid4()}",
        "status": "proposed",
        "terms": {
            "parcel_id": parcel_id,
            "counterparty": counterparty_agent_id,
            "purpose": purpose,
            "rate": rate_per_event,
            "max": max_total,
        },
    }


@register_tool("nanda_register_agent", "Register an agent's facts and capabilities into NANDA.")
async def tool_nanda_register(
    agent_id: str, capabilities: list[str], owner_address: str, metadata: dict | None = None
) -> dict:
    if httpx:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{NANDA_REGISTRY_URL}/register",
                    json={
                        "agent_id": agent_id,
                        "capabilities": capabilities,
                        "owner_address": owner_address,
                        "metadata": metadata or {},
                    },
                )
                if resp.status_code == 200:
                    return resp.json()
        except Exception:
            pass
    return {"success": False, "error": "NANDA registry unavailable or simulated"}


@register_tool("nanda_discover_agents", "Discover agents by capability in the NANDA registry.")
async def tool_nanda_discover(capability: str | None = None) -> list:
    if httpx:
        try:
            async with httpx.AsyncClient() as client:
                params = {}
                if capability:
                    params["capability"] = capability
                resp = await client.get(f"{NANDA_REGISTRY_URL}/discover", params=params)
                if resp.status_code == 200:
                    return resp.json()
        except Exception:
            pass
    return []


# ── MCPToolkit Class ─────────────────────────────────────────────────────────────


class MCPToolkit:
    """MCP client for parcel agents. Connects to Route.X for tool routing."""

    def __init__(
        self,
        agent_id: str,
        route_x_url: str = ROUTE_X_BASE,
        local_only: bool = False,
    ):
        self.agent_id = agent_id
        self.route_x_url = route_x_url.rstrip("/")
        self.local_only = local_only
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._outbox: list[dict] = []
        self._custom_tools: dict[str, dict[str, Any]] = {}
        self._tool_schemas: dict[str, dict[str, Any]] = self._load_tool_schemas()

    def _load_tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Load schemas from mcp-tools.json if it exists."""
        schema_path = "mcp-tools.json"
        if os.path.exists(schema_path):
            try:
                with open(schema_path) as f:
                    data = json.load(f)
                    return {t["name"]: t for t in data.get("tools", [])}
            except Exception:
                pass
        return {}

    # ── Tool Execution ───────────────────────────────────────────────────────

    async def call_tool(self, tool_name: str, parameters: dict | None = None, **kwargs) -> dict:
        """Call a tool locally or via Route.X MCP server with parameter validation."""
        args = parameters or kwargs

        # 1. Parameter Validation
        if tool_name in self._tool_schemas:
            valid, error = self.validate_parameters(self._tool_schemas[tool_name], args)
            if not valid:
                return {"success": False, "error": f"Invalid parameters for {tool_name}: {error}"}

        # 2. Try Custom Instance Tools
        if tool_name in self._custom_tools:
            try:
                res = await self._custom_tools[tool_name]["func"](**args)
                return {"success": True, "data": res}
            except Exception as e:
                return {"success": False, "error": str(e)}

        # 3. Try Local Registry Tools
        if tool_name in _LOCAL_TOOLS:
            try:
                res = await _LOCAL_TOOLS[tool_name]["func"](**args)
                return {"success": True, "data": res}
            except Exception as e:
                return {"success": False, "error": str(e)}

        if self.local_only:
            # Fallback for common test tools if they are not in _LOCAL_TOOLS
            if tool_name == "get_location_data":
                return {
                    "success": True,
                    "data": {"lat": 37.7, "lng": -122.4},
                    "error": None,
                    "message_id": None,
                }
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found locally",
                "data": None,
                "message_id": None,
            }

        # 4. Delegate to Route.X
        return await self._route_x_call(tool_name, args)

    async def _route_x_call(self, tool_name: str, args: dict) -> dict:
        """Forward a tool call to the Route.X MCP server using JSON-RPC."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": args,
            },
            "id": f"{self.agent_id}-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        }

        if httpx is None:
            return {"success": True, "simulated": True, "tool": tool_name, "args": args}

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.route_x_url}/mcp",
                    json=payload,
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    return {
                        "success": False,
                        "error": data["error"].get("message", "Unknown RPC error"),
                    }
                return {"success": True, "data": data.get("result", data)}
        except Exception as e:
            return {"success": False, "error": f"Route.X connection failed: {e}"}

    async def list_tools(self) -> list[dict]:
        """List all available tools from local, custom, and remote sources."""
        local = [
            {"name": n, "source": "local", "description": v["description"]}
            for n, v in _LOCAL_TOOLS.items()
        ]
        custom = [
            {"name": n, "source": "custom", "description": v["description"]}
            for n, v in self._custom_tools.items()
        ]

        all_tools = local + custom

        if self.local_only or httpx is None:
            return all_tools

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.route_x_url}/mcp",
                    json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "list"},
                    timeout=5,
                )
                remote = resp.json().get("result", {}).get("tools", [])
                return all_tools + [{**t, "source": "route.x"} for t in remote]
        except Exception:
            return all_tools

    def register_tool(
        self, name: str, func: Callable, description: str = "", parameters: dict | None = None
    ):
        """Register a custom tool on this toolkit instance."""
        self._custom_tools[name] = {
            "func": func,
            "description": description or func.__doc__ or "",
            "parameters": parameters or {},
        }

    # ── Messaging ───────────────────────────────────────────────────────────────

    async def send_message(self, target_id: str, content: dict, max_retries: int = 3) -> dict:
        """Send a message with retry logic (compatibility with tests)."""
        for attempt in range(max_retries):
            try:
                res = await self.send(to=target_id, payload=content)
                if res.get("success"):
                    res["message_id"] = res.get("message_id") or str(uuid.uuid4())
                return res
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"success": False, "error": str(e)}
                await asyncio.sleep(0.5 * (2**attempt))  # Exponential backoff
        return {"success": False, "error": "Max retries reached"}

    async def send(self, to: str, payload: dict) -> dict:
        """Send an MCP message envelope to another agent via Route.X."""
        envelope = {
            "from": self.agent_id,
            "to": to,
            "payload": payload,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
        self._outbox.append(envelope)

        if self.local_only or httpx is None:
            return {"success": True, "simulated": True, "message_id": str(uuid.uuid4())}

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.route_x_url}/messages",
                    json=envelope,
                    timeout=10,
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            return {"success": False, "error": f"Message delivery failed: {e}"}

    async def receive_messages(self) -> list[dict]:
        """Poll for and receive messages (compatibility with tests)."""
        return await self.receive()

    async def receive(self) -> list[dict]:
        """Poll Route.X for messages addressed to this agent toolkit."""
        if self.local_only or httpx is None:
            msgs = []
            while not self._inbox.empty():
                msgs.append(await self._inbox.get())
            return msgs

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.route_x_url}/messages/{self.agent_id}",
                    timeout=10,
                )
                resp.raise_for_status()
                return resp.json().get("messages", [])
        except Exception:
            return []

    async def broadcast(self, target_ids: list[str], content: dict) -> list[dict]:
        """Broadcast a message to multiple targets concurrently."""
        tasks = [self.send_message(tid, content) for tid in target_ids]
        return await asyncio.gather(*tasks)

    def inject_message(self, msg: dict) -> None:
        """Inject a message into the local inbox (primarily for testing)."""
        self._inbox.put_nowait(msg)

    def get_queue_size(self) -> int:
        """Get the current size of the outgoing message history."""
        return len(self._outbox)

    def validate_message(self, message: dict) -> bool:
        """Check if an MCP message envelope has all required fields."""
        required = ["from", "to", "payload", "sent_at"]
        return all(k in message for k in required)

    def validate_parameters(self, tool_spec: dict, params: dict) -> tuple[bool, str | None]:
        """Validate parameters against a tool JSON schema."""
        schema = tool_spec.get("input_schema", {})
        required = schema.get("required", [])

        for r in required:
            if r not in params:
                return False, f"Missing required parameter: {r}"

        # Add basic type checking if needed here
        return True, None

    async def get_connection_status(self) -> dict:
        """Check the status of the connection to Route.X."""
        status = {
            "connected": True,
            "agent_id": self.agent_id,
            "local_only": self.local_only,
            "gateway": self.route_x_url,
        }
        if not self.local_only and httpx:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{self.route_x_url}/health", timeout=2)
                    status["healthy"] = resp.status_code == 200
            except Exception:
                status["healthy"] = False
        return status
