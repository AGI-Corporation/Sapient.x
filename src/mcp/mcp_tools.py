"""MCPToolkit — Web4AGI

Model Context Protocol (MCP) integration for parcel agents.
Connects to Route.X MCP server for tool discovery and inter-agent messaging.

Route.X repo: https://github.com/AGI-Corporation/Route.X
"""

import json
import asyncio
import uuid
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

try:
    import httpx
except ImportError:
    httpx = None


ROUTE_X_BASE = "http://localhost:8001"  # Default Route.X MCP server


# ── Tool Registry ───────────────────────────────────────────────────────────────

# Module-level registry stores (fn, description, parameters_schema) per tool name
_LOCAL_TOOLS: Dict[str, Dict] = {}


def register_tool(name: str, description: str = "", parameters: Optional[Dict] = None):
    """Decorator to register a local MCP tool at module level."""
    def decorator(fn: Callable):
        _LOCAL_TOOLS[name] = {
            "fn": fn,
            "description": description or fn.__doc__ or "",
            "parameters": parameters or {},
        }
        return fn
    return decorator


@register_tool("parcel.get_state", description="Get the current state of a parcel.", parameters={"parcel_id": {"type": "string", "required": True}})
async def tool_get_state(parcel_id: str) -> Dict:
    return {"tool": "parcel.get_state", "parcel_id": parcel_id, "note": "Delegate to ParcelAgent"}


@register_tool("parcel.list_neighbors", description="List neighboring parcels within a radius.", parameters={"parcel_id": {"type": "string", "required": True}, "radius_meters": {"type": "number", "required": False}})
async def tool_list_neighbors(parcel_id: str, radius_meters: float = 100.0) -> Dict:
    return {"tool": "parcel.list_neighbors", "parcel_id": parcel_id, "radius": radius_meters}


@register_tool("trade.create_offer", description="Create a trade offer.", parameters={"seller_id": {"type": "string", "required": True}, "asset": {"type": "string", "required": True}, "amount_usdx": {"type": "number", "required": True}})
async def tool_create_offer(seller_id: str, asset: str, amount_usdx: float) -> Dict:
    return {"tool": "trade.create_offer", "seller": seller_id, "asset": asset, "amount": amount_usdx}


@register_tool("trade.get_offers", description="List current trade offers.", parameters={"parcel_id": {"type": "string", "required": False}})
async def tool_get_offers(parcel_id: Optional[str] = None) -> Dict:
    return {"tool": "trade.get_offers", "filter": parcel_id}


@register_tool("optimize.run", description="Run optimization workflow for a parcel.", parameters={"parcel_id": {"type": "string", "required": True}, "context": {"type": "object", "required": False}})
async def tool_optimize(parcel_id: str, context: Optional[Dict] = None) -> Dict:
    return {"tool": "optimize.run", "parcel_id": parcel_id, "context": context or {}}


@register_tool("payment.transfer", description="Transfer USDx between parcels.", parameters={"from_id": {"type": "string", "required": True}, "to_id": {"type": "string", "required": True}, "amount_usdx": {"type": "number", "required": True}})
async def tool_payment_transfer(from_id: str, to_id: str, amount_usdx: float) -> Dict:
    return {"tool": "payment.transfer", "from": from_id, "to": to_id, "amount": amount_usdx}


@register_tool("get_location_data", description="Fetch location-specific data for given coordinates.", parameters={"lat": {"type": "number", "required": True}, "lng": {"type": "number", "required": True}})
async def tool_get_location_data(lat: float, lng: float) -> Dict:
    return {"success": True, "data": {"lat": lat, "lng": lng, "zone": "sf-downtown"}}


# ── Memory Store ────────────────────────────────────────────────────────────────

class _MemoryStore:
    """Simple in-memory history store for MCPToolkit."""

    def __init__(self):
        self._history: List[Dict] = []

    def record(self, entry: Dict) -> None:
        self._history.append(entry)

    def get_history(self) -> List[Dict]:
        return list(self._history)


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
        self.route_x_url = route_x_url
        self.local_only = local_only
        self._inbox: asyncio.Queue = asyncio.Queue()
        # Instance-level tool overrides (registered via instance method)
        self._instance_tools: Dict[str, Dict] = {}
        self._memory = _MemoryStore()

    # ── Tool Registration (instance-level) ──────────────────────────────────

    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str = "",
        parameters: Optional[Dict] = None,
    ) -> None:
        """Register a custom tool on this toolkit instance."""
        self._instance_tools[name] = {
            "fn": func,
            "description": description or func.__doc__ or "",
            "parameters": parameters or {},
        }

    def _all_tools(self) -> Dict[str, Dict]:
        """Merge module-level and instance-level registries (instance wins)."""
        merged = dict(_LOCAL_TOOLS)
        merged.update(self._instance_tools)
        return merged

    # ── Tool Listing ────────────────────────────────────────────────────────

    def list_tools(self) -> List[Dict]:
        """List available MCP tools (local + remote)."""
        tools = []
        for name, meta in self._all_tools().items():
            tools.append({
                "name": name,
                "description": meta.get("description", ""),
                "parameters": meta.get("parameters", {}),
                "source": "local",
            })
        return tools

    # ── Tool Execution ───────────────────────────────────────────────────────

    async def call_tool(self, tool_name: str, parameters: Optional[Dict] = None, **kwargs) -> Dict:
        """Call a tool locally or via Route.X MCP server."""
        params = parameters or kwargs
        all_tools = self._all_tools()

        if tool_name in all_tools:
            try:
                result = await all_tools[tool_name]["fn"](**params)
                if isinstance(result, dict):
                    return {"success": True, "data": result, **result}
                return {"success": True, "data": result}
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        if self.local_only:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}

        return await self._route_x_call(tool_name, params)

    async def _route_x_call(self, tool_name: str, args: Dict) -> Dict:
        """Forward a tool call to the Route.X MCP server."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": args},
            "id": f"{self.agent_id}-{int(datetime.utcnow().timestamp() * 1000)}",
        }
        if httpx is None:
            return {"success": True, "simulated": True, "tool": tool_name, "args": args}
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.route_x_url}/mcp", json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return data.get("result", data)

    # ── Messaging ───────────────────────────────────────────────────────────────

    async def _send_raw(self, envelope: Dict) -> Dict:
        """Internal: send a raw message envelope."""
        if self.local_only or httpx is None:
            print(f"[MCP Sim] raw send: {json.dumps(envelope)[:80]}")
            return {"success": True, "simulated": True, "message_id": str(uuid.uuid4())}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{self.route_x_url}/messages", json=envelope, timeout=10)
                resp.raise_for_status()
                result = resp.json()
                if "message_id" not in result:
                    result["message_id"] = str(uuid.uuid4())
                return result
        except Exception:
            return {"success": True, "simulated": True, "message_id": str(uuid.uuid4())}

    async def send_message(
        self,
        target_id: str,
        content: Dict,
        max_retries: int = 0,
    ) -> Dict:
        """Send a message to another agent. Supports optional retries."""
        envelope = {
            "from": self.agent_id,
            "to": target_id,
            "payload": content,
            "sent_at": datetime.utcnow().isoformat(),
        }
        last_exc: Optional[Exception] = None
        for attempt in range(max(1, max_retries)):
            try:
                result = await self._send_raw(envelope)
                message_id = result.get("message_id") or str(uuid.uuid4())
                return {"success": True, "message_id": message_id}
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < max_retries - 1:
                    continue
        return {"success": False, "error": str(last_exc)}

    async def _poll_messages(self) -> List[Dict]:
        """Internal: poll Route.X for messages."""
        if self.local_only or httpx is None:
            msgs: List[Dict] = []
            while not self._inbox.empty():
                msgs.append(await self._inbox.get())
            return msgs
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.route_x_url}/messages/{self.agent_id}", timeout=10)
            resp.raise_for_status()
            return resp.json().get("messages", [])

    async def receive_messages(self) -> List[Dict]:
        """Receive all pending messages for this agent."""
        return await self._poll_messages()

    async def broadcast(self, target_ids: List[str], content: Dict) -> List[Dict]:
        """Send the same message to multiple agents concurrently."""
        tasks = [self.send_message(target_id=tid, content=content) for tid in target_ids]
        return list(await asyncio.gather(*tasks))

    async def send(self, to: str, payload: Dict) -> Dict:
        """Low-level send; kept for backward compatibility."""
        return await self.send_message(target_id=to, content=payload)

    async def receive(self) -> List[Dict]:
        """Low-level receive; kept for backward compatibility."""
        return await self.receive_messages()

    def inject_message(self, msg: Dict) -> None:
        """Inject a message into the local inbox (for testing)."""
        self._inbox.put_nowait(msg)

    def get_queue_size(self) -> int:
        """Return the number of messages currently in the local inbox queue."""
        return self._inbox.qsize()

    # ── Validation ──────────────────────────────────────────────────────────

    def validate_message(self, msg: Dict) -> bool:
        """Validate that a message has the required fields."""
        required = {"from", "to", "content", "timestamp"}
        return required.issubset(msg.keys())

    def validate_parameters(self, tool_spec: Dict, params: Dict) -> bool:
        """Validate parameters against a tool specification."""
        spec_params = tool_spec.get("parameters", {})
        for param_name, param_meta in spec_params.items():
            if param_meta.get("required", False) and param_name not in params:
                return False
        return True

    # ── Status ──────────────────────────────────────────────────────────────

    async def get_connection_status(self) -> Dict:
        """Return the current connection status of this toolkit."""
        connected = not self.local_only and httpx is not None
        if not self.local_only and httpx is not None:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{self.route_x_url}/health", timeout=3)
                    connected = resp.status_code == 200
            except Exception:
                connected = False
        return {
            "connected": connected,
            "agent_id": self.agent_id,
            "route_x_url": self.route_x_url,
            "local_only": self.local_only,
        }

    async def list_tools_remote(self) -> List[Dict]:
        """List available MCP tools from Route.X (async, includes remote)."""
        local = self.list_tools()
        if self.local_only or httpx is None:
            return local
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.route_x_url}/mcp",
                    json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "list"},
                    timeout=5,
                )
                remote = resp.json().get("result", {}).get("tools", [])
                return local + [{**t, "source": "route.x"} for t in remote]
        except Exception:
            return local
