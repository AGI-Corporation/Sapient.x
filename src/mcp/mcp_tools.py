"""MCPToolkit — Web4AGI

Model Context Protocol (MCP) integration for parcel agents.
Connects to Route.X MCP server for tool discovery and inter-agent messaging.

Route.X repo: https://github.com/AGI-Corporation/Route.X
"""

import json
import asyncio
from typing import Callable, Dict, List, Optional, Any
from datetime import datetime

try:
    import httpx
except ImportError:
    httpx = None


ROUTE_X_BASE = "http://localhost:8001"  # Default Route.X MCP server


# ── Tool Registry ───────────────────────────────────────────────────────────────

_LOCAL_TOOLS: Dict[str, Callable] = {}
_LOCAL_TOOL_METADATA: Dict[str, Dict] = {}


def register_tool(name: str, description: str = "", parameters: Optional[Dict] = None):
    """Decorator to register a local MCP tool."""
    def decorator(fn: Callable):
        _LOCAL_TOOLS[name] = fn
        _LOCAL_TOOL_METADATA[name] = {
            "description": description or f"Tool: {name}",
            "parameters": parameters or {},
        }
        return fn
    return decorator


@register_tool("parcel.get_state", description="Get the current state of a parcel agent")
async def tool_get_state(parcel_id: str) -> Dict:
    return {"tool": "parcel.get_state", "parcel_id": parcel_id, "note": "Delegate to ParcelAgent"}


@register_tool("parcel.list_neighbors", description="List neighboring parcel agents within a radius")
async def tool_list_neighbors(parcel_id: str, radius_meters: float = 100.0) -> Dict:
    return {"tool": "parcel.list_neighbors", "parcel_id": parcel_id, "radius": radius_meters}


@register_tool("trade.create_offer", description="Create a trade offer")
async def tool_create_offer(seller_id: str, asset: str, amount_usdx: float) -> Dict:
    return {"tool": "trade.create_offer", "seller": seller_id, "asset": asset, "amount": amount_usdx}


@register_tool("trade.get_offers", description="Get current trade offers")
async def tool_get_offers(parcel_id: Optional[str] = None) -> Dict:
    return {"tool": "trade.get_offers", "filter": parcel_id}


@register_tool("optimize.run", description="Run the LangGraph optimization workflow")
async def tool_optimize(parcel_id: str, context: Dict = None) -> Dict:
    return {"tool": "optimize.run", "parcel_id": parcel_id, "context": context or {}}


@register_tool("payment.transfer", description="Transfer USDx between parcels")
async def tool_payment_transfer(from_id: str, to_id: str, amount_usdx: float) -> Dict:
    return {"tool": "payment.transfer", "from": from_id, "to": to_id, "amount": amount_usdx}


@register_tool(
    "get_location_data",
    description="Fetch spatial data for a lat/lng coordinate",
    parameters={"lat": {"type": "number"}, "lng": {"type": "number"}},
)
async def tool_get_location_data(lat: float = 0.0, lng: float = 0.0) -> Dict:
    return {"success": True, "data": {"lat": lat, "lng": lng, "type": "location"}}


# ── MCPToolkit Class ─────────────────────────────────────────────────────────────

class MCPToolkit:
    """MCP client for parcel agents. Connects to Route.X for tool routing."""

    _REQUIRED_MESSAGE_FIELDS = ("from", "to", "content", "timestamp")

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
        self._tools: Dict[str, Dict] = {}  # instance-level tool registry

    # ── Tool Execution ───────────────────────────────────────────────────────

    def list_tools(self) -> List[Dict]:
        """List all available MCP tools (synchronous, local only)."""
        tools = []
        for name in _LOCAL_TOOLS:
            meta = _LOCAL_TOOL_METADATA.get(name, {})
            tools.append({
                "name": name,
                "source": "local",
                "description": meta.get("description", f"Tool: {name}"),
                "parameters": meta.get("parameters", {}),
            })
        for name, info in self._tools.items():
            tools.append({
                "name": name,
                "source": "instance",
                "description": info.get("description", f"Tool: {name}"),
                "parameters": info.get("parameters", {}),
            })
        return tools

    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str = "",
        parameters: Optional[Dict] = None,
    ) -> None:
        """Register a callable as a tool on this MCPToolkit instance."""
        self._tools[name] = {
            "func": func,
            "description": description,
            "parameters": parameters or {},
        }

    async def call_tool(
        self,
        tool_name: str,
        parameters: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Dict:
        """Call a tool locally or via Route.X MCP server."""
        # Merge explicit 'parameters' dict into kwargs
        if parameters is not None:
            kwargs.update(parameters)

        # Check instance-level tools first
        if tool_name in self._tools:
            func = self._tools[tool_name]["func"]
            if asyncio.iscoroutinefunction(func):
                return await func(**kwargs)
            return func(**kwargs)

        # Check module-level registry
        if tool_name in _LOCAL_TOOLS:
            return await _LOCAL_TOOLS[tool_name](**kwargs)

        if self.local_only:
            return {"success": False, "error": f"Tool '{tool_name}' not found locally"}

        # Delegate to Route.X
        return await self._route_x_call(tool_name, kwargs)

    async def _route_x_call(self, tool_name: str, args: Dict) -> Dict:
        """Forward a tool call to the Route.X MCP server."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": args,
            },
            "id": f"{self.agent_id}-{int(datetime.utcnow().timestamp() * 1000)}",
        }
        if httpx is None:
            return {"success": True, "simulated": True, "tool": tool_name, "args": args}
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.route_x_url}/mcp",
                json=payload,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("result", data)

    async def alist_tools(self) -> List[Dict]:
        """List tools asynchronously, including remote tools from Route.X."""
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

    # ── Messaging ───────────────────────────────────────────────────────────────

    async def _send_raw(self, to: str, content: Dict) -> Dict:
        """Send a raw message envelope to another agent."""
        return await self.send(to=to, payload=content)

    async def send_message(
        self,
        target_id: str,
        content: Dict,
        max_retries: int = 1,
    ) -> Dict:
        """Send a message with optional retry logic."""
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                result = await self._send_raw(target_id, content)
                if not isinstance(result, dict):
                    result = {"success": True}
                result.setdefault(
                    "message_id",
                    f"msg-{int(datetime.utcnow().timestamp() * 1000)}-{attempt}",
                )
                return result
            except Exception as exc:  # noqa: BLE001 — catching any send failure for retry
                last_error = exc
        return {"success": False, "error": str(last_error)}

    async def receive_messages(self) -> List[Dict]:
        """Poll for incoming messages."""
        return await self._poll_messages()

    async def _poll_messages(self) -> List[Dict]:
        """Internal: drain the local inbox or poll Route.X."""
        return await self.receive()

    async def broadcast(self, target_ids: List[str], content: Dict) -> List[Dict]:
        """Send a message to multiple agents concurrently."""
        tasks = [self.send_message(target_id=tid, content=content) for tid in target_ids]
        return list(await asyncio.gather(*tasks))

    def validate_message(self, message: Dict) -> bool:
        """Return True if *message* contains all required fields."""
        return all(field in message for field in self._REQUIRED_MESSAGE_FIELDS)

    def get_queue_size(self) -> int:
        """Return the number of messages currently in the local inbox."""
        return self._inbox.qsize()

    def validate_parameters(self, tool_spec: Dict, params: Dict) -> bool:
        """Validate *params* against the required parameters in *tool_spec*."""
        parameters = tool_spec.get("parameters", {})
        for name, spec in parameters.items():
            if spec.get("required", False) and name not in params:
                return False
        return True

    async def get_connection_status(self) -> Dict:
        """Return the current connection status for this MCPToolkit instance."""
        return {
            "connected": not self.local_only,
            "agent_id": self.agent_id,
            "local_only": self.local_only,
            "route_x_url": self.route_x_url,
        }

    async def send(self, to: str, payload: Dict) -> Dict:
        """Send a message to another agent via Route.X."""
        envelope = {
            "from": self.agent_id,
            "to": to,
            "payload": payload,
            "sent_at": datetime.utcnow().isoformat(),
        }
        if self.local_only or httpx is None:
            print(f"[MCP Sim] {self.agent_id} -> {to}: {json.dumps(payload)[:80]}")
            return {"success": True, "simulated": True}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.route_x_url}/messages",
                    json=envelope,
                    timeout=10,
                )
                resp.raise_for_status()
                return resp.json()
        except (httpx.RequestError, httpx.HTTPStatusError):
            # Simulation mode when Route.X is unreachable
            print(f"[MCP Sim] {self.agent_id} -> {to}: {json.dumps(payload)[:80]}")
            return {"success": True, "simulated": True}

    async def receive(self) -> List[Dict]:
        """Poll Route.X for messages addressed to this agent."""
        if self.local_only or httpx is None:
            msgs = []
            while not self._inbox.empty():
                msgs.append(await self._inbox.get())
            return msgs
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.route_x_url}/messages/{self.agent_id}",
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("messages", [])

    def inject_message(self, msg: Dict) -> None:
        """Inject a message into the local inbox (for testing)."""
        self._inbox.put_nowait(msg)
