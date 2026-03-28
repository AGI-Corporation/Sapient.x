"""MCPToolkit — Web4AGI

Model Context Protocol (MCP) integration for parcel agents.
Connects to Route.X MCP server for tool discovery and inter-agent messaging.

Route.X repo: https://github.com/AGI-Corporation/Route.X
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

try:
    import httpx
except ImportError:
    httpx = None


ROUTE_X_BASE = "http://localhost:8001"  # Default Route.X MCP server


# ── Tool Registry ───────────────────────────────────────────────────────────────

_LOCAL_TOOLS: Dict[str, Callable] = {}


def register_tool(name: str):
    """Decorator to register a local MCP tool."""
    def decorator(fn: Callable):
        _LOCAL_TOOLS[name] = fn
        return fn
    return decorator


@register_tool("parcel.get_state")
async def tool_get_state(parcel_id: str) -> Dict:
    return {"tool": "parcel.get_state", "parcel_id": parcel_id, "note": "Delegate to ParcelAgent"}


@register_tool("parcel.list_neighbors")
async def tool_list_neighbors(parcel_id: str, radius_meters: float = 100.0) -> Dict:
    return {"tool": "parcel.list_neighbors", "parcel_id": parcel_id, "radius": radius_meters}


@register_tool("trade.create_offer")
async def tool_create_offer(seller_id: str, asset: str, amount_usdx: float) -> Dict:
    return {"tool": "trade.create_offer", "seller": seller_id, "asset": asset, "amount": amount_usdx}


@register_tool("trade.get_offers")
async def tool_get_offers(parcel_id: Optional[str] = None) -> Dict:
    return {"tool": "trade.get_offers", "filter": parcel_id}


@register_tool("optimize.run")
async def tool_optimize(parcel_id: str, context: Dict = None) -> Dict:
    return {"tool": "optimize.run", "parcel_id": parcel_id, "context": context or {}}


@register_tool("payment.transfer")
async def tool_payment_transfer(from_id: str, to_id: str, amount_usdx: float) -> Dict:
    return {"tool": "payment.transfer", "from": from_id, "to": to_id, "amount": amount_usdx}


# ── Bitrefill eCommerce MCP Tools ───────────────────────────────────────────────

@register_tool("bitrefill.ping")
async def tool_bitrefill_ping() -> Dict:
    """Check whether the Bitrefill API is reachable."""
    from src.agents.bitrefill_agent import BitrefillAgent
    agent = BitrefillAgent()
    return await agent.ping()


@register_tool("bitrefill.categories")
async def tool_bitrefill_categories() -> Dict:
    """Return the full Bitrefill product-type → categories map."""
    from src.agents.bitrefill_agent import BitrefillAgent
    agent = BitrefillAgent()
    return await agent.get_categories()


@register_tool("bitrefill.search")
async def tool_bitrefill_search(
    query: str,
    country: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 10,
) -> Dict:
    """Search Bitrefill for gift cards, eSIMs, mobile top-ups, and more."""
    from src.agents.bitrefill_agent import BitrefillAgent
    agent = BitrefillAgent()
    return await agent.search(query=query, country=country, category=category, limit=limit)


@register_tool("bitrefill.detail")
async def tool_bitrefill_detail(product_id: str) -> Dict:
    """Get detailed information about a Bitrefill product by its ID."""
    from src.agents.bitrefill_agent import BitrefillAgent
    agent = BitrefillAgent()
    return await agent.get_product_detail(product_id)


@register_tool("bitrefill.create_invoice")
async def tool_bitrefill_create_invoice(
    product_id: str,
    quantity: int = 1,
    value: Optional[float] = None,
    payment_method: str = "balance",
) -> Dict:
    """Create a Bitrefill invoice for a product (requires API credentials)."""
    from src.agents.bitrefill_agent import make_bitrefill_agent
    agent = make_bitrefill_agent()
    product: Dict[str, Any] = {"product_id": product_id, "quantity": quantity}
    if value is not None:
        product["value"] = value
    return await agent.bitrefill.create_invoice(products=[product], payment_method=payment_method)


@register_tool("bitrefill.pay_invoice")
async def tool_bitrefill_pay_invoice(invoice_id: str) -> Dict:
    """Pay a Bitrefill invoice using the account balance (requires API credentials)."""
    from src.agents.bitrefill_agent import make_bitrefill_agent
    agent = make_bitrefill_agent()
    return await agent.bitrefill.pay_invoice(invoice_id)


@register_tool("bitrefill.get_invoices")
async def tool_bitrefill_get_invoices(limit: int = 20) -> Dict:
    """List recent Bitrefill invoices (requires API credentials)."""
    from src.agents.bitrefill_agent import make_bitrefill_agent
    agent = make_bitrefill_agent()
    return await agent.get_invoices(limit=limit)


@register_tool("bitrefill.get_orders")
async def tool_bitrefill_get_orders(limit: int = 20) -> Dict:
    """List recent Bitrefill orders (requires API credentials)."""
    from src.agents.bitrefill_agent import make_bitrefill_agent
    agent = make_bitrefill_agent()
    return await agent.get_orders(limit=limit)


@register_tool("bitrefill.get_order")
async def tool_bitrefill_get_order(order_id: str) -> Dict:
    """Get a specific Bitrefill order by ID (requires API credentials)."""
    from src.agents.bitrefill_agent import make_bitrefill_agent
    agent = make_bitrefill_agent()
    return await agent.get_order(order_id)


@register_tool("bitrefill.unseal_order")
async def tool_bitrefill_unseal_order(order_id: str) -> Dict:
    """Reveal codes and PINs for a Bitrefill order (requires API credentials)."""
    from src.agents.bitrefill_agent import make_bitrefill_agent
    agent = make_bitrefill_agent()
    return await agent.unseal_order(order_id)


@register_tool("bitrefill.purchase")
async def tool_bitrefill_purchase(
    product_id: str,
    quantity: int = 1,
    value: Optional[float] = None,
    payment_method: str = "balance",
    use_x402: bool = False,
    x402_amount_usdx: Optional[float] = None,
) -> Dict:
    """End-to-end Bitrefill purchase: search → create invoice → pay via x402 or balance.

    When use_x402 is True the agent first settles x402_amount_usdx via the
    x402 protocol (USDC/stablecoin) before paying the invoice with balance.
    Requires BITREFILL_API_ID, BITREFILL_API_SECRET, and (if use_x402) X402_PRIVATE_KEY.
    """
    from src.agents.bitrefill_agent import make_bitrefill_agent
    agent = make_bitrefill_agent()
    return await agent.purchase(
        product_id=product_id,
        quantity=quantity,
        value=value,
        payment_method=payment_method,
        use_x402=use_x402,
        x402_amount_usdx=x402_amount_usdx,
    )


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

    # ── Tool Execution ───────────────────────────────────────────────────────

    async def call_tool(self, tool_name: str, **kwargs) -> Dict:
        """Call a tool locally or via Route.X MCP server."""
        # Try local first
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

    async def list_tools(self) -> List[Dict]:
        """List available MCP tools from Route.X."""
        local = [{"name": name, "source": "local"} for name in _LOCAL_TOOLS]
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
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.route_x_url}/messages",
                json=envelope,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()

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
