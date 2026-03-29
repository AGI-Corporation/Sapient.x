"""Parcel Digital Agent — Web4AGI

Each parcel in the Metaverse is represented by a ParcelAgent.
Agents can:
  - Manage parcel metadata and state
  - Communicate with other parcel agents via MCP
  - Execute USDx trades via x402 protocol
  - Optimize via LangGraph + Sentient Foundation models
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.mcp.mcp_tools import MCPToolkit
from src.payments.x402_client import X402Client


@dataclass
class ParcelState:
    parcel_id: str
    owner_address: str
    location: dict[str, float]  # {lat, lng, alt}
    metadata: dict[str, Any] = field(default_factory=dict)
    balance_usdx: float = 0.0
    active: bool = True
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ParcelAgent:
    """An autonomous agent bound to a single parcel in the Web4AGI metaverse."""

    def __init__(
        self,
        parcel_id: str | None = None,
        owner_address: str = "",
        location: dict[str, float] | None = None,
        wallet_private_key: str | None = None,
    ):
        self.parcel_id = parcel_id or str(uuid.uuid4())
        self.owner_address = owner_address
        self.location = location or {"lat": 37.7749, "lng": -122.4194, "alt": 0.0}
        self.state = ParcelState(
            parcel_id=self.parcel_id,
            owner_address=owner_address,
            location=self.location,
        )
        self.x402 = X402Client(private_key=wallet_private_key)
        self.mcp = MCPToolkit(agent_id=self.parcel_id)
        self._message_queue: asyncio.Queue = asyncio.Queue()

    # ── State Management ──────────────────────────────────────────────────────

    def update_metadata(self, key: str, value: Any) -> None:
        """Update a metadata field on the parcel."""
        self.state.metadata[key] = value
        self.state.last_updated = datetime.utcnow().isoformat()

    def get_state(self) -> dict[str, Any]:
        """Return the current parcel state as a dict."""
        return {
            "parcel_id": self.state.parcel_id,
            "owner": self.state.owner_address,
            "location": self.state.location,
            "balance_usdx": self.state.balance_usdx,
            "metadata": self.state.metadata,
            "active": self.state.active,
            "last_updated": self.state.last_updated,
        }

    # ── Communication ─────────────────────────────────────────────────────────

    async def send_message(self, target_parcel_id: str, content: dict[str, Any]) -> dict:
        """Send an MCP message to another parcel agent."""
        return await self.mcp.send(
            to=target_parcel_id,
            payload={
                "from": self.parcel_id,
                "timestamp": datetime.utcnow().isoformat(),
                **content,
            },
        )

    async def receive_messages(self) -> list:
        """Drain the incoming message queue."""
        messages = []
        while not self._message_queue.empty():
            messages.append(await self._message_queue.get())
        return messages

    # ── Trading ───────────────────────────────────────────────────────────────

    async def deposit(self, amount_usdx: float) -> dict:
        """Deposit USDx into the parcel wallet via x402."""
        result = await self.x402.deposit(amount=amount_usdx)
        if result.get("success"):
            self.state.balance_usdx += amount_usdx
        return result

    async def trade(
        self,
        counterparty_id: str,
        amount_usdx: float,
        trade_type: str = "transfer",
        contract_terms: dict | None = None,
    ) -> dict:
        """Execute a USDx trade with another parcel agent."""
        if self.state.balance_usdx < amount_usdx:
            return {"success": False, "error": "Insufficient USDx balance"}

        result = await self.x402.transfer(
            to_address=counterparty_id,
            amount=amount_usdx,
            memo=f"{trade_type}:{self.parcel_id}->{counterparty_id}",
            contract_terms=contract_terms,
        )
        if result.get("success"):
            self.state.balance_usdx -= amount_usdx
        return result

    async def sign_contract(self, counterparty_id: str, contract: dict[str, Any]) -> dict:
        """Sign a smart contract with another parcel agent."""
        return await self.x402.sign_contract(
            contract=contract,
            counterparty=counterparty_id,
            signer=self.owner_address,
        )

    # ── Optimization ──────────────────────────────────────────────────────────

    async def optimize(self, context: dict | None = None) -> dict:
        """Run the LangGraph optimization workflow for this parcel."""
        from src.graphs.langgraph_workflow import run_parcel_optimization

        return await run_parcel_optimization(
            parcel_state=self.get_state(),
            context=context or {},
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def run(self, cycles: int = 0) -> None:
        """Main agent loop. Runs indefinitely (cycles=0) or for N cycles."""
        count = 0
        while self.state.active:
            msgs = await self.receive_messages()
            for msg in msgs:
                await self._handle_message(msg)
            await asyncio.sleep(1)
            count += 1
            if cycles and count >= cycles:
                break

    async def _handle_message(self, msg: dict) -> None:
        """Route incoming MCP messages to appropriate handlers."""
        # Handle enveloped messages from MCPToolkit/Route.X
        data = msg.get("payload", msg)
        msg_type = data.get("type", "unknown")
        sender = data.get("from", msg.get("from", "unknown"))

        if msg_type == "trade_request":
            await self.trade(
                counterparty_id=sender,
                amount_usdx=data["amount"],
                trade_type=data.get("trade_type", "transfer"),
            )
        elif msg_type == "contract_offer":
            await self.sign_contract(
                counterparty_id=sender,
                contract=data["contract"],
            )
        elif msg_type == "optimize":
            await self.optimize(context=msg.get("context"))
        else:
            print(f"[ParcelAgent {self.parcel_id}] Unknown message type: {msg_type}")


if __name__ == "__main__":
    agent = ParcelAgent(
        owner_address="0xYourWalletAddress",
        location={"lat": 37.7749, "lng": -122.4194, "alt": 0.0},
    )
    print(agent.get_state())
    asyncio.run(agent.run(cycles=3))
