"""Parcel Digital Agent — Web4AGI

Each parcel in the Metaverse is represented by a ParcelAgent.
Agents can:
  - Manage parcel metadata and state
  - Communicate with other parcel agents via MCP
  - Execute USDx trades via x402 protocol
  - Optimize via LangGraph + Sentient Foundation models
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ParcelAgent:
    """An autonomous agent bound to a single parcel in the Web4AGI metaverse."""

    def __init__(
        self,
        parcel_id: str | None = None,
        owner_address: str = "",
        location: dict[str, float] | None = None,
        wallet_private_key: str | None = None,
    ):
        self.logger = logging.getLogger(f"ParcelAgent.{parcel_id or 'init'}")
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
        self.goals = ["maximize_community_engagement", "minimize_utility_risk"]

    # ── State Management ──────────────────────────────────────────────────────

    def update_metadata(self, key: str, value: Any) -> None:
        """Update a metadata field on the parcel."""
        self.state.metadata[key] = value
        self.state.last_updated = datetime.now(timezone.utc).isoformat()

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
            "goals": self.goals,
        }

    # ── Communication (Refined MCP Interoperability) ──────────────────────────

    async def send_message(self, target_parcel_id: str, content: dict[str, Any]) -> dict:
        """Send a standardized MCP message to another parcel agent."""
        self.logger.debug(f"Sending MCP message to {target_parcel_id}: {content.get('type')}")
        return await self.mcp.send_message(
            target_id=target_parcel_id,
            content={
                "parcel_id": self.parcel_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **content,
            },
        )

    async def receive_messages(self) -> list[dict]:
        """Poll MCP toolkit and local queue for incoming messages."""
        # 1. Fetch from Route.X (MCP)
        remote_msgs = await self.mcp.receive_messages()

        # 2. Fetch from local Queue
        local_msgs = []
        while not self._message_queue.empty():
            local_msgs.append(await self._message_queue.get())

        return remote_msgs + local_msgs

    # ── Discovery & Autonomous Logic ──────────────────────────────────────────

    async def discover_neighbors(self, radius_meters: float = 100.0) -> list[str]:
        """Discover neighboring parcel IDs using Spatial Fabric MCP tools."""
        self.logger.info(f"Discovering neighbors within {radius_meters}m")
        # In this OS, hierarchy tool returns the parcel_id for coordinates.
        # We simulate neighbor discovery by querying around the current location.
        neighbors = []
        try:
            res = await self.mcp.call_tool(
                "parcel_get_place_hierarchy",
                lat=self.location["lat"] + 0.001,
                lon=self.location["lng"] + 0.001,
            )
            if res.get("success") and "data" in res:
                data = res.get("data", {})
                pid = data.get("parcel_id") if isinstance(data, dict) else None
                if pid and pid != self.parcel_id:
                    neighbors.append(pid)
        except Exception as e:
            self.logger.error(f"Neighbor discovery failed: {e}")

        return neighbors

    async def discover_by_capability(self, capability: str) -> list[str]:
        """Discover other agents via the NANDA registry by capability."""
        self.logger.info(f"Discovering agents with capability: {capability}")
        try:
            res = await self.mcp.call_tool("nanda_discover_agents", capability=capability)
            if res.get("success") and isinstance(res.get("data"), list):
                return [a["agent_id"] for a in res["data"] if a["agent_id"] != self.parcel_id]
        except Exception as e:
            self.logger.error(f"NANDA discovery failed: {e}")
        return []

    async def register_in_nanda(self) -> dict:
        """Register this agent's capabilities in the NANDA registry."""
        capabilities = ["parcel_management", "usdx_payments"]
        if "maximize_community_engagement" in self.goals:
            capabilities.append("community_engagement")

        self.logger.info(f"Registering in NANDA with capabilities: {capabilities}")
        return await self.mcp.call_tool(
            "nanda_register_agent",
            agent_id=self.parcel_id,
            capabilities=capabilities,
            owner_address=self.owner_address,
            metadata=self.state.metadata,
        )

    async def perform_autonomous_actions(self) -> None:
        """Execute autonomous logic based on current goals."""
        # 1. Goal: minimize_utility_risk
        if "minimize_utility_risk" in self.goals:
            try:
                risk = await self.mcp.call_tool("spatial_get_utility_risk", parcel_id=self.parcel_id)
                if risk.get("success"):
                    self.update_metadata("utility_risk", risk["data"].get("risk_level", "unknown"))
            except Exception as e:
                self.logger.error(f"Failed to fetch utility risk: {e}")

        # 2. Goal: maximize_community_engagement
        if "maximize_community_engagement" in self.goals:
            # 2a. Discovery neighbors
            neighbors = await self.discover_neighbors()

            # 2b. Discovery Stikk spots
            try:
                spots = await self.mcp.call_tool(
                    "spatial_get_stikk_spots",
                    lat=self.location["lat"],
                    lon=self.location["lng"],
                )
                if spots.get("success") and spots["data"]:
                    self.update_metadata("nearby_stikk_count", len(spots["data"]))
            except Exception as e:
                self.logger.error(f"Failed to fetch Stikk spots: {e}")

            for neighbor_id in neighbors:
                self.logger.info(f"Proposing check-in incentive contract to {neighbor_id}")
                # We use the new incentive specialized trade_type
                await self.trade(
                    counterparty_id=neighbor_id,
                    amount_usdx=0.5,
                    trade_type="incentive:check_in",
                    contract_terms={"purpose": "community_growth"},
                )

    # ── Trading (Refined Interoperability) ────────────────────────────────────

    async def deposit(self, amount_usdx: float) -> dict:
        """Deposit USDx into the parcel wallet via x402."""
        self.logger.info(f"Initiating deposit of {amount_usdx} USDx")
        try:
            # We use the MCP tool for balance query interoperability
            await self.mcp.call_tool("parcel_get_usdx_balance", parcel_id=self.parcel_id)

            result = await self.x402.deposit(amount=amount_usdx)
            if result.get("success"):
                self.state.balance_usdx += amount_usdx
                self.logger.info(
                    f"Successfully deposited {amount_usdx} USDx. New balance: {self.state.balance_usdx}"
                )
            else:
                self.logger.error(f"Deposit failed: {result.get('error', 'Unknown error')}")
            return result
        except Exception as e:
            self.logger.exception(f"Exception during deposit: {e}")
            return {"success": False, "error": str(e)}

    async def trade(
        self,
        counterparty_id: str,
        amount_usdx: float,
        trade_type: str = "transfer",
        contract_terms: dict | None = None,
    ) -> dict:
        """Execute a USDx trade with another parcel agent using MCP tools."""
        self.logger.info(f"Initiating trade: {trade_type} {amount_usdx} USDx to {counterparty_id}")

        # Use refined MCP toolkit for validation/interoperability
        if self.state.balance_usdx < amount_usdx:
            self.logger.warning(
                f"Trade failed: Insufficient balance ({self.state.balance_usdx} < {amount_usdx})"
            )
            return {"success": False, "error": "Insufficient USDx balance"}

        try:
            # Execute transfer via refined X402 client
            result = await self.x402.transfer(
                to_address=counterparty_id,
                amount=amount_usdx,
                memo=f"{trade_type}:{self.parcel_id}->{counterparty_id}",
                contract_terms=contract_terms,
            )

            if result.get("success"):
                self.state.balance_usdx -= amount_usdx
                self.logger.info(f"Trade successful. New balance: {self.state.balance_usdx}")

                # Update counterparty balance if they are in the same OS process
                from src.main import PARCEL_AGENTS

                if counterparty_id in PARCEL_AGENTS:
                    PARCEL_AGENTS[counterparty_id].state.balance_usdx += amount_usdx

                # Notify counterparty via MCP message
                await self.send_message(
                    counterparty_id,
                    {
                        "type": "payment_received",
                        "amount": amount_usdx,
                        "tx_hash": result.get("tx_hash"),
                        "trade_type": trade_type,
                    },
                )
            else:
                self.logger.error(f"Trade failed: {result.get('error', 'Unknown error')}")
            return result
        except Exception as e:
            self.logger.exception(f"Exception during trade: {e}")
            return {"success": False, "error": str(e)}

    async def sign_contract(self, counterparty_id: str, contract: dict[str, Any]) -> dict:
        """Sign a smart contract with another parcel agent using X402 protocol."""
        self.logger.info(f"Signing contract with {counterparty_id}: {contract.get('type')}")
        return await self.x402.sign_contract(
            contract=contract,
            counterparty=counterparty_id,
            signer=self.owner_address,
        )

    # ── Optimization ──────────────────────────────────────────────────────────

    async def optimize(self, context: dict | None = None) -> dict:
        """Run the LangGraph optimization workflow for this parcel."""
        from src.graphs.langgraph_workflow import run_parcel_optimization

        self.logger.info("Starting parcel optimization workflow")
        return await run_parcel_optimization(
            parcel_state=self.get_state(),
            context=context or {},
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def run(self, cycles: int = 0) -> None:
        """Main agent loop. Runs indefinitely (cycles=0) or for N cycles."""
        self.logger.info(f"Starting agent run loop (cycles={cycles})")
        count = 0
        while self.state.active:
            # 1. Process incoming messages
            msgs = await self.receive_messages()
            for msg in msgs:
                payload = msg.get("payload", msg.get("content", msg))
                await self._handle_message(payload)

            # 2. Autonomous actions
            await self.perform_autonomous_actions()

            await asyncio.sleep(1)
            count += 1
            if cycles and count >= cycles:
                break
        self.logger.info("Agent run loop stopped")

    async def _handle_message(self, msg: dict) -> None:
        """Route incoming MCP messages to appropriate handlers."""
        msg_type = msg.get("type", "unknown")
        sender = msg.get("from", msg.get("parcel_id", "unknown"))

        self.logger.info(f"Handling message of type '{msg_type}' from {sender}")

        if msg_type == "trade_request":
            await self.trade(
                counterparty_id=sender,
                amount_usdx=msg["amount"],
                trade_type=msg.get("trade_type", "transfer"),
            )
        elif msg_type == "contract_offer":
            # Auto-accept small incentive contracts for demo purposes
            if msg["contract"].get("type") == "check_in_incentive":
                self.logger.info(f"Auto-accepting incentive contract from {sender}")
                await self.sign_contract(counterparty_id=sender, contract=msg["contract"])
            else:
                await self.sign_contract(
                    counterparty_id=sender,
                    contract=msg["contract"],
                )
        elif msg_type == "optimize":
            await self.optimize(context=msg.get("context"))
        elif msg_type == "payment_received":
            self.logger.info(f"Confirmed payment of {msg.get('amount')} received from {sender}")
        else:
            self.logger.warning(f"Unknown message type: {msg_type}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = ParcelAgent(
        owner_address="0xYourWalletAddress",
        location={"lat": 37.7749, "lng": -122.4194, "alt": 0.0},
    )
    print(agent.get_state())
    asyncio.run(agent.run(cycles=3))
