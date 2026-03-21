"""TradeAgent — Web4AGI

Orchestrates multi-parcel trade negotiations and contract execution.
Built on top of ParcelAgent with added auction and batch-trade logic.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from src.mcp.mcp_tools import MCPToolkit
from src.payments.x402_client import X402Client


class TradeOffer:
    def __init__(
        self,
        offer_id: str,
        seller_id: str,
        asset: str,
        amount_usdx: float,
        ttl_seconds: int = 300,
    ):
        self.offer_id = offer_id
        self.seller_id = seller_id
        self.asset = asset
        self.amount_usdx = amount_usdx
        self.expires_at = datetime.now(timezone.utc).timestamp() + ttl_seconds
        self.bids: list[dict] = []
        self.accepted: dict | None = None

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc).timestamp() > self.expires_at

    def add_bid(self, bidder_id: str, bid_amount: float) -> None:
        self.bids.append(
            {
                "bidder": bidder_id,
                "amount": bid_amount,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )

    def best_bid(self) -> dict | None:
        if not self.bids:
            return None
        return max(self.bids, key=lambda b: b["amount"])


class TradeAgent:
    """Manages trade orchestration across multiple parcel agents."""

    def __init__(self, agent_id: str, wallet_private_key: str | None = None):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"TradeAgent.{agent_id}")
        self.offers: dict[str, TradeOffer] = {}
        self.trade_history: list[dict] = []

        # Interoperability Layers
        self.mcp = MCPToolkit(agent_id=self.agent_id)
        self.x402 = X402Client(private_key=wallet_private_key)

    # ── Offer Management (Refined Communication) ───────────────────────────

    def create_offer(
        self,
        seller_id: str,
        asset: str,
        amount_usdx: float,
        ttl_seconds: int = 300,
    ) -> TradeOffer:
        offer_id = f"offer-{seller_id}-{int(datetime.now(timezone.utc).timestamp())}"
        offer = TradeOffer(offer_id, seller_id, asset, amount_usdx, ttl_seconds)
        self.offers[offer_id] = offer
        self.logger.info(f"Created trade offer {offer_id} for asset {asset}")
        return offer

    async def broadcast_offer(
        self, offer_id: str, target_agents: list[str] | None = None
    ) -> list[dict]:
        """Broadcast an active offer to multiple agents via MCP.
        If target_agents is None, discovers them via NANDA for matching asset/trade types.
        Uses asyncio.gather (per D-02) for parallel message delivery.
        """
        offer = self.offers.get(offer_id)
        if not offer:
            return [{"success": False, "error": "Offer not found"}]

        if target_agents is None:
            self.logger.info(f"Discovering matching agents via NANDA for asset: {offer.asset}")
            try:
                # We search for agents with 'trading' or asset-specific capabilities
                res = await self.mcp.call_tool("nanda_discover_agents", capability="trading")
                if res.get("success") and isinstance(res.get("data"), list):
                    target_agents = [
                        a["agent_id"]
                        for a in res["data"]
                        if a["agent_id"] not in (self.agent_id, offer.seller_id)
                    ]
            except Exception as e:
                self.logger.error(f"NANDA discovery for broadcast failed: {e}")

        if not target_agents:
            self.logger.warning(f"No target agents found for broadcast of {offer_id}")
            return []

        self.logger.info(f"Broadcasting offer {offer_id} to {len(target_agents)} agents")

        # Per D-02: Parallelize delivery
        tasks = [
            self.mcp.send_message(
                target_id=tid,
                content={
                    "type": "trade_offer_announcement",
                    "offer_id": offer.offer_id,
                    "asset": offer.asset,
                    "min_amount": offer.amount_usdx,
                    "expires_at": offer.expires_at,
                },
            )
            for tid in target_agents
        ]
        return await asyncio.gather(*tasks)

    def place_bid(self, offer_id: str, bidder_id: str, bid_amount: float) -> dict:
        offer = self.offers.get(offer_id)
        if not offer:
            return {"success": False, "error": "Offer not found"}
        if offer.is_expired():
            return {"success": False, "error": "Offer expired"}

        offer.add_bid(bidder_id, bid_amount)
        self.logger.info(f"Placed bid of {bid_amount} from {bidder_id} on {offer_id}")
        return {"success": True, "offer_id": offer_id, "bid": bid_amount}

    async def close_offer(self, offer_id: str) -> dict:
        """Close an offer, select winner, and initiate settlement."""
        offer = self.offers.get(offer_id)
        if not offer:
            return {"success": False, "error": "Offer not found"}

        winner = offer.best_bid()
        if not winner:
            self.logger.warning(f"Closing offer {offer_id} with no bids")
            return {"success": False, "error": "No bids placed"}

        offer.accepted = winner
        record = {
            "offer_id": offer_id,
            "seller": offer.seller_id,
            "winner": winner["bidder"],
            "amount": winner["amount"],
            "asset": offer.asset,
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.trade_history.append(record)
        self.logger.info(
            f"Offer {offer_id} closed. Winner: {winner['bidder']} at {winner['amount']}"
        )

        # 1. Notify participants via MCP
        await self.mcp.send_message(
            winner["bidder"],
            {"type": "trade_won", "offer_id": offer_id, "amount": winner["amount"]},
        )

        # 2. Re-verify with seller
        await self.mcp.send_message(
            offer.seller_id,
            {
                "type": "trade_closed",
                "offer_id": offer_id,
                "winner": winner["bidder"],
                "amount": winner["amount"],
            },
        )

        return {"success": True, **record}

    # ── Settlement (Payment Layer) ───────────────────────────────────────────

    async def settle_trade(self, offer_id: str) -> dict:
        """Execute the final USDx transfer for a closed trade."""
        offer = self.offers.get(offer_id)
        if not offer or not offer.accepted:
            return {"success": False, "error": "Trade not ready for settlement"}

        winner = offer.accepted
        self.logger.info(f"Settling trade {offer_id}: {winner['bidder']} -> {offer.seller_id}")

        # In a real scenario, the 'winner' agent would call this or we'd orchestrate
        # Here we simulate the cross-agent settlement using X402
        try:
            result = await self.x402.transfer(
                to_address=offer.seller_id, amount=winner["amount"], memo=f"settle:{offer_id}"
            )
            if result.get("success"):
                self.logger.info(
                    f"Settlement successful for {offer_id}. TX: {result.get('tx_hash')}"
                )
            return result
        except Exception as e:
            self.logger.exception(f"Settlement failed for {offer_id}: {e}")
            return {"success": False, "error": str(e)}

    # ── Batch Trades ───────────────────────────────────────────────────────

    async def batch_transfer(
        self,
        sender_agent: Any,
        recipients: list[dict],  # [{"parcel_id": ..., "amount": ...}]
    ) -> list[dict]:
        """Execute multiple USDx transfers concurrently."""
        tasks = [
            sender_agent.trade(
                counterparty_id=r["parcel_id"],
                amount_usdx=r["amount"],
                trade_type="batch_transfer",
            )
            for r in recipients
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r if isinstance(r, dict) else {"success": False, "error": str(r)} for r in results]

    # ── Contract Templates ─────────────────────────────────────────────────

    @staticmethod
    def parcel_lease_contract(
        lessor_id: str,
        lessee_id: str,
        parcel_id: str,
        monthly_usdx: float,
        duration_months: int,
    ) -> dict:
        return {
            "type": "parcel_lease",
            "version": "1.0",
            "parties": {"lessor": lessor_id, "lessee": lessee_id},
            "parcel_id": parcel_id,
            "terms": {
                "monthly_rent_usdx": monthly_usdx,
                "duration_months": duration_months,
                "total_usdx": monthly_usdx * duration_months,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending_signatures",
        }

    @staticmethod
    def data_access_contract(
        provider_id: str,
        consumer_id: str,
        dataset: str,
        price_usdx: float,
    ) -> dict:
        return {
            "type": "data_access",
            "version": "1.0",
            "parties": {"provider": provider_id, "consumer": consumer_id},
            "dataset": dataset,
            "terms": {
                "price_usdx": price_usdx,
                "access_type": "read",
                "duration": "perpetual",
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending_signatures",
        }

    # ── History ───────────────────────────────────────────────────────────────

    def get_history(self, limit: int = 50) -> list[dict]:
        return self.trade_history[-limit:]

    def volume_usdx(self) -> float:
        return sum(t["amount"] for t in self.trade_history)
