"""TradeAgent — Web4AGI

Orchestrates multi-parcel trade negotiations and contract execution.
Built on top of ParcelAgent with added auction and batch-trade logic.
"""

import asyncio
from datetime import datetime
from typing import Any


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
        self.expires_at = datetime.utcnow().timestamp() + ttl_seconds
        self.bids: list[dict] = []
        self.accepted: dict | None = None

    def is_expired(self) -> bool:
        return datetime.utcnow().timestamp() > self.expires_at

    def add_bid(self, bidder_id: str, bid_amount: float) -> None:
        self.bids.append({"bidder": bidder_id, "amount": bid_amount, "ts": datetime.utcnow().isoformat()})

    def best_bid(self) -> dict | None:
        if not self.bids:
            return None
        return max(self.bids, key=lambda b: b["amount"])


class TradeAgent:
    """Manages trade orchestration across multiple parcel agents."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.offers: dict[str, TradeOffer] = {}
        self.trade_history: list[dict] = []

    # ── Offer Management ───────────────────────────────────────────────────

    def create_offer(
        self,
        seller_id: str,
        asset: str,
        amount_usdx: float,
        ttl_seconds: int = 300,
    ) -> TradeOffer:
        offer_id = f"offer-{seller_id}-{int(datetime.utcnow().timestamp())}"
        offer = TradeOffer(offer_id, seller_id, asset, amount_usdx, ttl_seconds)
        self.offers[offer_id] = offer
        return offer

    def place_bid(self, offer_id: str, bidder_id: str, bid_amount: float) -> dict:
        offer = self.offers.get(offer_id)
        if not offer:
            return {"success": False, "error": "Offer not found"}
        if offer.is_expired():
            return {"success": False, "error": "Offer expired"}
        offer.add_bid(bidder_id, bid_amount)
        return {"success": True, "offer_id": offer_id, "bid": bid_amount}

    def close_offer(self, offer_id: str) -> dict:
        offer = self.offers.get(offer_id)
        if not offer:
            return {"success": False, "error": "Offer not found"}
        winner = offer.best_bid()
        if not winner:
            return {"success": False, "error": "No bids placed"}
        offer.accepted = winner
        record = {
            "offer_id": offer_id,
            "seller": offer.seller_id,
            "winner": winner["bidder"],
            "amount": winner["amount"],
            "asset": offer.asset,
            "closed_at": datetime.utcnow().isoformat(),
        }
        self.trade_history.append(record)
        return {"success": True, **record}

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
        return [
            r if isinstance(r, dict) else {"success": False, "error": str(r)}
            for r in results
        ]

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
            "created_at": datetime.utcnow().isoformat(),
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
            "created_at": datetime.utcnow().isoformat(),
            "status": "pending_signatures",
        }

    # ── History ───────────────────────────────────────────────────────────────

    def get_history(self, limit: int = 50) -> list[dict]:
        return self.trade_history[-limit:]

    def volume_usdx(self) -> float:
        return sum(t["amount"] for t in self.trade_history)
