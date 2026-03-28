"""Global OS State Storage — Sapient.x."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.parcel_agent import ParcelAgent
    from src.agents.trade_agent import TradeAgent

# ── Global state ──────────────────────────────────────────────────────────────

# Registry of all active parcel agents: parcel_id -> ParcelAgent instance
PARCEL_AGENTS: dict[str, "ParcelAgent"] = {}

# Registry of all active trade agents: agent_id -> TradeAgent instance
TRADE_AGENTS: dict[str, "TradeAgent"] = {}
