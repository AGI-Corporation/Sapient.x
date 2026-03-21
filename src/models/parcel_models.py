"""Pydantic schemas for Web4AGI REST API.

All request/response bodies are validated here.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ── Metadata ──────────────────────────────────────────────────────────────


class ResponseMetadata(BaseModel):
    """Standard Sapient.x response metadata for observability."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="The UTC ISO8601 timestamp of the response",
    )
    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="A unique identifier for this response (Request/Response correlation)",
    )
    version: str = Field("1.0.0", description="The current OS version")


# ── Location ───────────────────────────────────────────────────────────────


class Location(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lng: float = Field(..., ge=-180, le=180, description="Longitude")
    alt: float = Field(0.0, description="Altitude in meters")


# ── Parcel ─────────────────────────────────────────────────────────────────


class ParcelCreate(BaseModel):
    owner_address: str = Field(..., description="Wallet address of the parcel owner")
    location: Location
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("owner_address")
    @classmethod
    def validate_address(cls, v: str) -> str:
        if not v.startswith("0x") or len(v) < 10:
            raise ValueError("owner_address must be a valid 0x wallet address")
        return v.lower()


class ParcelRead(BaseModel):
    parcel_id: str
    owner: str
    location: Location
    balance_usdx: float
    metadata: dict[str, Any]
    active: bool
    last_updated: str
    meta: ResponseMetadata = Field(default_factory=ResponseMetadata)


class ParcelUpdate(BaseModel):
    metadata: dict[str, Any] | None = None
    active: bool | None = None


# ── Trades ─────────────────────────────────────────────────────────────────


class TradeRequest(BaseModel):
    from_parcel_id: str
    to_parcel_id: str
    amount_usdx: float = Field(..., gt=0, description="Amount in USDx (must be positive)")
    trade_type: str = Field("transfer", description="transfer | lease | data_access")
    contract_terms: dict[str, Any] | None = None


class TradeResponse(BaseModel):
    success: bool
    tx_id: str | None = None
    amount_usdx: float
    from_parcel_id: str
    to_parcel_id: str
    error: str | None = None
    meta: ResponseMetadata = Field(default_factory=ResponseMetadata)


class OfferCreate(BaseModel):
    seller_parcel_id: str
    asset: str = Field(..., description="Asset identifier being sold")
    amount_usdx: float = Field(..., gt=0)
    ttl_seconds: int = Field(300, ge=60, le=86400)


class BidRequest(BaseModel):
    offer_id: str
    bidder_parcel_id: str
    bid_amount_usdx: float = Field(..., gt=0)


# ── Contracts ──────────────────────────────────────────────────────────────


class ContractRequest(BaseModel):
    contract_type: str = Field(..., description="parcel_lease | data_access | custom")
    party_a: str
    party_b: str
    terms: dict[str, Any]


class ContractResponse(BaseModel):
    contract_id: str
    contract_type: str
    status: str
    parties: dict[str, str]
    terms: dict[str, Any]
    created_at: str
    tx_hash: str | None = None
    meta: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ── Optimization ────────────────────────────────────────────────────────────


class OptimizeRequest(BaseModel):
    parcel_id: str
    context: dict[str, Any] = Field(default_factory=dict)


class OptimizeResponse(BaseModel):
    parcel_id: str
    assessment: str | None = None
    strategies: list[str] = []
    chosen_strategy: str | None = None
    actions_taken: list[dict[str, Any]] = []
    reflection: str | None = None
    score: float = 0.0
    meta: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ── Payments ──────────────────────────────────────────────────────────────


class DepositRequest(BaseModel):
    parcel_id: str
    amount_usdx: float = Field(..., gt=0)
    source: str = "stablecoin_bridge"


class PaymentStreamRequest(BaseModel):
    from_parcel_id: str
    to_parcel_id: str
    rate_usdx_per_second: float = Field(..., gt=0)
    duration_seconds: int = Field(..., ge=60)


class IncentiveRequest(BaseModel):
    parcel_id: str
    target_parcel_id: str
    amount_usdx: float = Field(..., gt=0)
    incentive_type: str = "check_in"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── MCP Messages ───────────────────────────────────────────────────────────


class MCPMessage(BaseModel):
    from_parcel_id: str
    to_parcel_id: str
    msg_type: str = Field(..., description="trade_request | contract_offer | optimize | custom")
    payload: dict[str, Any]


class MCPToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


# ── NANDA Agent Registry ───────────────────────────────────────────────────


class AgentFact(BaseModel):
    agent_id: str
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    owner_address: str
    status: str = "active"


class RegistryResponse(BaseModel):
    success: bool
    message: str
    agent: AgentFact | None = None
    meta: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ── Generic responses ───────────────────────────────────────────────────────


class SuccessResponse(BaseModel):
    """Successful operation result with standard metadata."""

    success: bool = Field(True, description="Indicates if the operation was successful")
    message: str = Field("OK", description="A human-readable success message")
    data: Any | None = Field(None, description="The payload of the response")
    meta: ResponseMetadata = Field(
        default_factory=ResponseMetadata, description="Observability metadata"
    )


class ErrorResponse(BaseModel):
    """Standard error response for failed operations."""

    success: bool = Field(False, description="Always false for error responses")
    error: str = Field(..., description="The error category or message")
    detail: str | None = Field(None, description="Detailed trace or diagnostic information")
    meta: ResponseMetadata = Field(
        default_factory=ResponseMetadata, description="Observability metadata"
    )
