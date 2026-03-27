"""Unit tests for Pydantic models in src/models/parcel_models.py."""

import pytest
from pydantic import ValidationError

from src.models.parcel_models import (
    Location,
    ParcelCreate,
    ParcelRead,
    ParcelUpdate,
    TradeRequest,
    TradeResponse,
    OfferCreate,
    BidRequest,
    ContractRequest,
    ContractResponse,
    OptimizeRequest,
    OptimizeResponse,
    DepositRequest,
    PaymentStreamRequest,
    MCPMessage,
    MCPToolCall,
    SuccessResponse,
    ErrorResponse,
)


# ── Location ───────────────────────────────────────────────────────────────────

def test_location_valid():
    loc = Location(lat=37.7749, lng=-122.4194, alt=10.0)
    assert loc.lat == 37.7749
    assert loc.alt == 10.0


def test_location_default_alt():
    loc = Location(lat=0.0, lng=0.0)
    assert loc.alt == 0.0


def test_location_lat_out_of_range():
    with pytest.raises(ValidationError):
        Location(lat=91.0, lng=0.0)


def test_location_lat_min_boundary():
    with pytest.raises(ValidationError):
        Location(lat=-91.0, lng=0.0)


def test_location_lng_out_of_range():
    with pytest.raises(ValidationError):
        Location(lat=0.0, lng=181.0)


def test_location_lng_min_boundary():
    with pytest.raises(ValidationError):
        Location(lat=0.0, lng=-181.0)


def test_location_boundary_values():
    loc = Location(lat=90.0, lng=180.0)
    assert loc.lat == 90.0


# ── ParcelCreate ───────────────────────────────────────────────────────────────

def test_parcel_create_valid():
    pc = ParcelCreate(
        owner_address="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
        location=Location(lat=37.0, lng=-122.0),
    )
    assert pc.owner_address == "0x742d35cc6634c0532925a3b844bc9e7595f0beb"


def test_parcel_create_validates_address_lowercase():
    """owner_address is normalized to lowercase."""
    pc = ParcelCreate(
        owner_address="0xABCDEF1234567890ABCDEF1234567890ABCDEF12",
        location=Location(lat=0.0, lng=0.0),
    )
    assert pc.owner_address == pc.owner_address.lower()


def test_parcel_create_rejects_non_0x_address():
    with pytest.raises(ValidationError):
        ParcelCreate(
            owner_address="invalid_address",
            location=Location(lat=0.0, lng=0.0),
        )


def test_parcel_create_rejects_short_0x_address():
    with pytest.raises(ValidationError):
        ParcelCreate(
            owner_address="0x123",
            location=Location(lat=0.0, lng=0.0),
        )


def test_parcel_create_default_metadata():
    pc = ParcelCreate(
        owner_address="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
        location=Location(lat=0.0, lng=0.0),
    )
    assert pc.metadata == {}


def test_parcel_create_with_metadata():
    pc = ParcelCreate(
        owner_address="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
        location=Location(lat=0.0, lng=0.0),
        metadata={"zone": "downtown", "tier": 1},
    )
    assert pc.metadata["zone"] == "downtown"


# ── ParcelRead ─────────────────────────────────────────────────────────────────

def test_parcel_read_valid():
    pr = ParcelRead(
        parcel_id="p-001",
        owner="0xOwner",
        location=Location(lat=0.0, lng=0.0),
        balance_usdx=100.0,
        metadata={},
        active=True,
        last_updated="2026-01-01T00:00:00",
    )
    assert pr.parcel_id == "p-001"
    assert pr.balance_usdx == 100.0


# ── ParcelUpdate ───────────────────────────────────────────────────────────────

def test_parcel_update_all_optional():
    pu = ParcelUpdate()
    assert pu.metadata is None
    assert pu.active is None


def test_parcel_update_metadata_only():
    pu = ParcelUpdate(metadata={"zone": "residential"})
    assert pu.metadata["zone"] == "residential"
    assert pu.active is None


def test_parcel_update_active_only():
    pu = ParcelUpdate(active=False)
    assert pu.active is False
    assert pu.metadata is None


# ── TradeRequest ───────────────────────────────────────────────────────────────

def test_trade_request_valid():
    tr = TradeRequest(
        from_parcel_id="p-001",
        to_parcel_id="p-002",
        amount_usdx=50.0,
    )
    assert tr.trade_type == "transfer"
    assert tr.contract_terms is None


def test_trade_request_amount_must_be_positive():
    with pytest.raises(ValidationError):
        TradeRequest(from_parcel_id="p-001", to_parcel_id="p-002", amount_usdx=0.0)


def test_trade_request_negative_amount_rejected():
    with pytest.raises(ValidationError):
        TradeRequest(from_parcel_id="p-001", to_parcel_id="p-002", amount_usdx=-5.0)


def test_trade_request_with_contract_terms():
    tr = TradeRequest(
        from_parcel_id="p-001",
        to_parcel_id="p-002",
        amount_usdx=10.0,
        trade_type="lease",
        contract_terms={"duration_months": 6},
    )
    assert tr.contract_terms["duration_months"] == 6


# ── TradeResponse ─────────────────────────────────────────────────────────────

def test_trade_response_success():
    tr = TradeResponse(
        success=True,
        amount_usdx=25.0,
        from_parcel_id="p-001",
        to_parcel_id="p-002",
        tx_id="0xabc",
    )
    assert tr.error is None


def test_trade_response_failure():
    tr = TradeResponse(
        success=False,
        amount_usdx=0.0,
        from_parcel_id="p-001",
        to_parcel_id="p-002",
        error="Insufficient balance",
    )
    assert "Insufficient" in tr.error


# ── OfferCreate ────────────────────────────────────────────────────────────────

def test_offer_create_valid():
    oc = OfferCreate(seller_parcel_id="p-001", asset="land", amount_usdx=100.0)
    assert oc.ttl_seconds == 300


def test_offer_create_ttl_minimum():
    with pytest.raises(ValidationError):
        OfferCreate(seller_parcel_id="p-001", asset="land", amount_usdx=100.0, ttl_seconds=59)


def test_offer_create_ttl_maximum():
    with pytest.raises(ValidationError):
        OfferCreate(
            seller_parcel_id="p-001", asset="land", amount_usdx=100.0, ttl_seconds=86401
        )


def test_offer_create_amount_must_be_positive():
    with pytest.raises(ValidationError):
        OfferCreate(seller_parcel_id="p-001", asset="land", amount_usdx=0.0)


# ── BidRequest ─────────────────────────────────────────────────────────────────

def test_bid_request_valid():
    br = BidRequest(offer_id="o-001", bidder_parcel_id="p-002", bid_amount_usdx=110.0)
    assert br.bid_amount_usdx == 110.0


def test_bid_request_amount_must_be_positive():
    with pytest.raises(ValidationError):
        BidRequest(offer_id="o-001", bidder_parcel_id="p-002", bid_amount_usdx=0.0)


# ── ContractRequest ───────────────────────────────────────────────────────────

def test_contract_request_valid():
    cr = ContractRequest(
        contract_type="parcel_lease",
        party_a="0xA",
        party_b="0xB",
        terms={"duration_months": 12},
    )
    assert cr.contract_type == "parcel_lease"


# ── ContractResponse ──────────────────────────────────────────────────────────

def test_contract_response_optional_tx_hash():
    cr = ContractResponse(
        contract_id="c-001",
        contract_type="parcel_lease",
        status="signed",
        parties={"lessor": "0xA", "lessee": "0xB"},
        terms={},
        created_at="2026-01-01T00:00:00",
    )
    assert cr.tx_hash is None


# ── OptimizeRequest ────────────────────────────────────────────────────────────

def test_optimize_request_default_context():
    req = OptimizeRequest(parcel_id="p-001")
    assert req.context == {}


def test_optimize_request_with_context():
    req = OptimizeRequest(parcel_id="p-001", context={"market": "bullish"})
    assert req.context["market"] == "bullish"


# ── OptimizeResponse ──────────────────────────────────────────────────────────

def test_optimize_response_defaults():
    resp = OptimizeResponse(parcel_id="p-001")
    assert resp.score == 0.0
    assert resp.strategies == []
    assert resp.actions_taken == []
    assert resp.assessment is None


# ── DepositRequest ────────────────────────────────────────────────────────────

def test_deposit_request_valid():
    dr = DepositRequest(parcel_id="p-001", amount_usdx=50.0)
    assert dr.source == "stablecoin_bridge"


def test_deposit_request_amount_must_be_positive():
    with pytest.raises(ValidationError):
        DepositRequest(parcel_id="p-001", amount_usdx=0.0)


# ── PaymentStreamRequest ──────────────────────────────────────────────────────

def test_payment_stream_request_valid():
    psr = PaymentStreamRequest(
        from_parcel_id="p-001",
        to_parcel_id="p-002",
        rate_usdx_per_second=0.001,
        duration_seconds=3600,
    )
    assert psr.duration_seconds == 3600


def test_payment_stream_request_duration_minimum():
    with pytest.raises(ValidationError):
        PaymentStreamRequest(
            from_parcel_id="p-001",
            to_parcel_id="p-002",
            rate_usdx_per_second=0.001,
            duration_seconds=59,
        )


def test_payment_stream_rate_must_be_positive():
    with pytest.raises(ValidationError):
        PaymentStreamRequest(
            from_parcel_id="p-001",
            to_parcel_id="p-002",
            rate_usdx_per_second=0.0,
            duration_seconds=3600,
        )


# ── MCPMessage ────────────────────────────────────────────────────────────────

def test_mcp_message_valid():
    msg = MCPMessage(
        from_parcel_id="p-001",
        to_parcel_id="p-002",
        msg_type="trade_request",
        payload={"amount": 10.0},
    )
    assert msg.msg_type == "trade_request"
    assert msg.payload["amount"] == 10.0


# ── MCPToolCall ───────────────────────────────────────────────────────────────

def test_mcp_tool_call_default_arguments():
    tc = MCPToolCall(tool_name="parcel.get_state")
    assert tc.arguments == {}


def test_mcp_tool_call_with_arguments():
    tc = MCPToolCall(tool_name="parcel.get_state", arguments={"parcel_id": "p-001"})
    assert tc.arguments["parcel_id"] == "p-001"


# ── SuccessResponse / ErrorResponse ──────────────────────────────────────────

def test_success_response_defaults():
    resp = SuccessResponse()
    assert resp.success is True
    assert resp.message == "OK"
    assert resp.data is None


def test_success_response_with_data():
    resp = SuccessResponse(data={"key": "value"})
    assert resp.data["key"] == "value"


def test_error_response_defaults():
    resp = ErrorResponse(error="Something went wrong")
    assert resp.success is False
    assert resp.detail is None


def test_error_response_with_detail():
    resp = ErrorResponse(error="Not found", detail="Parcel p-999 does not exist")
    assert "p-999" in resp.detail
