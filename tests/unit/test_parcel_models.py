"""Unit tests for Web4AGI Pydantic models (parcel_models.py)."""

import pytest
from pydantic import ValidationError

from src.models.parcel_models import (
    BidRequest,
    ContractRequest,
    ContractResponse,
    DepositRequest,
    ErrorResponse,
    Location,
    MCPMessage,
    MCPToolCall,
    OfferCreate,
    OptimizeRequest,
    OptimizeResponse,
    ParcelCreate,
    ParcelRead,
    ParcelUpdate,
    PaymentStreamRequest,
    SuccessResponse,
    TradeRequest,
    TradeResponse,
)


# ── Location ───────────────────────────────────────────────────────────────


class TestLocation:
    def test_valid_location(self):
        loc = Location(lat=37.7749, lng=-122.4194, alt=10.0)
        assert loc.lat == 37.7749
        assert loc.lng == -122.4194
        assert loc.alt == 10.0

    def test_default_altitude(self):
        loc = Location(lat=0.0, lng=0.0)
        assert loc.alt == 0.0

    def test_boundary_lat_valid(self):
        loc = Location(lat=90.0, lng=0.0)
        assert loc.lat == 90.0
        loc2 = Location(lat=-90.0, lng=0.0)
        assert loc2.lat == -90.0

    def test_boundary_lng_valid(self):
        loc = Location(lat=0.0, lng=180.0)
        assert loc.lng == 180.0
        loc2 = Location(lat=0.0, lng=-180.0)
        assert loc2.lng == -180.0

    def test_lat_out_of_range(self):
        with pytest.raises(ValidationError):
            Location(lat=91.0, lng=0.0)
        with pytest.raises(ValidationError):
            Location(lat=-91.0, lng=0.0)

    def test_lng_out_of_range(self):
        with pytest.raises(ValidationError):
            Location(lat=0.0, lng=181.0)
        with pytest.raises(ValidationError):
            Location(lat=0.0, lng=-181.0)


# ── ParcelCreate ───────────────────────────────────────────────────────────


class TestParcelCreate:
    def test_valid_parcel_create(self):
        pc = ParcelCreate(
            owner_address="0xABCDEF123456",
            location={"lat": 37.7, "lng": -122.4},
        )
        assert pc.owner_address == "0xabcdef123456"  # lowercased
        assert pc.location.lat == 37.7

    def test_owner_address_lowercased(self):
        pc = ParcelCreate(
            owner_address="0xABCDEF123456UPPERCASE",
            location={"lat": 0.0, "lng": 0.0},
        )
        assert pc.owner_address == pc.owner_address.lower()

    def test_missing_0x_prefix(self):
        with pytest.raises(ValidationError) as exc_info:
            ParcelCreate(owner_address="ABC123", location={"lat": 0.0, "lng": 0.0})
        assert "owner_address" in str(exc_info.value)

    def test_address_too_short(self):
        with pytest.raises(ValidationError):
            ParcelCreate(owner_address="0xabc", location={"lat": 0.0, "lng": 0.0})

    def test_with_metadata(self):
        pc = ParcelCreate(
            owner_address="0xABCDEF123456",
            location={"lat": 10.0, "lng": 20.0},
            metadata={"zone": "commercial", "floors": 5},
        )
        assert pc.metadata["zone"] == "commercial"

    def test_default_empty_metadata(self):
        pc = ParcelCreate(
            owner_address="0xABCDEF123456",
            location={"lat": 0.0, "lng": 0.0},
        )
        assert pc.metadata == {}


# ── ParcelRead ─────────────────────────────────────────────────────────────


class TestParcelRead:
    def test_valid_parcel_read(self):
        pr = ParcelRead(
            parcel_id="parcel-001",
            owner="0xowner",
            location={"lat": 1.0, "lng": 2.0},
            balance_usdx=50.0,
            metadata={"key": "val"},
            active=True,
            last_updated="2026-01-01T00:00:00",
        )
        assert pr.parcel_id == "parcel-001"
        assert pr.balance_usdx == 50.0
        assert pr.active is True


# ── ParcelUpdate ───────────────────────────────────────────────────────────


class TestParcelUpdate:
    def test_all_none_is_valid(self):
        pu = ParcelUpdate()
        assert pu.metadata is None
        assert pu.active is None

    def test_update_metadata_only(self):
        pu = ParcelUpdate(metadata={"zone": "sf"})
        assert pu.metadata["zone"] == "sf"
        assert pu.active is None

    def test_update_active_only(self):
        pu = ParcelUpdate(active=False)
        assert pu.active is False
        assert pu.metadata is None


# ── TradeRequest ───────────────────────────────────────────────────────────


class TestTradeRequest:
    def test_valid_trade_request(self):
        tr = TradeRequest(
            from_parcel_id="parcel-001",
            to_parcel_id="parcel-002",
            amount_usdx=10.0,
        )
        assert tr.trade_type == "transfer"
        assert tr.contract_terms is None

    def test_amount_must_be_positive(self):
        with pytest.raises(ValidationError):
            TradeRequest(
                from_parcel_id="parcel-001",
                to_parcel_id="parcel-002",
                amount_usdx=0.0,
            )
        with pytest.raises(ValidationError):
            TradeRequest(
                from_parcel_id="parcel-001",
                to_parcel_id="parcel-002",
                amount_usdx=-5.0,
            )

    def test_with_contract_terms(self):
        tr = TradeRequest(
            from_parcel_id="parcel-001",
            to_parcel_id="parcel-002",
            amount_usdx=100.0,
            trade_type="lease",
            contract_terms={"duration_months": 6},
        )
        assert tr.trade_type == "lease"
        assert tr.contract_terms["duration_months"] == 6


# ── TradeResponse ──────────────────────────────────────────────────────────


class TestTradeResponse:
    def test_success_response(self):
        tr = TradeResponse(
            success=True,
            tx_id="tx-abc",
            amount_usdx=25.0,
            from_parcel_id="p1",
            to_parcel_id="p2",
        )
        assert tr.success is True
        assert tr.error is None

    def test_error_response(self):
        tr = TradeResponse(
            success=False,
            amount_usdx=25.0,
            from_parcel_id="p1",
            to_parcel_id="p2",
            error="Insufficient balance",
        )
        assert tr.success is False
        assert tr.tx_id is None


# ── OfferCreate ────────────────────────────────────────────────────────────


class TestOfferCreate:
    def test_valid_offer(self):
        offer = OfferCreate(
            seller_parcel_id="parcel-001",
            asset="premium_location",
            amount_usdx=200.0,
            ttl_seconds=600,
        )
        assert offer.ttl_seconds == 600

    def test_default_ttl(self):
        offer = OfferCreate(
            seller_parcel_id="parcel-001",
            asset="data_access",
            amount_usdx=50.0,
        )
        assert offer.ttl_seconds == 300

    def test_amount_must_be_positive(self):
        with pytest.raises(ValidationError):
            OfferCreate(seller_parcel_id="p1", asset="x", amount_usdx=0.0)

    def test_ttl_below_minimum(self):
        with pytest.raises(ValidationError):
            OfferCreate(seller_parcel_id="p1", asset="x", amount_usdx=10.0, ttl_seconds=30)

    def test_ttl_above_maximum(self):
        with pytest.raises(ValidationError):
            OfferCreate(seller_parcel_id="p1", asset="x", amount_usdx=10.0, ttl_seconds=90000)


# ── BidRequest ─────────────────────────────────────────────────────────────


class TestBidRequest:
    def test_valid_bid(self):
        bid = BidRequest(
            offer_id="offer-123",
            bidder_parcel_id="parcel-002",
            bid_amount_usdx=55.0,
        )
        assert bid.bid_amount_usdx == 55.0

    def test_bid_amount_must_be_positive(self):
        with pytest.raises(ValidationError):
            BidRequest(offer_id="offer-123", bidder_parcel_id="p2", bid_amount_usdx=0.0)


# ── ContractRequest / ContractResponse ────────────────────────────────────


class TestContractModels:
    def test_contract_request(self):
        cr = ContractRequest(
            contract_type="parcel_lease",
            party_a="0xPartyA",
            party_b="0xPartyB",
            terms={"duration_months": 12, "monthly_usdx": 100.0},
        )
        assert cr.contract_type == "parcel_lease"
        assert cr.terms["duration_months"] == 12

    def test_contract_response(self):
        resp = ContractResponse(
            contract_id="contract-001",
            contract_type="data_access",
            status="pending_signatures",
            parties={"provider": "0xA", "consumer": "0xB"},
            terms={"price_usdx": 25.0},
            created_at="2026-01-01T00:00:00",
        )
        assert resp.tx_hash is None
        assert resp.status == "pending_signatures"

    def test_contract_response_with_tx_hash(self):
        resp = ContractResponse(
            contract_id="c-002",
            contract_type="custom",
            status="signed",
            parties={"a": "0x1", "b": "0x2"},
            terms={},
            created_at="2026-01-01T00:00:00",
            tx_hash="0xdeadbeef",
        )
        assert resp.tx_hash == "0xdeadbeef"


# ── Optimize Models ────────────────────────────────────────────────────────


class TestOptimizeModels:
    def test_optimize_request(self):
        req = OptimizeRequest(parcel_id="parcel-001")
        assert req.context == {}

    def test_optimize_request_with_context(self):
        req = OptimizeRequest(parcel_id="parcel-001", context={"market": "bullish"})
        assert req.context["market"] == "bullish"

    def test_optimize_response_defaults(self):
        resp = OptimizeResponse(parcel_id="parcel-001")
        assert resp.strategies == []
        assert resp.score == 0.0
        assert resp.assessment is None


# ── Payment Models ─────────────────────────────────────────────────────────


class TestPaymentModels:
    def test_deposit_request(self):
        req = DepositRequest(parcel_id="parcel-001", amount_usdx=100.0)
        assert req.source == "stablecoin_bridge"

    def test_deposit_amount_must_be_positive(self):
        with pytest.raises(ValidationError):
            DepositRequest(parcel_id="p1", amount_usdx=0.0)

    def test_payment_stream_request(self):
        req = PaymentStreamRequest(
            from_parcel_id="p1",
            to_parcel_id="p2",
            rate_usdx_per_second=0.01,
            duration_seconds=3600,
        )
        assert req.rate_usdx_per_second == 0.01

    def test_payment_stream_duration_minimum(self):
        with pytest.raises(ValidationError):
            PaymentStreamRequest(
                from_parcel_id="p1",
                to_parcel_id="p2",
                rate_usdx_per_second=0.01,
                duration_seconds=30,
            )

    def test_payment_stream_rate_must_be_positive(self):
        with pytest.raises(ValidationError):
            PaymentStreamRequest(
                from_parcel_id="p1",
                to_parcel_id="p2",
                rate_usdx_per_second=0.0,
                duration_seconds=3600,
            )


# ── MCP Models ─────────────────────────────────────────────────────────────


class TestMCPModels:
    def test_mcp_message(self):
        msg = MCPMessage(
            from_parcel_id="p1",
            to_parcel_id="p2",
            msg_type="trade_request",
            payload={"amount": 10.0},
        )
        assert msg.msg_type == "trade_request"
        assert msg.payload["amount"] == 10.0

    def test_mcp_tool_call(self):
        call = MCPToolCall(tool_name="parcel.get_state")
        assert call.arguments == {}

    def test_mcp_tool_call_with_args(self):
        call = MCPToolCall(
            tool_name="parcel.list_neighbors",
            arguments={"parcel_id": "p1", "radius_meters": 200.0},
        )
        assert call.arguments["radius_meters"] == 200.0


# ── Generic Responses ──────────────────────────────────────────────────────


class TestGenericResponses:
    def test_success_response_defaults(self):
        resp = SuccessResponse()
        assert resp.success is True
        assert resp.message == "OK"
        assert resp.data is None

    def test_success_response_with_data(self):
        resp = SuccessResponse(message="Created", data={"id": "p1"})
        assert resp.data["id"] == "p1"

    def test_error_response(self):
        resp = ErrorResponse(error="Not found")
        assert resp.success is False
        assert resp.detail is None

    def test_error_response_with_detail(self):
        resp = ErrorResponse(error="Validation failed", detail="field 'amount' is required")
        assert resp.detail == "field 'amount' is required"
