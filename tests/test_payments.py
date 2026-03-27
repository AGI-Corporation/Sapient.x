"""Tests for X402Client in payments/x402_client.py."""

import hashlib
import hmac
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.payments.x402_client import X402Client, _to_micro, make_x402_client


# ── _to_micro Helper ───────────────────────────────────────────────────────────

def test_to_micro_whole_number():
    assert _to_micro(1.0) == 1_000_000


def test_to_micro_fractional():
    assert _to_micro(0.5) == 500_000


def test_to_micro_small_amount():
    assert _to_micro(0.000001) == 1


def test_to_micro_zero():
    assert _to_micro(0.0) == 0


def test_to_micro_large_amount():
    assert _to_micro(1000.0) == 1_000_000_000


# ── X402Client Initialization ─────────────────────────────────────────────────

def test_x402_client_defaults(x402_client):
    """X402Client stores private_key and gateway_url."""
    assert x402_client.private_key == "test_key_abc123_do_not_use_in_production"
    assert x402_client.gateway_url.startswith("https://")


def test_x402_client_strips_trailing_slash():
    """X402Client strips trailing slash from gateway_url."""
    client = X402Client(private_key="k", gateway_url="https://example.com/api/v1/")
    assert not client.gateway_url.endswith("/")


def test_x402_client_empty_private_key():
    """X402Client accepts empty private key (for read-only use)."""
    client = X402Client()
    assert client.private_key == ""


# ── Nonce Generation ──────────────────────────────────────────────────────────

def test_next_nonce_increments(x402_client):
    """_next_nonce returns monotonically increasing values."""
    n1 = x402_client._next_nonce()
    n2 = x402_client._next_nonce()
    n3 = x402_client._next_nonce()
    assert n2 == n1 + 1
    assert n3 == n2 + 1


# ── _sign Tests ────────────────────────────────────────────────────────────────

def test_sign_produces_hex_string(x402_client):
    """_sign returns a 64-character hex string (HMAC-SHA256)."""
    sig = x402_client._sign({"action": "deposit", "amount": 100})
    assert isinstance(sig, str)
    assert len(sig) == 64


def test_sign_is_deterministic(x402_client):
    """_sign produces the same signature for identical payloads."""
    payload = {"action": "deposit", "amount": 100, "nonce": 1}
    sig1 = x402_client._sign(payload)
    sig2 = x402_client._sign(payload)
    assert sig1 == sig2


def test_sign_excludes_signature_key(x402_client):
    """_sign ignores an existing 'signature' key to avoid circular signing."""
    payload_without = {"action": "deposit", "nonce": 1}
    payload_with = {"action": "deposit", "nonce": 1, "signature": "old_sig"}
    assert x402_client._sign(payload_without) == x402_client._sign(payload_with)


def test_sign_changes_with_different_payload(x402_client):
    """_sign produces different signatures for different payloads."""
    sig1 = x402_client._sign({"action": "deposit", "amount": 100})
    sig2 = x402_client._sign({"action": "deposit", "amount": 200})
    assert sig1 != sig2


def test_sign_changes_with_different_key():
    """_sign produces different signatures for different private keys."""
    client1 = X402Client(private_key="key_alpha")
    client2 = X402Client(private_key="key_beta")
    payload = {"action": "deposit", "nonce": 42}
    assert client1._sign(payload) != client2._sign(payload)


# ── _post Simulation Mode ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_post_simulation_without_httpx(x402_client):
    """_post returns a simulated success response when httpx is None."""
    with patch("src.payments.x402_client.httpx", None):
        result = await x402_client._post("deposit", {"amount": 100})
    assert result["success"] is True
    assert result["simulated"] is True
    assert result["endpoint"] == "deposit"


@pytest.mark.asyncio
async def test_get_simulation_without_httpx(x402_client):
    """_get returns a simulated success response when httpx is None."""
    with patch("src.payments.x402_client.httpx", None):
        result = await x402_client._get("balance", {"address": "0xAbc"})
    assert result["success"] is True
    assert result["simulated"] is True


# ── _post with HTTP mock ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_post_sends_json_body(x402_client):
    """_post sends the body dict as JSON to the gateway URL."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"success": True, "tx_id": "abc"}

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_resp
        )
        result = await x402_client._post("deposit", {"amount_micro": 1_000_000})

    assert result["success"] is True


@pytest.mark.asyncio
async def test_get_sends_params(x402_client):
    """_get sends query params to the gateway GET endpoint."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"balance_micro": 5_000_000}

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_resp
        )
        result = await x402_client._get("balance", {"address": "0xAbc"})

    assert result["balance_micro"] == 5_000_000


# ── deposit Tests ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_deposit_payload_structure(x402_client):
    """deposit builds a correctly structured payload with a valid signature."""
    with patch("src.payments.x402_client.httpx", None):
        result = await x402_client.deposit(amount=25.0)
    assert result["simulated"] is True
    # The body stored in the simulated response proves signature was added
    body = result["body"]
    assert body["action"] == "deposit"
    assert body["amount_micro"] == 25_000_000
    assert "signature" in body
    assert "nonce" in body


@pytest.mark.asyncio
async def test_deposit_default_source(x402_client):
    """deposit uses 'stablecoin_bridge' as default source."""
    with patch("src.payments.x402_client.httpx", None):
        result = await x402_client.deposit(amount=10.0)
    assert result["body"]["source"] == "stablecoin_bridge"


@pytest.mark.asyncio
async def test_deposit_custom_source(x402_client):
    """deposit forwards custom source parameter."""
    with patch("src.payments.x402_client.httpx", None):
        result = await x402_client.deposit(amount=10.0, source="wire_transfer")
    assert result["body"]["source"] == "wire_transfer"


# ── transfer Tests ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_transfer_payload_structure(x402_client):
    """transfer builds a correctly structured payload."""
    with patch("src.payments.x402_client.httpx", None):
        result = await x402_client.transfer(
            to_address="0xRecipient",
            amount=50.0,
            memo="test payment",
        )
    body = result["body"]
    assert body["action"] == "transfer"
    assert body["to"] == "0xRecipient"
    assert body["amount_micro"] == 50_000_000
    assert body["memo"] == "test payment"
    assert "signature" in body


@pytest.mark.asyncio
async def test_transfer_includes_contract_terms(x402_client):
    """transfer includes optional contract_terms when provided."""
    terms = {"type": "lease", "duration_months": 6}
    with patch("src.payments.x402_client.httpx", None):
        result = await x402_client.transfer(
            to_address="0xAddr",
            amount=10.0,
            contract_terms=terms,
        )
    assert result["body"]["contract_terms"] == terms


@pytest.mark.asyncio
async def test_transfer_without_contract_terms(x402_client):
    """transfer omits contract_terms key when not provided."""
    with patch("src.payments.x402_client.httpx", None):
        result = await x402_client.transfer(to_address="0xAddr", amount=10.0)
    assert "contract_terms" not in result["body"]


# ── sign_contract Tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sign_contract_payload(x402_client, sample_contract):
    """sign_contract builds a payload with action='sign_contract'."""
    with patch("src.payments.x402_client.httpx", None):
        result = await x402_client.sign_contract(
            contract=sample_contract,
            counterparty="0xCounterparty",
            signer="0xSigner",
        )
    body = result["body"]
    assert body["action"] == "sign_contract"
    assert body["counterparty"] == "0xCounterparty"
    assert body["signer"] == "0xSigner"
    assert body["contract"] == sample_contract
    assert "signature" in body


# ── balance Tests ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_balance_calls_get(x402_client):
    """balance delegates to _get with address parameter."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"balance": 999.9}

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_resp
        )
        result = await x402_client.balance("0xMyAddr")

    assert result["balance"] == 999.9


# ── get_contract Tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_contract_calls_get(x402_client):
    """get_contract delegates to _get with contract_id in the path."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"contract_id": "c-001", "status": "signed"}

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_resp
        )
        result = await x402_client.get_contract("c-001")

    assert result["contract_id"] == "c-001"


# ── stream_payments Tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_payments_payload(x402_client):
    """stream_payments builds the expected payload."""
    with patch("src.payments.x402_client.httpx", None):
        result = await x402_client.stream_payments(
            to_address="0xStream",
            rate_usdx_per_second=0.01,
            duration_seconds=3600,
        )
    body = result["body"]
    assert body["action"] == "stream"
    assert body["to"] == "0xStream"
    assert body["duration_seconds"] == 3600
    assert body["rate_micro_per_second"] == _to_micro(0.01)
    assert "signature" in body


# ── make_x402_client Tests ────────────────────────────────────────────────────

def test_make_x402_client_from_env():
    """make_x402_client builds an X402Client from a dict of env vars."""
    env = {
        "X402_PRIVATE_KEY": "env_key_abc",
        "X402_GATEWAY": "https://custom-gateway.example.com/api/v1",
    }
    client = make_x402_client(env=env)
    assert isinstance(client, X402Client)
    assert client.private_key == "env_key_abc"
    assert client.gateway_url == "https://custom-gateway.example.com/api/v1"


def test_make_x402_client_defaults_when_env_missing():
    """make_x402_client uses defaults when env vars are absent."""
    client = make_x402_client(env={})
    assert client.private_key == ""
    assert client.gateway_url == "https://x402.org/api/v1"
