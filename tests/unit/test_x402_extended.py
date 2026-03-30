"""Unit tests for X402Client uncovered paths."""

import hashlib
from unittest.mock import AsyncMock, patch

import pytest

from src.payments.x402_client import X402Client, _to_micro, make_x402_client


# ── _to_micro helper ───────────────────────────────────────────────────────


def test_to_micro_conversion():
    assert _to_micro(1.0) == 1_000_000
    assert _to_micro(0.5) == 500_000
    assert _to_micro(0.000001) == 1


def test_to_micro_rounding():
    # 1/3 USDx should not cause floating-point issues
    val = _to_micro(1 / 3)
    assert isinstance(val, int)


# ── get_address determinism ────────────────────────────────────────────────


def test_get_address_is_deterministic():
    c1 = X402Client(private_key="same_key")
    c2 = X402Client(private_key="same_key")
    assert c1.get_address() == c2.get_address()


def test_get_address_differs_per_key():
    c1 = X402Client(private_key="key_a")
    c2 = X402Client(private_key="key_b")
    assert c1.get_address() != c2.get_address()


# ── transfer with contract_terms ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_transfer_with_contract_terms():
    client = X402Client(private_key="test_key", local_only=True)
    result = await client.transfer(
        to_address="0xRecipient",
        amount=10.0,
        memo="lease",
        contract_terms={"duration_months": 6},
    )
    assert result["success"] is True
    # contract_terms is included in the posted body (captured in simulated mode)
    assert result["body"]["contract_terms"] == {"duration_months": 6}


@pytest.mark.asyncio
async def test_transfer_without_contract_terms_no_key():
    client = X402Client(private_key="test_key", local_only=True)
    result = await client.transfer(to_address="0xRecipient", amount=5.0)
    assert "contract_terms" not in result["body"]


# ── stream_payments ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_payments_local():
    client = X402Client(private_key="test_key", local_only=True)
    result = await client.stream_payments(
        to_address="0xRecipient",
        rate_usdx_per_second=0.01,
        duration_seconds=3600,
    )
    assert result["success"] is True
    assert result["simulated"] is True
    body = result["body"]
    assert body["action"] == "stream"
    assert body["duration_seconds"] == 3600


# ── get_contract ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_contract_local():
    client = X402Client(private_key="test_key", local_only=True)
    result = await client.get_contract("contract-001")
    assert result["success"] is True


# ── balance raw response ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_balance_local_mode():
    client = X402Client(private_key="test_key", local_only=True)
    result = await client.balance("0xSomeAddress")
    assert result["success"] is True


# ── make_x402_client factory ───────────────────────────────────────────────


def test_make_x402_client_from_env_dict():
    env = {
        "X402_PRIVATE_KEY": "my_private_key",
        "X402_GATEWAY": "https://custom.gateway/api/v1",
    }
    client = make_x402_client(env=env)
    assert client.private_key == "my_private_key"
    assert "custom.gateway" in client.gateway_url


def test_make_x402_client_defaults_when_env_empty():
    # When private_key env var is absent, the client falls back to "default_test_key"
    # because X402Client coerces an empty string to its default value.
    client = make_x402_client(env={})
    assert client.private_key == "default_test_key"


def test_make_x402_client_from_os_environ():
    """Calling make_x402_client() without args reads os.environ."""
    import os

    with patch.dict(os.environ, {"X402_PRIVATE_KEY": "env_key"}):
        client = make_x402_client()
    assert client.private_key == "env_key"


# ── sign_message / sign_transaction ───────────────────────────────────────


def test_sign_message_returns_hex_string():
    client = X402Client(private_key="k")
    sig = client.sign_message("hello world")
    assert isinstance(sig, str)
    assert len(sig) == 64  # SHA-256 hex digest length


def test_sign_transaction_fields():
    client = X402Client(private_key="k")
    tx = {"to": "0xABC", "value": 10.0, "nonce": 5}
    signed = client.sign_transaction(tx)
    assert signed["to"] == "0xABC"
    assert signed["value"] == 10.0
    assert signed["nonce"] == 5
    assert signed["v"] == 27
    assert signed["r"].startswith("0x")
    assert signed["s"].startswith("0x")


# ── verify_signature non-local ─────────────────────────────────────────────


def test_verify_signature_local_always_true():
    client = X402Client(private_key="k", local_only=True)
    assert client.verify_signature("msg", "any_sig", "0xAddress") is True


def test_verify_signature_non_local_correct():
    client = X402Client(private_key="k", local_only=False)
    msg = "test_message"
    sig = client.sign_message(msg)
    assert client.verify_signature(msg, sig, "0xAddress") is True


def test_verify_signature_non_local_wrong_sig():
    client = X402Client(private_key="k", local_only=False)
    assert client.verify_signature("msg", "wrong_sig", "0xAddress") is False


# ── sign_contract ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sign_contract_local():
    client = X402Client(private_key="test_key", local_only=True)
    contract = {"type": "parcel_lease", "version": "1.0", "terms": {}}
    result = await client.sign_contract(
        contract=contract, counterparty="0xOther", signer="0xOwner"
    )
    assert result["success"] is True
    assert result["body"]["action"] == "sign_contract"


# ── nonce increments ───────────────────────────────────────────────────────


def test_nonce_increments_on_each_call():
    client = X402Client(private_key="k")
    n1 = client._next_nonce()
    n2 = client._next_nonce()
    n3 = client._next_nonce()
    assert n2 == n1 + 1
    assert n3 == n2 + 1


# ── get_balance local ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_balance_local_returns_simulated():
    client = X402Client(private_key="k", local_only=True)
    balance = await client.get_balance("0xAddr")
    assert balance == 1000.0
