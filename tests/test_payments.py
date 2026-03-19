"""Tests for X402 payment client and wallet operations."""

from unittest.mock import AsyncMock, patch

import pytest

# ── X402Client Tests ─────────────────────────────────────────────────────────

def test_x402_client_creation(x402_client):
    """Test X402Client initialization."""
    assert x402_client is not None
    assert hasattr(x402_client, 'private_key')


def test_x402_client_wallet_address(x402_client):
    """Test wallet address generation."""
    address = x402_client.get_address()
    assert address.startswith('0x')
    assert len(address) == 42  # Ethereum address length


@pytest.mark.asyncio
async def test_x402_get_balance(x402_client, test_wallet_address):
    """Test USDx balance query."""
    # Mock the blockchain call
    with patch.object(x402_client, '_query_balance', new_callable=AsyncMock) as mock_query:
        mock_query.return_value = 100.5

        balance = await x402_client.get_balance(test_wallet_address)
        assert balance == 100.5
        mock_query.assert_called_once()


@pytest.mark.asyncio
async def test_x402_create_payment(x402_client):
    """Test creating a payment transaction."""
    payment = await x402_client.create_payment(
        to_address="0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",
        amount_usdx=50.0,
        memo="Test payment"
    )

    assert payment["success"] is True
    assert "tx_hash" in payment
    assert payment["amount"] == 50.0


@pytest.mark.asyncio
async def test_x402_sign_transaction(x402_client):
    """Test transaction signing."""
    tx_data = {
        "to": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",
        "value": 25.0,
        "nonce": 1,
    }

    signed_tx = x402_client.sign_transaction(tx_data)
    assert "signature" in signed_tx
    assert "r" in signed_tx
    assert "s" in signed_tx
    assert "v" in signed_tx


@pytest.mark.asyncio
async def test_x402_verify_signature(x402_client):
    """Test signature verification."""
    message = "Test message for signing"
    signature = x402_client.sign_message(message)

    is_valid = x402_client.verify_signature(
        message=message,
        signature=signature,
        signer_address=x402_client.get_address()
    )

    assert is_valid is True


@pytest.mark.asyncio
async def test_x402_batch_payment(x402_client):
    """Test batch payment processing."""
    payments = [
        {"to": "0xAddress1", "amount": 10.0},
        {"to": "0xAddress2", "amount": 15.0},
        {"to": "0xAddress3", "amount": 20.0},
    ]

    with patch.object(x402_client, 'create_payment', new_callable=AsyncMock) as mock_pay:
        mock_pay.return_value = {"success": True, "tx_hash": "0xhash"}

        results = await x402_client.batch_payment(payments)
        assert len(results) == 3
        assert all(r["success"] for r in results)
        assert mock_pay.call_count == 3


def test_x402_encode_function_call(x402_client):
    """Test smart contract function encoding."""
    encoded = x402_client.encode_function(
        function_name="transfer",
        params=["0xRecipient", 100]
    )

    assert isinstance(encoded, str)
    assert encoded.startswith("0x")


@pytest.mark.asyncio
async def test_x402_transaction_history(x402_client, test_wallet_address):
    """Test transaction history retrieval."""
    with patch.object(x402_client, '_fetch_history', new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = [
            {"tx_hash": "0xhash1", "amount": 50.0, "type": "sent"},
            {"tx_hash": "0xhash2", "amount": 75.0, "type": "received"},
        ]

        history = await x402_client.get_transaction_history(test_wallet_address)
        assert len(history) == 2
        assert history[0]["type"] == "sent"
        assert history[1]["type"] == "received"


@pytest.mark.asyncio
async def test_x402_gas_estimation(x402_client):
    """Test gas price estimation for transactions."""
    gas_price = await x402_client.estimate_gas(
        to_address="0xRecipient",
        amount_usdx=100.0
    )

    assert isinstance(gas_price, (int, float))
    assert gas_price > 0


def test_x402_invalid_address(x402_client):
    """Test handling of invalid addresses."""
    with pytest.raises(ValueError):
        x402_client.validate_address("invalid_address")


@pytest.mark.asyncio
async def test_x402_insufficient_balance(x402_client):
    """Test payment with insufficient balance."""
    with patch.object(x402_client, 'get_balance', new_callable=AsyncMock) as mock_balance:
        mock_balance.return_value = 10.0

        result = await x402_client.create_payment(
            to_address="0xRecipient",
            amount_usdx=100.0  # More than balance
        )

        assert result["success"] is False
        assert "insufficient" in result["error"].lower()
