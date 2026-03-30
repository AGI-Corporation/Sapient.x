"""Unit tests for ParcelAgent uncovered paths."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.parcel_agent import ParcelAgent


@pytest.fixture
def agent():
    a = ParcelAgent(
        parcel_id="test-ext-001",
        owner_address="0xAbCDEF123456",
        location={"lat": 37.7, "lng": -122.4, "alt": 0.0},
        wallet_private_key="test_key",
    )
    a.mcp.local_only = True
    return a


# ── receive_messages ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_receive_messages_empty_queue(agent):
    """Draining an empty queue returns empty list."""
    messages = await agent.receive_messages()
    assert messages == []


@pytest.mark.asyncio
async def test_receive_messages_with_queued_items(agent):
    """Draining a queue with items returns all of them."""
    agent._message_queue.put_nowait({"type": "trade_request", "amount": 5.0})
    agent._message_queue.put_nowait({"type": "optimize"})

    messages = await agent.receive_messages()
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_receive_messages_drains_completely(agent):
    """After receive_messages, the queue is empty."""
    agent._message_queue.put_nowait({"type": "ping"})
    await agent.receive_messages()
    assert agent._message_queue.empty()


# ── _handle_message ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_message_trade_request(agent):
    """trade_request message triggers a trade call."""
    await agent.deposit(50.0)
    msg = {
        "from": "other-parcel",
        "payload": {
            "type": "trade_request",
            "amount": 10.0,
            "from": "other-parcel",
        },
    }
    with patch.object(agent, "trade", new_callable=AsyncMock) as mock_trade:
        mock_trade.return_value = {"success": True}
        await agent._handle_message(msg)
    mock_trade.assert_called_once_with(
        counterparty_id="other-parcel",
        amount_usdx=10.0,
        trade_type="transfer",
    )


@pytest.mark.asyncio
async def test_handle_message_contract_offer(agent):
    """contract_offer message triggers sign_contract call."""
    contract = {"type": "parcel_lease", "version": "1.0"}
    msg = {
        "from": "seller",
        "payload": {
            "type": "contract_offer",
            "from": "seller",
            "contract": contract,
        },
    }
    with patch.object(agent, "sign_contract", new_callable=AsyncMock) as mock_sign:
        mock_sign.return_value = {"success": True}
        await agent._handle_message(msg)
    mock_sign.assert_called_once_with(counterparty_id="seller", contract=contract)


@pytest.mark.asyncio
async def test_handle_message_optimize(agent):
    """optimize message triggers optimize call."""
    msg = {"payload": {"type": "optimize"}, "context": {"market": "flat"}}
    with patch.object(agent, "optimize", new_callable=AsyncMock) as mock_opt:
        mock_opt.return_value = {"assessment": "ok"}
        await agent._handle_message(msg)
    mock_opt.assert_called_once()


@pytest.mark.asyncio
async def test_handle_message_unknown_type(agent, capsys):
    """Unknown message type prints a warning and does not raise."""
    msg = {"payload": {"type": "unknown_xyz", "from": "agent-x"}}
    await agent._handle_message(msg)  # should not raise
    captured = capsys.readouterr()
    assert "Unknown message type" in captured.out


@pytest.mark.asyncio
async def test_handle_message_no_payload_envelope(agent):
    """Message without 'payload' key uses the message dict itself as data."""
    msg = {"type": "optimize", "from": "agent-x"}
    with patch.object(agent, "optimize", new_callable=AsyncMock) as mock_opt:
        mock_opt.return_value = {}
        await agent._handle_message(msg)
    mock_opt.assert_called_once()


# ── run ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_limited_cycles(agent):
    """run() stops after the specified number of cycles."""
    # Patch asyncio.sleep to avoid waiting
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await agent.run(cycles=2)
    # After run, agent is still active (we just hit the cycle limit)
    assert agent.state.active is True


@pytest.mark.asyncio
async def test_run_processes_queued_messages(agent):
    """run() processes messages from the queue on each cycle."""
    agent._message_queue.put_nowait({"payload": {"type": "optimize", "from": "x"}})

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with patch.object(agent, "_handle_message", new_callable=AsyncMock) as mock_handle:
            await agent.run(cycles=1)

    mock_handle.assert_called_once()


@pytest.mark.asyncio
async def test_run_stops_when_inactive(agent):
    """run() exits immediately when agent.state.active is False."""
    agent.state.active = False
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await agent.run(cycles=0)
    mock_sleep.assert_not_called()


# ── trade with contract_terms ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_trade_with_contract_terms(agent):
    """trade() passes contract_terms through to x402."""
    await agent.deposit(100.0)
    terms = {"duration_months": 6}
    with patch.object(agent.x402, "transfer", new_callable=AsyncMock) as mock_transfer:
        mock_transfer.return_value = {"success": True}
        result = await agent.trade(
            counterparty_id="other",
            amount_usdx=20.0,
            trade_type="lease",
            contract_terms=terms,
        )
    assert result["success"] is True
    call_kwargs = mock_transfer.call_args.kwargs
    assert call_kwargs.get("contract_terms") == terms
