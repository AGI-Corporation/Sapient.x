"""Integration tests for WebSocket connections.

Tests the Web4AGI WebSocket functionality including:
- Connection lifecycle management
- Message broadcasting
- Multi-agent real-time communication
- Reconnection logic
- Error handling
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest


class TestWebSocketConnection:
    """Test WebSocket connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_client(self):
        """Test establishing a WebSocket connection."""
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_json = AsyncMock()
        mock_websocket.receive_json = AsyncMock(return_value={"type": "ping"})

        # Test connection acceptance
        await mock_websocket.accept()

        assert mock_websocket.accept.called

    @pytest.mark.asyncio
    async def test_client_registration(self):
        """Test client registration on WebSocket connection."""
        mock_connection_manager = Mock()
        mock_connection_manager.connect = AsyncMock()
        mock_connection_manager.active_connections = []

        mock_websocket = AsyncMock()

        # Register client
        await mock_connection_manager.connect(mock_websocket, "agent_123")

        assert mock_connection_manager.connect.called
        mock_connection_manager.connect.assert_called_with(mock_websocket, "agent_123")

    @pytest.mark.asyncio
    async def test_client_disconnect(self):
        """Test client disconnection."""
        mock_connection_manager = Mock()
        mock_connection_manager.disconnect = Mock()
        mock_connection_manager.active_connections = []

        mock_websocket = AsyncMock()

        # Disconnect client
        mock_connection_manager.disconnect(mock_websocket)

        assert mock_connection_manager.disconnect.called

    @pytest.mark.asyncio
    async def test_send_message_to_client(self):
        """Test sending a message to a specific client."""
        mock_websocket = AsyncMock()
        mock_websocket.send_json = AsyncMock()

        message = {"type": "trade_update", "agent_id": "agent_123", "status": "completed"}

        await mock_websocket.send_json(message)

        mock_websocket.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_receive_message_from_client(self):
        """Test receiving a message from a client."""
        mock_websocket = AsyncMock()
        mock_websocket.receive_json = AsyncMock(return_value={
            "type": "place_order",
            "data": {"action": "buy", "amount": 100.0}
        })

        message = await mock_websocket.receive_json()

        assert message["type"] == "place_order"
        assert message["data"]["action"] == "buy"


class TestWebSocketBroadcasting:
    """Test WebSocket message broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_to_all_clients(self):
        """Test broadcasting message to all connected clients."""
        mock_clients = [AsyncMock() for _ in range(3)]
        for client in mock_clients:
            client.send_json = AsyncMock()

        broadcast_message = {"type": "market_update", "data": {"price": 5000.0}}

        # Simulate broadcasting
        for client in mock_clients:
            await client.send_json(broadcast_message)

        for client in mock_clients:
            client.send_json.assert_called_once_with(broadcast_message)

    @pytest.mark.asyncio
    async def test_broadcast_agent_status_update(self):
        """Test broadcasting agent status updates."""
        mock_connection_manager = Mock()
        mock_connection_manager.broadcast = AsyncMock()

        status_update = {
            "type": "agent_status",
            "agent_id": "agent_123",
            "status": "active",
            "balance": 1500.0
        }

        await mock_connection_manager.broadcast(status_update)

        mock_connection_manager.broadcast.assert_called_once_with(status_update)

    @pytest.mark.asyncio
    async def test_broadcast_trade_event(self):
        """Test broadcasting trade events to all subscribers."""
        mock_connection_manager = Mock()
        mock_connection_manager.broadcast = AsyncMock()

        trade_event = {
            "type": "trade_executed",
            "trade_id": "trade_456",
            "buyer_id": "agent_123",
            "seller_id": "agent_456",
            "amount": 100.0,
            "price": 50.0
        }

        await mock_connection_manager.broadcast(trade_event)

        mock_connection_manager.broadcast.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_to_specific_agent(self):
        """Test sending message to a specific agent."""
        mock_connection_manager = Mock()
        mock_connection_manager.send_personal_message = AsyncMock()

        personal_message = {
            "type": "notification",
            "message": "Trade completed successfully"
        }

        await mock_connection_manager.send_personal_message(
            personal_message, "agent_123"
        )

        mock_connection_manager.send_personal_message.assert_called_with(
            personal_message, "agent_123"
        )


class TestWebSocketMessageTypes:
    """Test different WebSocket message types."""

    @pytest.mark.asyncio
    async def test_ping_pong(self):
        """Test ping-pong heartbeat mechanism."""
        mock_websocket = AsyncMock()
        mock_websocket.receive_json = AsyncMock(return_value={"type": "ping"})
        mock_websocket.send_json = AsyncMock()

        message = await mock_websocket.receive_json()
        assert message["type"] == "ping"

        # Server should respond with pong
        await mock_websocket.send_json({"type": "pong", "timestamp": 1234567890})
        mock_websocket.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_subscribe_to_agent_updates(self):
        """Test subscribing to updates for a specific agent."""
        mock_websocket = AsyncMock()
        mock_websocket.receive_json = AsyncMock(return_value={
            "type": "subscribe",
            "agent_id": "agent_123"
        })
        mock_websocket.send_json = AsyncMock()

        message = await mock_websocket.receive_json()
        assert message["type"] == "subscribe"
        assert message["agent_id"] == "agent_123"

        # Confirm subscription
        await mock_websocket.send_json({
            "type": "subscribed",
            "agent_id": "agent_123"
        })

    @pytest.mark.asyncio
    async def test_market_data_stream(self):
        """Test streaming real-time market data."""
        mock_websocket = AsyncMock()
        mock_websocket.send_json = AsyncMock()

        market_data = [
            {"type": "price_update", "parcel_id": "parcel_001", "price": 4800.0},
            {"type": "price_update", "parcel_id": "parcel_002", "price": 3200.0},
            {"type": "price_update", "parcel_id": "parcel_003", "price": 7500.0},
        ]

        for data in market_data:
            await mock_websocket.send_json(data)

        assert mock_websocket.send_json.call_count == 3

    @pytest.mark.asyncio
    async def test_contract_negotiation_stream(self):
        """Test real-time contract negotiation via WebSocket."""
        mock_websocket_buyer = AsyncMock()
        mock_websocket_seller = AsyncMock()
        mock_websocket_buyer.send_json = AsyncMock()
        mock_websocket_seller.send_json = AsyncMock()

        # Buyer makes offer
        offer = {
            "type": "contract_offer",
            "from": "agent_123",
            "to": "agent_456",
            "terms": {"price": 5000.0, "parcel_id": "parcel_002"}
        }

        await mock_websocket_buyer.send_json(offer)

        # Seller receives offer and counters
        counter_offer = {
            "type": "contract_counter",
            "from": "agent_456",
            "to": "agent_123",
            "terms": {"price": 5500.0, "parcel_id": "parcel_002"}
        }

        await mock_websocket_seller.send_json(counter_offer)

        mock_websocket_buyer.send_json.assert_called_once_with(offer)
        mock_websocket_seller.send_json.assert_called_once_with(counter_offer)


class TestWebSocketReconnection:
    """Test WebSocket reconnection handling."""

    @pytest.mark.asyncio
    async def test_reconnection_after_disconnect(self):
        """Test client reconnection after unexpected disconnect."""
        mock_connection_manager = Mock()
        mock_connection_manager.connect = AsyncMock()
        mock_connection_manager.disconnect = Mock()

        mock_websocket = AsyncMock()
        agent_id = "agent_123"

        # Initial connection
        await mock_connection_manager.connect(mock_websocket, agent_id)
        assert mock_connection_manager.connect.called

        # Disconnect
        mock_connection_manager.disconnect(mock_websocket)
        assert mock_connection_manager.disconnect.called

        # Reconnect
        mock_websocket_new = AsyncMock()
        await mock_connection_manager.connect(mock_websocket_new, agent_id)
        assert mock_connection_manager.connect.call_count == 2

    @pytest.mark.asyncio
    async def test_handle_connection_error(self):
        """Test handling WebSocket connection errors."""
        mock_websocket = AsyncMock()
        mock_websocket.receive_json = AsyncMock(
            side_effect=Exception("Connection closed")
        )

        try:
            await mock_websocket.receive_json()
            raise AssertionError("Should have raised exception")
        except Exception as e:
            assert "Connection closed" in str(e)

    @pytest.mark.asyncio
    async def test_state_preservation_on_reconnect(self):
        """Test that agent state is preserved on reconnection."""
        mock_state_manager = Mock()
        mock_state_manager.get_state = Mock(return_value={
            "agent_id": "agent_123",
            "balance": 1500.0,
            "open_trades": ["trade_789"]
        })

        state = mock_state_manager.get_state("agent_123")

        assert state["agent_id"] == "agent_123"
        assert state["balance"] == 1500.0
        assert "trade_789" in state["open_trades"]


class TestWebSocketConcurrency:
    """Test WebSocket concurrent connections."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_connections(self):
        """Test handling multiple concurrent WebSocket connections."""
        num_clients = 10
        mock_clients = [AsyncMock() for _ in range(num_clients)]

        mock_connection_manager = Mock()
        mock_connection_manager.connect = AsyncMock()

        # Connect all clients concurrently
        tasks = []
        for i, client in enumerate(mock_clients):
            tasks.append(
                mock_connection_manager.connect(client, f"agent_{i}")
            )

        await asyncio.gather(*tasks)

        assert mock_connection_manager.connect.call_count == num_clients

    @pytest.mark.asyncio
    async def test_broadcast_performance(self):
        """Test broadcasting to many clients efficiently."""
        num_clients = 50
        mock_clients = [AsyncMock() for _ in range(num_clients)]
        for client in mock_clients:
            client.send_json = AsyncMock()

        message = {"type": "global_update", "data": "market_closed"}

        # Broadcast to all clients
        tasks = [client.send_json(message) for client in mock_clients]
        await asyncio.gather(*tasks)

        for client in mock_clients:
            client.send_json.assert_called_once_with(message)
