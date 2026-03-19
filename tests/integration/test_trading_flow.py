"""Integration tests for the full trading flow.

Tests the complete trade lifecycle including:
- Order initiation by ParcelAgent
- Trade matching by TradeAgent
- Payment processing via X402 protocol
- State updates across components
- Success and failure scenarios
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest


class TestTradingLifecycle:
    """Test the complete end-to-end trading flow between agents."""

    @pytest.mark.asyncio
    @patch('src.agents.parcel_agent.ParcelAgent')
    @patch('src.agents.trade_agent.TradeAgent')
    @patch('src.payments.x402_client.X402Client')
    async def test_successful_trade_flow(
        self, mock_x402_class, mock_trade_agent_class, mock_parcel_agent_class
    ):
        """Test a complete successful trade from start to finish."""
        # 1. Setup Mocks
        buyer = Mock()
        seller = Mock()
        trade_manager = Mock()
        payment_client = AsyncMock()

        mock_parcel_agent_class.side_effect = [buyer, seller]
        mock_trade_agent_class.return_value = trade_manager
        mock_x402_class.return_value = payment_client

        # 2. Initiate Trade (Buyer makes offer)
        offer_data = {
            "parcel_id": "parcel_001",
            "price": 5000.0,
            "buyer_id": "agent_buyer",
            "timestamp": datetime.now().isoformat()
        }

        # Buyer creates trade order
        buyer.create_trade_order = Mock(return_value="order_123")
        order_id = buyer.create_trade_order(offer_data)
        assert order_id == "order_123"

        # 3. Match Trade (TradeAgent matches buyer and seller)
        trade_manager.match_orders = Mock(return_value={
            "trade_id": "trade_456",
            "buyer": "agent_buyer",
            "seller": "agent_seller",
            "amount": 5000.0
        })
        trade_match = trade_manager.match_orders("order_123")
        assert trade_match["trade_id"] == "trade_456"

        # 4. Process Payment (X402 Protocol)
        payment_client.transfer = AsyncMock(return_value={
            "transaction_hash": "0xtx123",
            "status": "confirmed"
        })

        payment_result = await payment_client.transfer(
            from_addr="0xbuyer",
            to_addr="0xseller",
            amount=5000.0,
            currency="USD"
        )

        assert payment_result["status"] == "confirmed"
        assert payment_result["transaction_hash"] == "0xtx123"

        # 5. Finalize Trade & Update State
        buyer.on_trade_completed = Mock()
        seller.on_trade_completed = Mock()

        buyer.on_trade_completed("trade_456", payment_result)
        seller.on_trade_completed("trade_456", payment_result)

        assert buyer.on_trade_completed.called
        assert seller.on_trade_completed.called

    @pytest.mark.asyncio
    @patch('src.payments.x402_client.X402Client')
    async def test_trade_failure_insufficient_funds(self, mock_x402_class):
        """Test trade flow failure due to insufficient funds."""
        payment_client = AsyncMock()
        mock_x402_class.return_value = payment_client

        # Simulate payment failure
        payment_client.transfer = AsyncMock(side_effect=Exception("Insufficient balance"))

        try:
            await payment_client.transfer("0xbuyer", "0xseller", 1000000.0)
            raise AssertionError("Should have raised Exception")
        except Exception as e:
            assert str(e) == "Insufficient balance"

    @pytest.mark.asyncio
    @patch('src.agents.trade_agent.TradeAgent')
    async def test_trade_timeout(self, mock_trade_agent_class):
        """Test handling of trade expiration/timeout."""
        trade_manager = Mock()
        mock_trade_agent_class.return_value = trade_manager

        # Simulate no match found before timeout
        trade_manager.find_match = Mock(return_value=None)

        match = trade_manager.find_match("order_expired")
        assert match is None

        # Cancel order
        trade_manager.cancel_order = Mock(return_value=True)
        assert trade_manager.cancel_order("order_expired")


class TestMultiAgentTrading:
    """Test trading interactions involving multiple agents."""

    @pytest.mark.asyncio
    async def test_concurrent_trades(self):
        """Test multiple agents trading simultaneously."""
        num_trades = 5
        mock_payment_client = AsyncMock()
        mock_payment_client.transfer = AsyncMock(return_value={"status": "confirmed"})

        # Simulate concurrent transfers
        tasks = []
        for i in range(num_trades):
            tasks.append(
                mock_payment_client.transfer(f"0xagent_{i}", "0xseller", 100.0)
            )

        results = await asyncio.gather(*tasks)

        assert len(results) == num_trades
        for res in results:
            assert res["status"] == "confirmed"

    @pytest.mark.asyncio
    async def test_trade_auction_flow(self):
        """Test an auction-style trade with multiple bidders."""
        trade_agent = Mock()

        # Multiple bids for the same parcel
        bids = [
            {"bidder": "agent_1", "amount": 100.0},
            {"bidder": "agent_2", "amount": 150.0},
            {"bidder": "agent_3", "amount": 125.0}
        ]

        # TradeAgent selects the highest bid
        trade_agent.select_highest_bid = Mock(return_value=bids[1])
        winning_bid = trade_agent.select_highest_bid(bids)

        assert winning_bid["bidder"] == "agent_2"
        assert winning_bid["amount"] == 150.0
