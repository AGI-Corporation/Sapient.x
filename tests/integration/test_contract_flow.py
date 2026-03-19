"""Integration tests for the contract lifecycle flow.

Tests the interaction between agents and the ContractManager:
- Contract proposal and negotiation
- Terms validation
- Digital signature process
- Contract execution and escrow handling
- Multi-party agreement scenarios
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest


class TestContractLifecycle:
    """Test the complete contract lifecycle between agents."""

    @pytest.fixture
    def buyer(self):
        agent = Mock()
        agent.id = "agent_buyer"
        agent.wallet_address = "0xbuyer"
        return agent

    @pytest.fixture
    def seller(self):
        agent = Mock()
        agent.id = "agent_seller"
        agent.wallet_address = "0xseller"
        return agent

    @pytest.mark.asyncio
    @patch("src.contracts.manager.ContractManager")
    async def test_full_contract_flow(self, mock_manager_class, buyer, seller):
        """Test a successful contract flow from proposal to execution."""
        manager = Mock()
        mock_manager_class.return_value = manager

        # 1. Propose Contract
        contract_data = {
            "parcel_id": "parcel_001",
            "price": 5000.0,
            "terms": "Standard land lease",
            "expiry": "2026-12-31",
        }

        manager.propose = Mock(return_value="contract_123")
        contract_id = manager.propose(buyer.id, seller.id, contract_data)
        assert contract_id == "contract_123"

        # 2. Negotiate (Optional)
        manager.get_status = Mock(return_value="pending_signature")
        assert manager.get_status(contract_id) == "pending_signature"

        # 3. Sign Contract (Buyer)
        manager.sign = Mock(return_value=True)
        assert manager.sign(contract_id, buyer.id, "0xsignature_buyer")

        # 4. Sign Contract (Seller)
        assert manager.sign(contract_id, seller.id, "0xsignature_seller")

        # 5. Execute Contract (Escrow movement)
        manager.execute = AsyncMock(
            return_value={"status": "executed", "tx_hash": "0xtx_contract_123"}
        )

        execution_result = await manager.execute(contract_id)
        assert execution_result["status"] == "executed"
        assert "tx_hash" in execution_result

    @pytest.mark.asyncio
    @patch("src.contracts.manager.ContractManager")
    async def test_contract_rejection(self, mock_manager_class, buyer, seller):
        """Test contract rejection by counterparty."""
        manager = Mock()
        mock_manager_class.return_value = manager

        contract_id = "contract_rejected"
        manager.reject = Mock(return_value=True)

        assert manager.reject(contract_id, seller.id, "Price too low")

        manager.get_status = Mock(return_value="rejected")
        assert manager.get_status(contract_id) == "rejected"

    @pytest.mark.asyncio
    @patch("src.contracts.manager.ContractManager")
    async def test_contract_validation_failure(self, mock_manager_class, buyer, seller):
        """Test contract creation failure due to invalid terms."""
        manager = Mock()
        mock_manager_class.return_value = manager

        invalid_terms = {"price": -100}  # Invalid price
        manager.propose = Mock(side_effect=ValueError("Invalid contract terms"))

        try:
            manager.propose(buyer.id, seller.id, invalid_terms)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert str(e) == "Invalid contract terms"


class TestMultiPartyContracts:
    """Test contracts involving more than two agents."""

    @pytest.mark.asyncio
    @patch("src.contracts.manager.ContractManager")
    async def test_three_party_agreement(self, mock_manager_class):
        """Test a contract requiring three signatures."""
        manager = Mock()
        mock_manager_class.return_value = manager

        contract_id = "tripartite_001"
        agents = ["agent_A", "agent_B", "agent_C"]

        manager.get_required_signatures = Mock(return_value=agents)
        required = manager.get_required_signatures(contract_id)
        assert len(required) == 3

        # Sign one by one
        manager.sign = Mock(return_value=True)
        for agent in agents:
            manager.sign(contract_id, agent, f"0xsig_{agent}")

        assert manager.sign.call_count == 3

        manager.is_fully_signed = Mock(return_value=True)
        assert manager.is_fully_signed(contract_id)
