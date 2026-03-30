"""Integration tests for the contract lifecycle flow.

Tests the ContractManager class directly:
- Contract proposal and negotiation
- Terms validation
- Digital signature process
- Contract execution and escrow handling
- Multi-party agreement scenarios
"""

import pytest

from src.contracts.manager import ContractManager


class TestContractLifecycle:
    """Test the complete contract lifecycle between agents."""

    @pytest.fixture
    def manager(self):
        return ContractManager()

    @pytest.mark.asyncio
    async def test_full_contract_flow(self, manager):
        """Test a successful contract flow from proposal to execution."""
        # 1. Propose Contract
        contract_id = manager.propose(
            party_a="agent_buyer",
            party_b="agent_seller",
            contract_data={"parcel_id": "parcel_001", "price": 5000.0},
        )
        assert isinstance(contract_id, str)

        # 2. Verify initial status
        assert manager.get_status(contract_id) == "pending_signature"

        # 3. Sign Contract (Buyer)
        assert manager.sign(contract_id, "agent_buyer", "0xsignature_buyer")

        # 4. Sign Contract (Seller) — triggers fully_signed
        assert manager.sign(contract_id, "agent_seller", "0xsignature_seller")
        assert manager.get_status(contract_id) == "fully_signed"

        # 5. Execute Contract
        result = await manager.execute(contract_id)
        assert result["status"] == "executed"
        assert "tx_hash" in result

    @pytest.mark.asyncio
    async def test_contract_rejection(self, manager):
        """Test contract rejection by counterparty."""
        contract_id = manager.propose(
            party_a="agent_buyer",
            party_b="agent_seller",
            contract_data={"price": 1000.0},
        )
        assert manager.reject(contract_id, "agent_seller", "Price too low")
        assert manager.get_status(contract_id) == "rejected"

    def test_contract_validation_failure(self, manager):
        """Test contract creation failure due to invalid terms."""
        with pytest.raises(ValueError, match="Invalid contract terms"):
            manager.propose(
                party_a="agent_buyer",
                party_b="agent_seller",
                contract_data={"price": -100},
            )

    def test_get_status_not_found(self, manager):
        """get_status returns None for unknown contracts."""
        assert manager.get_status("nonexistent") is None

    def test_sign_unknown_contract(self, manager):
        """sign returns False for unknown contracts."""
        assert manager.sign("nonexistent", "agent_a", "0xsig") is False

    def test_reject_unknown_contract(self, manager):
        """reject returns False for unknown contracts."""
        assert manager.reject("nonexistent", "agent_a") is False

    @pytest.mark.asyncio
    async def test_cannot_execute_rejected(self, manager):
        """Cannot execute a rejected contract."""
        contract_id = manager.propose(
            party_a="a", party_b="b", contract_data={"price": 10.0}
        )
        manager.reject(contract_id, "b", "No deal")
        result = await manager.execute(contract_id)
        assert result["success"] is False

    def test_list_contracts_filter_by_party(self, manager):
        """list_contracts returns only contracts for the requested party."""
        manager.propose("alice", "bob", {"price": 10.0})
        manager.propose("carol", "dave", {"price": 20.0})

        alice_contracts = manager.list_contracts(party="alice")
        assert len(alice_contracts) == 1
        assert alice_contracts[0]["party_a"] == "alice"

    def test_list_contracts_filter_by_status(self, manager):
        """list_contracts returns only contracts with the given status."""
        c1 = manager.propose("x", "y", {"price": 5.0})
        manager.propose("a", "b", {"price": 5.0})
        manager.reject(c1, "y", "bad deal")

        rejected = manager.list_contracts(status="rejected")
        assert all(c["status"] == "rejected" for c in rejected)


class TestMultiPartyContracts:
    """Test contracts involving more than two agents."""

    @pytest.mark.asyncio
    async def test_three_party_agreement_sequence(self):
        """Test a three-party agreement using two separate contracts."""
        manager = ContractManager()

        # Party A–B contract
        c1 = manager.propose("agent_A", "agent_B", {"price": 100.0})
        manager.sign(c1, "agent_A", "0xsig_A_ab")
        manager.sign(c1, "agent_B", "0xsig_B_ab")

        # Party B–C contract
        c2 = manager.propose("agent_B", "agent_C", {"price": 50.0})
        manager.sign(c2, "agent_B", "0xsig_B_bc")
        manager.sign(c2, "agent_C", "0xsig_C_bc")

        assert manager.get_status(c1) == "fully_signed"
        assert manager.get_status(c2) == "fully_signed"

        r1 = await manager.execute(c1)
        r2 = await manager.execute(c2)
        assert r1["status"] == "executed"
        assert r2["status"] == "executed"

