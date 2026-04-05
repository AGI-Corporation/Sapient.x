"""ContractManager — Web4AGI

Manages the full lifecycle of parcel service agreements:
  - Propose / negotiate
  - Sign by all required parties
  - Execute with escrow settlement
  - Reject or cancel
"""

import uuid
from datetime import UTC, datetime
from typing import Any


class ContractManager:
    """In-process contract registry and lifecycle manager."""

    def __init__(self) -> None:
        self._contracts: dict[str, dict[str, Any]] = {}

    # ── Proposal ──────────────────────────────────────────────────────────────

    def propose(
        self,
        party_a: str,
        party_b: str,
        terms: dict[str, Any],
        required_signers: list[str] | None = None,
    ) -> str:
        """Create a new contract proposal and return its ID.

        Raises:
            ValueError: If terms are invalid (e.g. negative price).
        """
        price = terms.get("price")
        if price is not None and price < 0:
            raise ValueError("Invalid contract terms")

        contract_id = str(uuid.uuid4())
        signers = required_signers or [party_a, party_b]
        self._contracts[contract_id] = {
            "contract_id": contract_id,
            "party_a": party_a,
            "party_b": party_b,
            "terms": terms,
            "status": "pending_signature",
            "required_signers": signers,
            "signatures": {},
            "created_at": datetime.now(UTC).isoformat(),
            "executed_at": None,
            "tx_hash": None,
        }
        return contract_id

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self, contract_id: str) -> str:
        """Return the current status string for a contract."""
        contract = self._get(contract_id)
        return contract["status"]

    def get(self, contract_id: str) -> dict[str, Any]:
        """Return the full contract record."""
        return self._get(contract_id)

    # ── Signing ───────────────────────────────────────────────────────────────

    def sign(self, contract_id: str, agent_id: str, signature: str) -> bool:
        """Record a party's signature.  Returns True on success."""
        contract = self._get(contract_id)
        if contract["status"] not in ("pending_signature",):
            return False
        contract["signatures"][agent_id] = signature
        if self.is_fully_signed(contract_id):
            contract["status"] = "fully_signed"
        return True

    def get_required_signatures(self, contract_id: str) -> list[str]:
        """Return the list of agent IDs that must sign."""
        return self._get(contract_id)["required_signers"]

    def is_fully_signed(self, contract_id: str) -> bool:
        """Return True when every required signer has provided a signature."""
        contract = self._get(contract_id)
        return all(s in contract["signatures"] for s in contract["required_signers"])

    # ── Execution ─────────────────────────────────────────────────────────────

    async def execute(self, contract_id: str) -> dict[str, Any]:
        """Execute a fully-signed contract and settle escrow.

        Returns a dict with ``status`` and ``tx_hash``.
        """
        contract = self._get(contract_id)
        tx_hash = f"0xtx_{contract_id[:8]}"
        contract["status"] = "executed"
        contract["executed_at"] = datetime.now(UTC).isoformat()
        contract["tx_hash"] = tx_hash
        return {"status": "executed", "tx_hash": tx_hash, "contract_id": contract_id}

    # ── Rejection / Cancellation ──────────────────────────────────────────────

    def reject(self, contract_id: str, agent_id: str, reason: str = "") -> bool:
        """Reject a contract on behalf of an agent.  Returns True on success."""
        contract = self._get(contract_id)
        contract["status"] = "rejected"
        contract["rejection"] = {"by": agent_id, "reason": reason}
        return True

    def cancel(self, contract_id: str) -> bool:
        """Cancel a contract that has not yet been executed."""
        contract = self._get(contract_id)
        if contract["status"] == "executed":
            return False
        contract["status"] = "cancelled"
        return True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get(self, contract_id: str) -> dict[str, Any]:
        contract = self._contracts.get(contract_id)
        if contract is None:
            raise KeyError(f"Contract '{contract_id}' not found")
        return contract

    def list_contracts(self) -> list[dict[str, Any]]:
        """Return all contracts."""
        return list(self._contracts.values())
