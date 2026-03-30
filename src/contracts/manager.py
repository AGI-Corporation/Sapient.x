"""ContractManager — Web4AGI

Manages the lifecycle of smart contracts between parcel agents:
- Proposal, negotiation, digital signature, and execution.
"""

import uuid
from datetime import datetime
from typing import Any


class ContractManager:
    """In-memory smart contract registry for Web4AGI agents."""

    def __init__(self) -> None:
        self._contracts: dict[str, dict[str, Any]] = {}

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def propose(
        self,
        party_a: str,
        party_b: str,
        contract_data: dict[str, Any],
    ) -> str:
        """Create a new contract proposal. Returns the contract ID."""
        terms = contract_data.get("terms", contract_data)
        price = terms.get("price", 0)
        if isinstance(price, (int, float)) and price < 0:
            raise ValueError("Invalid contract terms: price cannot be negative")

        contract_id = str(uuid.uuid4())
        self._contracts[contract_id] = {
            "contract_id": contract_id,
            "party_a": party_a,
            "party_b": party_b,
            "terms": terms,
            "status": "pending_signature",
            "signatures": {},
            "created_at": datetime.utcnow().isoformat(),
            "executed_at": None,
            "tx_hash": None,
        }
        return contract_id

    def get(self, contract_id: str) -> dict[str, Any] | None:
        """Retrieve a contract by ID. Returns None if not found."""
        return self._contracts.get(contract_id)

    def get_status(self, contract_id: str) -> str | None:
        """Return the status string for a contract, or None if not found."""
        contract = self._contracts.get(contract_id)
        if contract is None:
            return None
        return contract["status"]

    def sign(self, contract_id: str, signer_id: str, signature: str) -> bool:
        """Record a signature from a party. Returns True on success."""
        contract = self._contracts.get(contract_id)
        if contract is None:
            return False
        if contract["status"] in ("executed", "rejected"):
            return False
        contract["signatures"][signer_id] = {
            "signature": signature,
            "signed_at": datetime.utcnow().isoformat(),
        }
        # Auto-advance status when both parties have signed
        if (
            contract["party_a"] in contract["signatures"]
            and contract["party_b"] in contract["signatures"]
        ):
            contract["status"] = "fully_signed"
        return True

    def reject(self, contract_id: str, rejector_id: str, reason: str = "") -> bool:
        """Reject a contract on behalf of a party. Returns True on success."""
        contract = self._contracts.get(contract_id)
        if contract is None:
            return False
        contract["status"] = "rejected"
        contract["rejection"] = {"by": rejector_id, "reason": reason}
        return True

    async def execute(self, contract_id: str) -> dict[str, Any]:
        """Execute a fully-signed contract. Returns the execution result."""
        import hashlib
        import time

        contract = self._contracts.get(contract_id)
        if contract is None:
            return {"success": False, "error": "Contract not found"}
        if contract["status"] not in ("fully_signed", "pending_signature"):
            return {"success": False, "error": f"Cannot execute contract in status '{contract['status']}'"}

        tx_hash = f"0x{hashlib.sha256(f'{contract_id}{time.time()}'.encode()).hexdigest()}"
        contract["status"] = "executed"
        contract["executed_at"] = datetime.utcnow().isoformat()
        contract["tx_hash"] = tx_hash
        return {"status": "executed", "tx_hash": tx_hash, "contract_id": contract_id}

    def list_contracts(
        self,
        party: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List contracts, optionally filtered by party or status."""
        results = list(self._contracts.values())
        if party:
            results = [c for c in results if c["party_a"] == party or c["party_b"] == party]
        if status:
            results = [c for c in results if c["status"] == status]
        return results
