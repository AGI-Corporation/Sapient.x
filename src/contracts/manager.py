"""ContractManager — Web4AGI

Manages the lifecycle of smart contracts between parcel agents:
proposal, negotiation, signing, execution, and dispute resolution.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


class ContractManager:
    """Manages smart contract creation, signing, and execution."""

    def __init__(self):
        self.id = str(uuid.uuid4())
        self.status = "pending_signature"
        self._contracts: Dict[str, Dict] = {}

    # ── Creation ──────────────────────────────────────────────────────────

    def propose(
        self,
        proposer_id: str,
        counterparty_id: str,
        terms: Dict[str, Any],
    ) -> str:
        """Propose a new contract. Returns the contract_id."""
        if not terms or terms.get("price") is not None and terms.get("price", 0) < 0:
            raise ValueError("Invalid contract terms")
        contract_id = f"contract-{uuid.uuid4().hex[:8]}"
        self._contracts[contract_id] = {
            "id": contract_id,
            "parties": [proposer_id, counterparty_id],
            "terms": terms,
            "status": "pending_signature",
            "signatures": [],
            "created_at": datetime.utcnow().isoformat(),
        }
        return contract_id

    # ── Status ────────────────────────────────────────────────────────────

    def get_status(self, contract_id: str) -> str:
        """Return the current status of a contract."""
        record = self._contracts.get(contract_id)
        if record is None:
            raise KeyError(f"Contract '{contract_id}' not found")
        return record["status"]

    # ── Signing ───────────────────────────────────────────────────────────

    def sign(self, contract_id: str, signer_id: str, signature: str) -> bool:
        """Add a signature from *signer_id* to the contract."""
        record = self._contracts.get(contract_id)
        if record is None:
            raise KeyError(f"Contract '{contract_id}' not found")
        record["signatures"].append({
            "signer": signer_id,
            "signature": signature,
            "signed_at": datetime.utcnow().isoformat(),
        })
        if self.is_fully_signed(contract_id):
            record["status"] = "signed"
        return True

    def reject(self, contract_id: str, rejector_id: str, reason: str = "") -> bool:
        """Reject the contract on behalf of *rejector_id*."""
        record = self._contracts.get(contract_id)
        if record is None:
            raise KeyError(f"Contract '{contract_id}' not found")
        record["status"] = "rejected"
        record["rejection_reason"] = reason
        return True

    # ── Execution ─────────────────────────────────────────────────────────

    async def execute(self, contract_id: str) -> Dict:
        """Execute a fully signed contract."""
        record = self._contracts.get(contract_id)
        if record is None:
            raise KeyError(f"Contract '{contract_id}' not found")
        tx_hash = f"0x{uuid.uuid4().hex}"
        record["status"] = "executed"
        record["tx_hash"] = tx_hash
        return {"status": "executed", "tx_hash": tx_hash}

    # ── Query helpers ─────────────────────────────────────────────────────

    def get_required_signatures(self, contract_id: str) -> List[str]:
        """Return list of party IDs required to sign the contract."""
        record = self._contracts.get(contract_id)
        if record is None:
            return []
        return list(record.get("parties", []))

    def is_fully_signed(self, contract_id: str) -> bool:
        """Return True if all required parties have signed."""
        record = self._contracts.get(contract_id)
        if record is None:
            return False
        required = set(record.get("parties", []))
        signed = {s["signer"] for s in record.get("signatures", [])}
        return required.issubset(signed)

    def get_contract(self, contract_id: str) -> Optional[Dict]:
        """Return the full contract record or None if not found."""
        return self._contracts.get(contract_id)
