"""ContractManager — Web4AGI

Manages the lifecycle of smart contracts between parcel agents:
creation, negotiation, signing, and execution.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


class ContractManager:
    """Create, sign, and execute smart contracts between agents."""

    def __init__(self) -> None:
        self._contracts: Dict[str, Dict] = {}

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def propose(
        self,
        proposer_id: str,
        counterparty_id: str,
        terms: Dict[str, Any],
    ) -> str:
        """Create a new contract proposal and return its ID."""
        contract_id = f"contract-{uuid.uuid4().hex[:8]}"
        self._contracts[contract_id] = {
            "id": contract_id,
            "proposer": proposer_id,
            "counterparty": counterparty_id,
            "terms": terms,
            "status": "pending_signature",
            "signatures": {},
            "created_at": datetime.utcnow().isoformat(),
        }
        return contract_id

    def get(self, contract_id: str) -> Optional[Dict]:
        """Return the contract record or *None* if not found."""
        return self._contracts.get(contract_id)

    def get_status(self, contract_id: str) -> Optional[str]:
        """Return the current status string for a contract."""
        contract = self._contracts.get(contract_id)
        return contract["status"] if contract else None

    def get_required_signatures(self, contract_id: str) -> List[str]:
        """Return the list of agent IDs that must sign the contract."""
        contract = self._contracts.get(contract_id)
        if not contract:
            return []
        return [contract["proposer"], contract["counterparty"]]

    # ── Signing ───────────────────────────────────────────────────────────────

    def sign(self, contract_id: str, signer_id: str, signature: str) -> bool:
        """Record a signature for *signer_id*. Returns True on success."""
        contract = self._contracts.get(contract_id)
        if not contract:
            return False
        contract["signatures"][signer_id] = signature
        required = self.get_required_signatures(contract_id)
        if all(r in contract["signatures"] for r in required):
            contract["status"] = "fully_signed"
        return True

    def is_fully_signed(self, contract_id: str) -> bool:
        """Return True when all required parties have signed."""
        contract = self._contracts.get(contract_id)
        if not contract:
            return False
        required = self.get_required_signatures(contract_id)
        return all(r in contract["signatures"] for r in required)

    def reject(self, contract_id: str, rejector_id: str, reason: str = "") -> bool:
        """Reject a contract. Returns True on success."""
        contract = self._contracts.get(contract_id)
        if not contract:
            return False
        contract["status"] = "rejected"
        contract["rejection_reason"] = reason
        contract["rejected_by"] = rejector_id
        return True

    # ── Execution ─────────────────────────────────────────────────────────────

    async def execute(self, contract_id: str) -> Dict:
        """Execute a fully-signed contract (simulated)."""
        contract = self._contracts.get(contract_id)
        if not contract:
            return {"success": False, "error": "Contract not found"}
        if not self.is_fully_signed(contract_id):
            return {"success": False, "error": "Contract not fully signed"}
        contract["status"] = "executed"
        tx_hash = "0x" + contract_id.replace("-", "")
        return {"status": "executed", "tx_hash": tx_hash, "contract_id": contract_id}
