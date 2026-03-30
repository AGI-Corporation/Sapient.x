"""ContractManager — Web4AGI

Manages the full lifecycle of smart contracts between parcel agents:
  - Proposal & negotiation
  - Multi-party digital signatures (via x402)
  - Escrow-style execution
  - State persistence (in-memory; plug in a DB adapter for production)
"""

import uuid
from datetime import UTC, datetime
from typing import Any


class ContractError(Exception):
    """Raised when a contract operation is invalid."""


class ContractManager:
    """Manage creation, signing, and execution of inter-agent contracts."""

    def __init__(self) -> None:
        self._contracts: dict[str, dict[str, Any]] = {}

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    def _get_or_raise(self, contract_id: str) -> dict[str, Any]:
        contract = self._contracts.get(contract_id)
        if contract is None:
            raise KeyError(f"Contract '{contract_id}' not found")
        return contract

    # ── Proposal ───────────────────────────────────────────────────────────────

    def propose(
        self,
        proposer_id: str,
        counterparty_id: str,
        terms: dict[str, Any],
    ) -> str:
        """Create a new contract proposal and return its contract_id."""
        if not terms:
            raise ContractError("Contract terms must not be empty")
        price = terms.get("price", 0)
        if isinstance(price, (int, float)) and price < 0:
            raise ContractError("Invalid contract terms")

        contract_id = f"contract-{uuid.uuid4().hex[:12]}"
        self._contracts[contract_id] = {
            "contract_id": contract_id,
            "proposer": proposer_id,
            "counterparty": counterparty_id,
            "terms": terms,
            "status": "pending_signature",
            "signatures": {},
            "required_signers": [proposer_id, counterparty_id],
            "created_at": self._now(),
            "updated_at": self._now(),
            "executed_at": None,
            "tx_hash": None,
            "rejection_reason": None,
        }
        return contract_id

    # ── Status ─────────────────────────────────────────────────────────────────

    def get_status(self, contract_id: str) -> str:
        return self._get_or_raise(contract_id)["status"]

    def get(self, contract_id: str) -> dict[str, Any]:
        return dict(self._get_or_raise(contract_id))

    # ── Signatures ─────────────────────────────────────────────────────────────

    def sign(self, contract_id: str, signer_id: str, signature: str) -> bool:
        """Record a digital signature from *signer_id*."""
        contract = self._get_or_raise(contract_id)
        if contract["status"] in ("executed", "rejected"):
            raise ContractError(f"Cannot sign a contract with status '{contract['status']}'")
        contract["signatures"][signer_id] = {
            "signature": signature,
            "signed_at": self._now(),
        }
        contract["updated_at"] = self._now()
        # Automatically transition to fully_signed when all required signers have signed
        if all(s in contract["signatures"] for s in contract["required_signers"]):
            contract["status"] = "fully_signed"
        return True

    def is_fully_signed(self, contract_id: str) -> bool:
        contract = self._get_or_raise(contract_id)
        return all(s in contract["signatures"] for s in contract["required_signers"])

    def get_required_signatures(self, contract_id: str) -> list[str]:
        return list(self._get_or_raise(contract_id)["required_signers"])

    def add_required_signer(self, contract_id: str, signer_id: str) -> None:
        contract = self._get_or_raise(contract_id)
        if signer_id not in contract["required_signers"]:
            contract["required_signers"].append(signer_id)

    # ── Rejection ──────────────────────────────────────────────────────────────

    def reject(self, contract_id: str, rejector_id: str, reason: str = "") -> bool:
        """Reject a contract proposal."""
        contract = self._get_or_raise(contract_id)
        contract["status"] = "rejected"
        contract["rejection_reason"] = reason
        contract["updated_at"] = self._now()
        return True

    # ── Execution ─────────────────────────────────────────────────────────────

    async def execute(self, contract_id: str) -> dict[str, Any]:
        """Execute a fully-signed contract (move escrow, update state)."""
        contract = self._get_or_raise(contract_id)
        if not self.is_fully_signed(contract_id):
            return {"status": "error", "error": "Not all required parties have signed"}
        tx_hash = f"0x{uuid.uuid4().hex}"
        contract["status"] = "executed"
        contract["executed_at"] = self._now()
        contract["tx_hash"] = tx_hash
        contract["updated_at"] = self._now()
        return {
            "status": "executed",
            "contract_id": contract_id,
            "tx_hash": tx_hash,
            "executed_at": contract["executed_at"],
        }

    # ── Listing ───────────────────────────────────────────────────────────────

    def list_contracts(
        self,
        party_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        contracts = list(self._contracts.values())
        if party_id:
            contracts = [
                c
                for c in contracts
                if c["proposer"] == party_id or c["counterparty"] == party_id
            ]
        if status:
            contracts = [c for c in contracts if c["status"] == status]
        return contracts
