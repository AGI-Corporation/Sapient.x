"""X402Client — Web4AGI

Handles USDx payments via the x402 HTTP Payment Protocol.
Each parcel agent uses this to deposit, transfer, and sign contracts.

Reference: https://x402.org
"""

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # graceful degradation for environments without httpx


X402_GATEWAY = "https://x402.org/api/v1"
USDX_DECIMALS = 6


@dataclass
class TransactionResult:
    """Result of an x402 payment transaction."""

    success: bool
    tx_hash: str = ""
    amount_micro: int = 0
    error: str = ""
    simulated: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "tx_hash": self.tx_hash,
            "amount_micro": self.amount_micro,
            "error": self.error,
            "simulated": self.simulated,
        }


def _to_micro(amount: float) -> int:
    """Convert human-readable USDx to micro-units (6 decimal places)."""
    return int(round(amount * 10**USDX_DECIMALS))


class X402Client:
    """
    Client for the x402 payment protocol.
    Supports deposit, transfer, contract signing, and balance queries.
    """

    def __init__(
        self,
        private_key: str | None = None,
        gateway_url: str = X402_GATEWAY,
        timeout: int = 30,
    ):
        self.private_key = private_key or ""
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout = timeout
        self._nonce = int(time.time() * 1000)

    # ── Internal Helpers ───────────────────────────────────────────────────

    def _next_nonce(self) -> int:
        self._nonce += 1
        return self._nonce

    def _sign(self, payload: dict) -> str:
        """HMAC-SHA256 sign a payload with the agent's private key."""
        # Exclude 'signature' key if already present to avoid circular signing
        signable = {k: v for k, v in payload.items() if k != "signature"}
        message = json.dumps(signable, sort_keys=True, separators=(",", ":"))
        sig = hmac.new(
            key=self.private_key.encode("utf-8"),
            msg=message.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()
        return sig

    async def _post(self, endpoint: str, body: dict) -> dict:
        """POST to the x402 gateway. Returns parsed JSON response."""
        if httpx is None:
            # Simulation mode when httpx is unavailable
            return {
                "success": True,
                "simulated": True,
                "endpoint": endpoint,
                "body": body,
            }
        url = f"{self.gateway_url}/{endpoint}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            return resp.json()

    async def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """GET from the x402 gateway. Returns parsed JSON response."""
        if httpx is None:
            return {"success": True, "simulated": True}
        url = f"{self.gateway_url}/{endpoint}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url, params=params or {})
            resp.raise_for_status()
            return resp.json()

    # ── Public API ────────────────────────────────────────────────────────────

    async def balance(self, address: str) -> dict:
        """Query the USDx balance of a wallet address."""
        return await self._get("balance", {"address": address})

    async def deposit(
        self,
        amount: float,
        source: str = "stablecoin_bridge",
    ) -> dict:
        """Deposit USDx into the agent's wallet."""
        nonce = self._next_nonce()
        payload: dict[str, Any] = {
            "action": "deposit",
            "amount_micro": _to_micro(amount),
            "source": source,
            "nonce": nonce,
        }
        payload["signature"] = self._sign(payload)
        return await self._post("deposit", payload)

    async def transfer(
        self,
        to_address: str,
        amount: float,
        memo: str = "",
        contract_terms: dict | None = None,
    ) -> dict:
        """Transfer USDx to another address via x402 payment header."""
        nonce = self._next_nonce()
        payload: dict[str, Any] = {
            "action": "transfer",
            "to": to_address,
            "amount_micro": _to_micro(amount),
            "memo": memo,
            "nonce": nonce,
        }
        if contract_terms:
            payload["contract_terms"] = contract_terms
        payload["signature"] = self._sign(payload)
        return await self._post("transfer", payload)

    async def sign_contract(
        self,
        contract: dict[str, Any],
        counterparty: str,
        signer: str,
    ) -> dict:
        """Sign a smart contract payload and submit to x402."""
        nonce = self._next_nonce()
        payload: dict[str, Any] = {
            "action": "sign_contract",
            "contract": contract,
            "counterparty": counterparty,
            "signer": signer,
            "nonce": nonce,
        }
        payload["signature"] = self._sign(payload)
        return await self._post("contracts/sign", payload)

    async def get_contract(self, contract_id: str) -> dict:
        """Fetch a contract by ID from the x402 registry."""
        return await self._get(f"contracts/{contract_id}")

    async def stream_payments(
        self,
        to_address: str,
        rate_usdx_per_second: float,
        duration_seconds: int,
    ) -> dict:
        """Open a USDx payment stream (e.g. for real-time parcel rent)."""
        nonce = self._next_nonce()
        payload: dict[str, Any] = {
            "action": "stream",
            "to": to_address,
            "rate_micro_per_second": _to_micro(rate_usdx_per_second),
            "duration_seconds": duration_seconds,
            "nonce": nonce,
        }
        payload["signature"] = self._sign(payload)
        return await self._post("streams/open", payload)


# ── Convenience factory ─────────────────────────────────────────────────────────


def make_x402_client(env: dict[str, str] | None = None) -> "X402Client":
    """Create an X402Client from environment variables."""
    import os

    cfg: Any = env if env is not None else os.environ
    return X402Client(
        private_key=cfg.get("X402_PRIVATE_KEY", ""),
        gateway_url=cfg.get("X402_GATEWAY", X402_GATEWAY),
    )
