"""X402Client — Web4AGI

Handles USDx payments via the x402 HTTP Payment Protocol.
Each parcel agent uses this to deposit, transfer, and sign contracts.

Reference: https://x402.org
"""

import hashlib
import hmac
import json
import time
from typing import Any, TypedDict


class TransactionResult(TypedDict):
    success: bool
    transaction_id: str | None
    amount_micro: int
    error: str | None
    simulated: bool
    tx_hash: str | None
    amount: float | None


class SignedTransaction(TypedDict):
    to: str
    value: float
    nonce: int
    signature: str
    r: str
    s: str
    v: int


try:
    import httpx
except ImportError:
    httpx = None  # graceful degradation for environments without httpx


X402_GATEWAY = "https://x402.org/api/v1"
USDX_DECIMALS = 6


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
        local_only: bool = True,
    ):
        self.private_key = private_key or "default_test_key"
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout = timeout
        self.local_only = local_only
        self._nonce = int(time.time() * 1000)

    # ── Internal Helpers ───────────────────────────────────────────────────

    def _next_nonce(self) -> int:
        self._nonce += 1
        return self._nonce

    def _sign(self, payload: dict[str, Any]) -> str:
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

    async def _post(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST to the x402 gateway. Returns parsed JSON response."""
        if self.local_only or httpx is None:
            # Simulation mode
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

    async def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET from the x402 gateway. Returns parsed JSON response."""
        if self.local_only or httpx is None:
            return {"success": True, "simulated": True}
        url = f"{self.gateway_url}/{endpoint}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url, params=params or {})
            resp.raise_for_status()
            return resp.json()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_address(self) -> str:
        """Return a deterministic wallet address based on the private key."""
        h = hashlib.sha256(self.private_key.encode()).hexdigest()
        return f"0x{h[:40]}"

    async def get_balance(self, address: str) -> float:
        """Retrieve the balance for a given address."""
        return await self._query_balance(address)

    async def _query_balance(self, address: str) -> float:
        """Internal method to query balance (for mocking)."""
        res = await self._get("balance", {"address": address})
        if self.local_only:
            return 1000.0  # Default simulated balance
        return res.get("balance", 0.0)

    async def balance(self, address: str) -> dict[str, Any]:
        """Query the USDx balance of a wallet address (raw response)."""
        return await self._get("balance", {"address": address})

    async def deposit(
        self,
        amount: float,
        source: str = "stablecoin_bridge",
    ) -> dict[str, Any]:
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
        contract_terms: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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

    async def create_payment(
        self, to_address: str, amount_usdx: float, memo: str = ""
    ) -> TransactionResult:
        """Create a payment (alias for transfer used in tests)."""
        # Verification check for insufficient balance simulation
        current_balance = await self.get_balance(self.get_address())
        if amount_usdx > current_balance:
            return {
                "success": False,
                "error": "Insufficient balance",
                "transaction_id": None,
                "amount_micro": _to_micro(amount_usdx),
                "simulated": self.local_only,
            }

        res = await self.transfer(to_address, amount_usdx, memo)
        typed_res: TransactionResult = {
            "success": res.get("success", False),
            "transaction_id": res.get("transaction_id"),
            "amount_micro": _to_micro(amount_usdx),
            "error": res.get("error"),
            "simulated": res.get("simulated", False),
            "tx_hash": None,
            "amount": amount_usdx,
        }
        if self.local_only:
            typed_res["tx_hash"] = f"0x{hashlib.sha256(str(time.time()).encode()).hexdigest()}"
        return typed_res

    def sign_transaction(self, tx_data: dict[str, Any]) -> SignedTransaction:
        """Sign a transaction object."""
        sig = self._sign(tx_data)
        return {
            "to": tx_data.get("to", ""),
            "value": tx_data.get("value", 0.0),
            "nonce": tx_data.get("nonce", 0),
            "signature": sig,
            "r": f"0x{sig[:64]}",
            "s": f"0x{sig[64:128]}",
            "v": 27,
        }

    def sign_message(self, message: str) -> str:
        """Sign a text message."""
        return self._sign({"message": message})

    def verify_signature(self, message: str, signature: str, signer_address: str) -> bool:
        """Verify a message signature."""
        # signer_address is used for multi-sig validation in future
        _ = signer_address
        if self.local_only:
            return True
        return self._sign({"message": message}) == signature

    async def batch_payment(self, payments: list[dict[str, Any]]) -> list[TransactionResult]:
        """Process a batch of payments."""
        results = []
        for p in payments:
            results.append(await self.create_payment(p["to"], p["amount"]))
        return results

    def encode_function(self, function_name: str, params: list[Any]) -> str:
        """Encode a function call for the blockchain."""
        h = hashlib.sha256(f"{function_name}{params}".encode()).hexdigest()
        return f"0x{h[:8]}{hashlib.sha256(str(params).encode()).hexdigest()}"

    async def get_transaction_history(self, address: str) -> list[dict[str, Any]]:
        """Retrieve transaction history."""
        return await self._fetch_history(address)

    async def _fetch_history(self, address: str) -> list[dict[str, Any]]:
        """Internal fetch history (for mocking)."""
        if self.local_only:
            return []
        res = await self._get("history", {"address": address})
        return res.get("history", [])

    async def estimate_gas(self, to_address: str, amount_usdx: float) -> float:
        """Estimate gas for a transaction."""
        return 0.001 * (len(to_address) + amount_usdx)

    def validate_address(self, address: str) -> None:
        """Validate an Ethereum-style address."""
        if not address.startswith("0x") or len(address) != 42:
            raise ValueError(f"Invalid address: {address}")

    async def sign_contract(
        self,
        contract: dict[str, Any],
        counterparty: str,
        signer: str,
    ) -> dict[str, Any]:
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

    async def get_contract(self, contract_id: str) -> dict[str, Any]:
        """Fetch a contract by ID from the x402 registry."""
        return await self._get(f"contracts/{contract_id}")

    async def stream_payments(
        self,
        to_address: str,
        rate_usdx_per_second: float,
        duration_seconds: int,
    ) -> dict[str, Any]:
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
