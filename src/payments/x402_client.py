"""X402Client — Web4AGI

Handles USDx payments via the x402 HTTP Payment Protocol.
Each parcel agent uses this to deposit, transfer, and sign contracts.

Reference: https://x402.org
"""

import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Any, TypedDict


class TransactionResult(TypedDict):
    """Result of an x402 transaction."""

    success: bool
    tx_hash: str | None
    amount: float | None
    error: str | None
    action: str | None


try:
    import httpx
except ImportError:
    httpx = None


X402_GATEWAY = os.getenv("X402_GATEWAY", "https://x402.org/api/v1")
USDX_DECIMALS = 6


def _to_micro(amount: float) -> int:
    """Convert human-readable USDx to micro-units (6 decimal places)."""
    return int(round(amount * 10**USDX_DECIMALS))


def _from_micro(micro_amount: int) -> float:
    """Convert micro-units back to human-readable USDx."""
    return float(micro_amount) / 10**USDX_DECIMALS


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
        """HMAC-SHA256 sign a payload with the agent's private key for X402 compliance."""
        if not self.private_key:
            return "unsigned-simulation"

        # Standard X402 signing: sort keys, remove whitespace, hash with private key
        signable = {k: v for k, v in payload.items() if k != "signature"}
        message = json.dumps(signable, sort_keys=True, separators=(",", ":"))

        return hmac.new(
            key=self.private_key.encode("utf-8"),
            msg=message.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()

    async def _post(self, endpoint: str, body: dict) -> dict:
        """POST to the x402 gateway. Returns parsed JSON response."""
        if httpx is None:
            return {
                "success": True,
                "simulated": True,
                "endpoint": endpoint,
                "tx_hash": f"sim-tx-{uuid.uuid4()}",
            }

        url = f"{self.gateway_url}/{endpoint}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, json=body)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                return {"success": False, "error": f"X402 Gateway Error: {e}"}

    async def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """GET from the x402 gateway. Returns parsed JSON response."""
        if httpx is None:
            return {"success": True, "simulated": True, "balance_usdx": 1000.0}

        url = f"{self.gateway_url}/{endpoint}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(url, params=params or {})
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                return {"success": False, "error": f"X402 Gateway Error: {e}"}

    # ── Public API ────────────────────────────────────────────────────────────

    def get_address(self) -> str:
        """Generate a deterministic wallet address from the private key."""
        if not self.private_key:
            return "0x0000000000000000000000000000000000000000"
        h = hashlib.sha256(self.private_key.encode()).hexdigest()
        return f"0x{h[:40]}"

    def validate_address(self, address: str) -> bool:
        """Check if an address is a valid 0x-prefixed hex string."""
        if not isinstance(address, str) or not address.startswith("0x") or len(address) < 10:
            raise ValueError(f"Invalid address format: {address}")
        return True

    async def get_balance(self, address: str) -> float:
        """Query the USDx balance of a wallet address."""
        return await self._query_balance(address)

    async def _query_balance(self, address: str) -> float:
        """Internal method for querying balance, easily mocked in tests."""
        data = await self._get("balance", {"address": address})
        if data.get("simulated"):
            return 1000.0
        return float(data.get("balance_usdx", 0.0))

    async def deposit(
        self,
        amount: float,
        source: str = "stablecoin_bridge",
    ) -> TransactionResult:
        """Deposit USDx into the agent's wallet."""
        nonce = self._next_nonce()
        payload: dict[str, Any] = {
            "action": "deposit",
            "amount_micro": _to_micro(amount),
            "source": source,
            "nonce": nonce,
        }
        payload["signature"] = self._sign(payload)
        res = await self._post("deposit", payload)

        return {
            "success": res.get("success", False),
            "tx_hash": res.get("tx_hash"),
            "amount": amount,
            "error": res.get("error"),
            "action": "deposit",
        }

    async def create_payment(
        self, to_address: str, amount_usdx: float, memo: str = ""
    ) -> TransactionResult:
        """Create and execute a payment transaction (transfer)."""
        # 1. Validation
        self.validate_address(to_address)

        # 2. Balance Check
        balance = await self.get_balance(self.get_address())
        if balance < amount_usdx:
            return {
                "success": False,
                "tx_hash": None,
                "amount": amount_usdx,
                "error": "Insufficient USDx balance",
                "action": "transfer",
            }

        # 3. Execute Transfer
        return await self.transfer(to_address, amount_usdx, memo)

    async def transfer(
        self,
        to_address: str,
        amount: float,
        memo: str = "",
        contract_terms: dict | None = None,
    ) -> TransactionResult:
        """Internal transfer implementation following x402 spec."""
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
        res = await self._post("transfer", payload)

        return {
            "success": res.get("success", False),
            "tx_hash": res.get("tx_hash")
            or (
                f"0x{hashlib.sha256(str(time.time()).encode()).hexdigest()}"
                if res.get("success")
                else None
            ),
            "amount": amount,
            "error": res.get("error"),
            "action": "transfer",
        }

    async def sign_transaction(self, tx_data: dict[str, Any]) -> dict[str, Any]:
        """Sign a raw transaction dictionary for x402 submission."""
        signature = self._sign(tx_data)
        return {
            **tx_data,
            "signature": signature,
            "r": f"0x{signature[:32]}",  # Simplified r/s for mock
            "s": f"0x{signature[32:64]}",
            "v": 27,
        }

    def sign_message(self, message: str) -> str:
        """Sign a text message using the agent's private key."""
        return self._sign({"message": message})

    def verify_signature(self, message: str, signature: str, signer_address: str) -> bool:
        """Verify an x402 signature against a signer address."""
        # In this implementation, we re-derive the signature to verify
        expected = self.sign_message(message)
        return signature == expected and signer_address == self.get_address()

    async def sign_contract(
        self,
        contract: dict[str, Any],
        counterparty: str,
        signer: str,
    ) -> TransactionResult:
        """Sign a smart contract payload and submit to x402 gateway."""
        nonce = self._next_nonce()
        payload: dict[str, Any] = {
            "action": "sign_contract",
            "contract": contract,
            "counterparty": counterparty,
            "signer": signer,
            "nonce": nonce,
        }
        payload["signature"] = self._sign(payload)
        res = await self._post("contracts/sign", payload)

        return {
            "success": res.get("success", False),
            "tx_hash": res.get("tx_hash"),
            "amount": None,
            "error": res.get("error"),
            "action": "sign_contract",
        }

    async def estimate_gas(self, to_address: str, amount_usdx: float) -> float:  # noqa: ARG002
        """Estimate the gas cost for a USDx transaction."""
        return 0.00015  # Typical stablecoin transfer cost on-chain

    async def get_transaction_history(self, address: str) -> list[dict[str, Any]]:
        """Fetch historical x402 transactions for an address."""
        return await self._fetch_history(address)

    async def _fetch_history(self, address: str) -> list[dict[str, Any]]:
        """Internal method for fetching history, easily mocked in tests."""
        data = await self._get(f"history/{address}")
        if data.get("simulated"):
            return []
        return data.get("transactions", [])

    def encode_function(self, function_name: str, params: list[Any]) -> str:
        """Encode a contract function call (simplified ABI encoding)."""
        param_str = ",".join(map(str, params))
        return f"0x{hashlib.sha256(f'{function_name}({param_str})'.encode()).hexdigest()[:8]}"

    async def batch_payment(self, payments: list[dict[str, Any]]) -> list[TransactionResult]:
        """Process multiple payments in sequence."""
        results = []
        for p in payments:
            res = await self.create_payment(p["to"], p["amount"], p.get("memo", ""))
            results.append(res)
        return results

    async def balance(self, address: str) -> dict:
        """Legacy compatibility method for querying balance dict."""
        bal = await self.get_balance(address)
        return {"address": address, "balance_usdx": bal, "success": True}


# ── Convenience factory ─────────────────────────────────────────────────────────


def make_x402_client(env: dict[str, str] | None = None) -> "X402Client":
    """Create an X402Client from environment variables."""
    import os

    cfg: Any = env if env is not None else os.environ
    return X402Client(
        private_key=cfg.get("X402_PRIVATE_KEY", ""),
        gateway_url=cfg.get("X402_GATEWAY", X402_GATEWAY),
    )
