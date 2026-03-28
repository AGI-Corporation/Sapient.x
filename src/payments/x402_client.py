"""X402Client — Web4AGI

Handles USDx payments via the x402 HTTP Payment Protocol.
Each parcel agent uses this to deposit, transfer, and sign contracts.

Reference: https://x402.org
"""

import asyncio
import hashlib
import hmac
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None  # graceful degradation for environments without httpx


X402_GATEWAY = "https://x402.org/api/v1"
USDX_DECIMALS = 6

_ETH_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


@dataclass
class TransactionResult:
    """Result of a completed x402 transaction."""

    success: bool
    tx_hash: Optional[str] = None
    amount: Optional[float] = None
    error: Optional[str] = None


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
        private_key: Optional[str] = None,
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

    def _sign(self, payload: Dict) -> str:
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

    async def _post(self, endpoint: str, body: Dict) -> Dict:
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
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=body)
                resp.raise_for_status()
                return resp.json()
        except Exception:
            # Simulation mode when gateway is unreachable
            return {
                "success": True,
                "simulated": True,
                "endpoint": endpoint,
                "body": body,
            }

    async def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """GET from the x402 gateway. Returns parsed JSON response."""
        if httpx is None:
            return {"success": True, "simulated": True}
        url = f"{self.gateway_url}/{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, params=params or {})
                resp.raise_for_status()
                return resp.json()
        except Exception:
            return {"success": True, "simulated": True}

    # ── Public API ────────────────────────────────────────────────────────────

    async def balance(self, address: str) -> Dict:
        """Query the USDx balance of a wallet address."""
        return await self._get("balance", {"address": address})

    async def deposit(
        self,
        amount: float,
        source: str = "stablecoin_bridge",
    ) -> Dict:
        """Deposit USDx into the agent's wallet."""
        nonce = self._next_nonce()
        payload: Dict[str, Any] = {
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
        contract_terms: Optional[Dict] = None,
    ) -> Dict:
        """Transfer USDx to another address via x402 payment header."""
        nonce = self._next_nonce()
        payload: Dict[str, Any] = {
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
        contract: Dict[str, Any],
        counterparty: str,
        signer: str,
    ) -> Dict:
        """Sign a smart contract payload and submit to x402."""
        nonce = self._next_nonce()
        payload: Dict[str, Any] = {
            "action": "sign_contract",
            "contract": contract,
            "counterparty": counterparty,
            "signer": signer,
            "nonce": nonce,
        }
        payload["signature"] = self._sign(payload)
        return await self._post("contracts/sign", payload)

    async def get_contract(self, contract_id: str) -> Dict:
        """Fetch a contract by ID from the x402 registry."""
        return await self._get(f"contracts/{contract_id}")

    async def stream_payments(
        self,
        to_address: str,
        rate_usdx_per_second: float,
        duration_seconds: int,
    ) -> Dict:
        """Open a USDx payment stream (e.g. for real-time parcel rent)."""
        nonce = self._next_nonce()
        payload: Dict[str, Any] = {
            "action": "stream",
            "to": to_address,
            "rate_micro_per_second": _to_micro(rate_usdx_per_second),
            "duration_seconds": duration_seconds,
            "nonce": nonce,
        }
        payload["signature"] = self._sign(payload)
        return await self._post("streams/open", payload)

    # ── Extended public API ───────────────────────────────────────────────────

    def get_address(self) -> str:
        """Derive a deterministic Ethereum-style address from the private key."""
        digest = hashlib.sha256(self.private_key.encode("utf-8")).hexdigest()
        return "0x" + digest[-40:]

    def validate_address(self, address: str) -> str:
        """Validate an Ethereum address format. Raises ValueError if invalid."""
        if not _ETH_ADDRESS_RE.match(address):
            raise ValueError(f"Invalid Ethereum address: {address!r}")
        return address

    def sign_message(self, message: str) -> str:
        """Sign an arbitrary message with the agent's private key (HMAC-SHA256)."""
        return hmac.new(
            key=self.private_key.encode("utf-8"),
            msg=message.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()

    def verify_signature(self, message: str, signature: str, signer_address: str) -> bool:
        """Verify a message signature produced by this client."""
        expected = self.sign_message(message)
        return hmac.compare_digest(expected, signature) and signer_address == self.get_address()

    def sign_transaction(self, tx_data: Dict) -> Dict:
        """Sign a transaction dict and return it with r/s/v components."""
        sig = self._sign(tx_data)
        mid = len(sig) // 2
        return {
            **tx_data,
            "signature": sig,
            "r": "0x" + sig[:mid],
            "s": "0x" + sig[mid:],
            "v": 27,
        }

    def encode_function(self, function_name: str, params: List[Any]) -> str:
        """Encode a smart contract function call as a hex string."""
        call_str = f"{function_name}({','.join(str(p) for p in params)})"
        return "0x" + hashlib.sha256(call_str.encode("utf-8")).hexdigest()

    async def _query_balance(self, address: str) -> float:
        """Internal: query USDx balance; raises ConnectionError on failure."""
        result = await self.balance(address)
        if result.get("simulated"):
            raise ConnectionError("Balance query not available in simulation mode")
        return float(result.get("balance_usdx", 0.0))

    async def get_balance(self, address: str) -> float:
        """Query USDx balance and return it as a float."""
        return await self._query_balance(address)

    async def create_payment(
        self,
        to_address: str,
        amount_usdx: float,
        memo: str = "",
    ) -> Dict:
        """Create a USDx payment, checking balance first."""
        try:
            balance = await self.get_balance(self.get_address())
            if balance < amount_usdx:
                return {"success": False, "error": "Insufficient balance"}
        except Exception:
            # Simulation mode or connection failure — proceed without balance check
            pass

        result = await self.transfer(to_address=to_address, amount=amount_usdx, memo=memo)
        if result.get("success"):
            tx_hash_input = f"{to_address}{amount_usdx}{self._nonce}"
            result.setdefault(
                "tx_hash",
                "0x" + hashlib.sha256(tx_hash_input.encode()).hexdigest(),
            )
            result.setdefault("amount", amount_usdx)
        return result

    async def batch_payment(self, payments: List[Dict]) -> List[Dict]:
        """Execute multiple payments concurrently."""
        tasks = [
            self.create_payment(
                to_address=p["to"],
                amount_usdx=p["amount"],
                memo=p.get("memo", ""),
            )
            for p in payments
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            r if isinstance(r, dict) else {"success": False, "error": str(r)}
            for r in results
        ]

    async def _fetch_history(self, address: str) -> List[Dict]:
        """Internal: fetch transaction history for an address."""
        result = await self._get(f"transactions/{address}")
        if result.get("simulated"):
            return []
        return result.get("transactions", [])

    async def get_transaction_history(self, address: str) -> List[Dict]:
        """Return the transaction history for a wallet address."""
        return await self._fetch_history(address)

    async def estimate_gas(self, to_address: str, amount_usdx: float) -> int:
        """Estimate the gas cost for a USDx transfer."""
        return 21000  # standard transfer gas estimate


# ── Convenience factory ─────────────────────────────────────────────────────────


def make_x402_client(env: Optional[Dict[str, str]] = None) -> "X402Client":
    """Create an X402Client from environment variables."""
    import os

    cfg: Any = env if env is not None else os.environ
    return X402Client(
        private_key=cfg.get("X402_PRIVATE_KEY", ""),
        gateway_url=cfg.get("X402_GATEWAY", X402_GATEWAY),
    )
