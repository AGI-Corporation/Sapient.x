"""BitrefillAgent — Web4AGI

Autonomous shopping agent that integrates with Bitrefill's eCommerce MCP API
to search for gift cards, eSIMs, mobile top-ups, and more, then pays for them
via the x402 payment protocol.

Bitrefill API docs: https://docs.bitrefill.com/docs/ecommerce-mcp
x402 protocol: https://x402.org
"""

import os
from base64 import b64encode
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None

from src.payments.x402_client import X402Client

BITREFILL_API_BASE = "https://www.bitrefill.com/api/v1"


# ── Bitrefill HTTP Client ──────────────────────────────────────────────────────


class BitrefillClient:
    """Thin async HTTP client wrapping the Bitrefill REST API.

    Public endpoints (search, categories, detail, ping) require no credentials.
    Authenticated endpoints (invoices, orders, account) require BITREFILL_API_ID
    and BITREFILL_API_SECRET set as environment variables or passed explicitly.
    """

    def __init__(
        self,
        api_id: str | None = None,
        api_secret: str | None = None,
        base_url: str = BITREFILL_API_BASE,
        timeout: int = 30,
    ):
        self.api_id = api_id or os.getenv("BITREFILL_API_ID", "")
        self.api_secret = api_secret or os.getenv("BITREFILL_API_SECRET", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ── Internal Helpers ───────────────────────────────────────────────────────

    def _auth_header(self) -> dict[str, str]:
        """Build HTTP Basic Auth header from API credentials."""
        if not self.api_id or not self.api_secret:
            return {}
        token = b64encode(f"{self.api_id}:{self.api_secret}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    def _is_authenticated(self) -> bool:
        return bool(self.api_id and self.api_secret)

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        authenticated: bool = False,
    ) -> dict:
        if httpx is None:
            return {"simulated": True, "path": path, "params": params}
        headers = self._auth_header() if authenticated else {}
        url = f"{self.base_url}/{path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url, params=params or {}, headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def _post(
        self,
        path: str,
        body: dict,
        authenticated: bool = True,
    ) -> dict:
        if httpx is None:
            return {"simulated": True, "path": path, "body": body}
        headers = self._auth_header() if authenticated else {}
        url = f"{self.base_url}/{path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            return resp.json()

    # ── Public Endpoints ───────────────────────────────────────────────────────

    async def ping(self) -> dict:
        """Check whether the Bitrefill API is available."""
        return await self._get("ping")

    async def get_categories(self) -> dict:
        """Return the full product-type → categories map."""
        return await self._get("categories")

    async def search(
        self,
        query: str,
        *,
        country: str | None = None,
        language: str | None = None,
        category: str | None = None,
        limit: int = 10,
        skip: int = 0,
    ) -> dict:
        """Search for gift cards, eSIMs, mobile top-ups, and more."""
        params: dict[str, Any] = {"q": query, "limit": limit, "skip": skip}
        if country:
            params["country"] = country
        if language:
            params["language"] = language
        if category:
            params["category"] = category
        return await self._get("products", params=params)

    async def get_product_detail(self, product_id: str) -> dict:
        """Get detailed information about a product by its ID."""
        return await self._get(f"products/{product_id}")

    # ── Authenticated Endpoints ────────────────────────────────────────────────

    async def get_account_balance(self) -> dict:
        """Retrieve the authenticated account balance."""
        return await self._get("account/balance", authenticated=True)

    async def create_invoice(
        self,
        products: list[dict[str, Any]],
        payment_method: str = "balance",
        *,
        webhook_url: str | None = None,
        auto_pay: bool = False,
    ) -> dict:
        """Create a new Bitrefill invoice.

        Args:
            products: List of product dicts, e.g.
                [{"product_id": "amazon-us-50", "quantity": 1}]
            payment_method: One of "balance", "bitcoin", "lightning".
            webhook_url: Optional URL to receive order status webhooks.
            auto_pay: If True, automatically pay with account balance.
        """
        body: dict[str, Any] = {
            "products": products,
            "payment_method": payment_method,
        }
        if webhook_url:
            body["webhook_url"] = webhook_url
        if auto_pay:
            body["auto_pay"] = auto_pay
        return await self._post("invoices", body)

    async def get_invoices(
        self,
        *,
        start: int = 0,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
    ) -> dict:
        """List invoices for the authenticated account."""
        params: dict[str, Any] = {"start": start, "limit": limit}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        return await self._get("invoices", params=params, authenticated=True)

    async def get_invoice(self, invoice_id: str) -> dict:
        """Get details of a specific invoice by ID."""
        return await self._get(f"invoices/{invoice_id}", authenticated=True)

    async def pay_invoice(self, invoice_id: str) -> dict:
        """Pay an unpaid invoice using the account balance."""
        return await self._post(f"invoices/{invoice_id}/pay", body={})

    async def get_orders(
        self,
        *,
        start: int = 0,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
    ) -> dict:
        """List orders for the authenticated account."""
        params: dict[str, Any] = {"start": start, "limit": limit}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        return await self._get("orders", params=params, authenticated=True)

    async def get_order(self, order_id: str) -> dict:
        """Get details of a specific order by ID."""
        return await self._get(f"orders/{order_id}", authenticated=True)

    async def unseal_order(self, order_id: str) -> dict:
        """Reveal codes and PINs for a specific order."""
        return await self._get(f"orders/{order_id}/unseal", authenticated=True)


# ── BitrefillAgent ─────────────────────────────────────────────────────────────


class BitrefillAgent:
    """Autonomous eCommerce agent that shops on Bitrefill and pays via x402.

    Workflow:
      1. Search for products using natural-language queries.
      2. Retrieve product details and select the best match.
      3. Create a Bitrefill invoice for the desired product.
      4. Pay the invoice either via:
         a) x402 payment protocol (USDC/stablecoin on-chain), or
         b) Bitrefill account balance (fallback).
      5. Retrieve and return the fulfilled order.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        bitrefill_api_id: str | None = None,
        bitrefill_api_secret: str | None = None,
        wallet_private_key: str | None = None,
        x402_gateway: str | None = None,
    ):
        self.agent_id = agent_id or "bitrefill-agent"
        self.bitrefill = BitrefillClient(
            api_id=bitrefill_api_id,
            api_secret=bitrefill_api_secret,
        )
        x402_kwargs: dict[str, Any] = {}
        if wallet_private_key:
            x402_kwargs["private_key"] = wallet_private_key
        if x402_gateway:
            x402_kwargs["gateway_url"] = x402_gateway
        self.x402 = X402Client(**x402_kwargs)

    # ── Discovery ──────────────────────────────────────────────────────────────

    async def ping(self) -> dict:
        """Check Bitrefill API availability."""
        return await self.bitrefill.ping()

    async def get_categories(self) -> dict:
        """Return Bitrefill product categories."""
        return await self.bitrefill.get_categories()

    async def search(
        self,
        query: str,
        country: str | None = None,
        category: str | None = None,
        limit: int = 10,
    ) -> dict:
        """Search Bitrefill for products matching the query."""
        return await self.bitrefill.search(
            query=query,
            country=country,
            category=category,
            limit=limit,
        )

    async def get_product_detail(self, product_id: str) -> dict:
        """Get full product details for a given product ID."""
        return await self.bitrefill.get_product_detail(product_id)

    # ── Purchasing ─────────────────────────────────────────────────────────────

    async def purchase(
        self,
        product_id: str,
        quantity: int = 1,
        value: float | None = None,
        payment_method: str = "balance",
        use_x402: bool = False,
        x402_amount_usdx: float | None = None,
        x402_recipient: str | None = None,
        webhook_url: str | None = None,
        extra_product_fields: dict[str, Any] | None = None,
    ) -> dict:
        """End-to-end purchase: create invoice → pay → return result.

        The Bitrefill API returns the invoice ID in the ``id`` field; the
        ``invoice_id`` field is checked as a fallback for forward-compatibility.

        Args:
            product_id: Bitrefill product identifier (e.g. "amazon-us-50").
            quantity: Number of units to purchase.
            value: Denomination value (required for variable-denomination products).
            payment_method: One of "balance", "bitcoin", or "lightning".
            use_x402: If True, settle payment via the x402 protocol *before*
                      paying the Bitrefill invoice with balance.  Both
                      ``x402_amount_usdx`` and ``x402_recipient`` are required.
            x402_amount_usdx: Amount in USDx to transfer via x402 (required
                              when use_x402 is True).
            x402_recipient: On-chain address to receive the stablecoin payment
                            (e.g. a Bitrefill escrow wallet or your own wallet
                            for top-up).  Required when use_x402 is True.
            webhook_url: Webhook URL for order status notifications.
            extra_product_fields: Additional product fields (phone_number, email, etc.).
        """
        product: dict[str, Any] = {"product_id": product_id, "quantity": quantity}
        if value is not None:
            product["value"] = value
        if extra_product_fields:
            product.update(extra_product_fields)

        # Step 1: Create invoice
        invoice_resp = await self.bitrefill.create_invoice(
            products=[product],
            payment_method=payment_method,
            webhook_url=webhook_url,
        )

        invoice_id = invoice_resp.get("id") or invoice_resp.get("invoice_id")
        if not invoice_id and not invoice_resp.get("simulated"):
            return {"success": False, "error": "Invoice creation failed", "details": invoice_resp}

        # Step 2 (optional): Settle via x402 before paying Bitrefill invoice.
        # When use_x402=True, the caller must provide x402_amount_usdx and
        # x402_recipient (the on-chain address to receive the stablecoin payment,
        # e.g. a Bitrefill escrow wallet or the caller's own wallet for top-up).
        x402_result: dict | None = None
        if use_x402:
            if not x402_amount_usdx:
                return {"success": False, "error": "x402_amount_usdx is required when use_x402=True"}
            if not x402_recipient:
                return {
                    "success": False,
                    "error": "x402_recipient is required when use_x402=True (provide the escrow or top-up wallet address)",
                }
            x402_result = await self.x402.transfer(
                to_address=x402_recipient,
                amount=x402_amount_usdx,
                memo=f"bitrefill:invoice:{invoice_id}",
            )
            if not x402_result.get("success") and not x402_result.get("simulated"):
                return {
                    "success": False,
                    "error": "x402 payment failed",
                    "x402": x402_result,
                    "invoice": invoice_resp,
                }

        # Step 3: Pay the Bitrefill invoice (balance method)
        if payment_method == "balance":
            pay_resp = await self.bitrefill.pay_invoice(invoice_id or "")
        else:
            # For bitcoin/lightning the client handles payment externally;
            # we return the invoice with payment instructions.
            pay_resp = {
                "status": "awaiting_payment",
                "payment_method": payment_method,
                "invoice": invoice_resp,
            }

        return {
            "success": True,
            "invoice_id": invoice_id,
            "invoice": invoice_resp,
            "payment": pay_resp,
            "x402": x402_result,
        }

    # ── Account & Order Management ──────────────────────────────────────────────

    async def get_account_balance(self) -> dict:
        """Return authenticated Bitrefill account balance."""
        return await self.bitrefill.get_account_balance()

    async def get_invoices(self, limit: int = 20) -> dict:
        """List recent invoices."""
        return await self.bitrefill.get_invoices(limit=limit)

    async def get_invoice(self, invoice_id: str) -> dict:
        """Get a specific invoice by ID."""
        return await self.bitrefill.get_invoice(invoice_id)

    async def get_orders(self, limit: int = 20) -> dict:
        """List recent orders."""
        return await self.bitrefill.get_orders(limit=limit)

    async def get_order(self, order_id: str) -> dict:
        """Get a specific order by ID."""
        return await self.bitrefill.get_order(order_id)

    async def unseal_order(self, order_id: str) -> dict:
        """Reveal codes and PINs for a completed order."""
        return await self.bitrefill.unseal_order(order_id)


# ── Factory ────────────────────────────────────────────────────────────────────


def make_bitrefill_agent(env: dict[str, str] | None = None) -> BitrefillAgent:
    """Create a BitrefillAgent from environment variables."""
    cfg: Any = env if env is not None else os.environ
    return BitrefillAgent(
        bitrefill_api_id=cfg.get("BITREFILL_API_ID", ""),
        bitrefill_api_secret=cfg.get("BITREFILL_API_SECRET", ""),
        wallet_private_key=cfg.get("X402_PRIVATE_KEY", ""),
        x402_gateway=cfg.get("X402_GATEWAY", ""),
    )
