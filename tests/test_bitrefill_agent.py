"""Tests for BitrefillAgent and BitrefillClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.agents.bitrefill_agent import BitrefillAgent, BitrefillClient, make_bitrefill_agent
from src.mcp.mcp_tools import MCPToolkit

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def bitrefill_client():
    """BitrefillClient with test credentials."""
    return BitrefillClient(
        api_id="test-api-id",
        api_secret="test-api-secret",
        base_url="https://www.bitrefill.com/api/v1",
    )


@pytest.fixture
def bitrefill_agent():
    """BitrefillAgent with test credentials."""
    return BitrefillAgent(
        agent_id="test-bitrefill-agent",
        bitrefill_api_id="test-api-id",
        bitrefill_api_secret="test-api-secret",
        wallet_private_key="test_x402_key",
    )


# ── BitrefillClient Tests ──────────────────────────────────────────────────────

def test_bitrefill_client_creation(bitrefill_client):
    """Test BitrefillClient initializes with credentials."""
    assert bitrefill_client.api_id == "test-api-id"
    assert bitrefill_client.api_secret == "test-api-secret"
    assert bitrefill_client.base_url == "https://www.bitrefill.com/api/v1"


def test_bitrefill_client_auth_header(bitrefill_client):
    """Test that auth header is generated from credentials."""
    header = bitrefill_client._auth_header()
    assert "Authorization" in header
    assert header["Authorization"].startswith("Basic ")


def test_bitrefill_client_no_credentials():
    """Test that unauthenticated client returns empty auth header."""
    client = BitrefillClient(api_id="", api_secret="")
    assert not client._is_authenticated()


def test_bitrefill_client_is_authenticated(bitrefill_client):
    """Test authentication status check."""
    assert bitrefill_client._is_authenticated() is True

    unauth = BitrefillClient()
    assert unauth._is_authenticated() is False


@pytest.mark.asyncio
async def test_bitrefill_ping_simulated():
    """Test ping falls back to simulation when httpx unavailable."""
    client = BitrefillClient()
    with patch("src.agents.bitrefill_agent.httpx", None):
        result = await client.ping()
    assert result.get("simulated") is True


@pytest.mark.asyncio
async def test_bitrefill_get_categories_simulated():
    """Test categories endpoint falls back to simulation mode."""
    client = BitrefillClient()
    with patch("src.agents.bitrefill_agent.httpx", None):
        result = await client.get_categories()
    assert result.get("simulated") is True


@pytest.mark.asyncio
async def test_bitrefill_search_simulated():
    """Test search falls back to simulation mode without httpx."""
    client = BitrefillClient()
    with patch("src.agents.bitrefill_agent.httpx", None):
        result = await client.search("Amazon", country="us", limit=5)
    assert result.get("simulated") is True
    assert result.get("params", {}).get("q") == "Amazon"
    assert result.get("params", {}).get("limit") == 5


@pytest.mark.asyncio
async def test_bitrefill_search_with_mock():
    """Test search passes correct query parameters."""
    client = BitrefillClient(api_id="id", api_secret="secret")
    mock_response = MagicMock()
    mock_response.json.return_value = {"results": [{"id": "amazon-us-50", "name": "Amazon $50"}]}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_http = AsyncMock()
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_http

        result = await client.search("Amazon", country="us")

    assert "results" in result


@pytest.mark.asyncio
async def test_bitrefill_create_invoice_simulated():
    """Test invoice creation falls back to simulation mode."""
    client = BitrefillClient(api_id="id", api_secret="secret")
    with patch("src.agents.bitrefill_agent.httpx", None):
        result = await client.create_invoice(
            products=[{"product_id": "amazon-us-50", "quantity": 1}],
            payment_method="balance",
        )
    assert result.get("simulated") is True
    assert result["body"]["payment_method"] == "balance"


@pytest.mark.asyncio
async def test_bitrefill_pay_invoice_simulated():
    """Test invoice payment falls back to simulation mode."""
    client = BitrefillClient(api_id="id", api_secret="secret")
    with patch("src.agents.bitrefill_agent.httpx", None):
        result = await client.pay_invoice("invoice-123")
    assert result.get("simulated") is True


@pytest.mark.asyncio
async def test_bitrefill_get_orders_simulated():
    """Test orders list falls back to simulation mode."""
    client = BitrefillClient(api_id="id", api_secret="secret")
    with patch("src.agents.bitrefill_agent.httpx", None):
        result = await client.get_orders(limit=5)
    assert result.get("simulated") is True


# ── BitrefillAgent Tests ───────────────────────────────────────────────────────

def test_bitrefill_agent_creation(bitrefill_agent):
    """Test BitrefillAgent initialization."""
    assert bitrefill_agent.agent_id == "test-bitrefill-agent"
    assert bitrefill_agent.bitrefill is not None
    assert bitrefill_agent.x402 is not None


@pytest.mark.asyncio
async def test_bitrefill_agent_ping(bitrefill_agent):
    """Test agent delegates ping to BitrefillClient."""
    with patch.object(
        bitrefill_agent.bitrefill, "ping", new_callable=AsyncMock
    ) as mock_ping:
        mock_ping.return_value = {"status": "ok"}
        result = await bitrefill_agent.ping()
    assert result["status"] == "ok"
    mock_ping.assert_called_once()


@pytest.mark.asyncio
async def test_bitrefill_agent_search(bitrefill_agent):
    """Test agent search delegates to BitrefillClient."""
    mock_results = {"results": [{"id": "netflix-us-15", "name": "Netflix $15"}]}
    with patch.object(
        bitrefill_agent.bitrefill, "search", new_callable=AsyncMock
    ) as mock_search:
        mock_search.return_value = mock_results
        result = await bitrefill_agent.search("Netflix", country="us", limit=5)
    assert result == mock_results
    mock_search.assert_called_once_with(query="Netflix", country="us", category=None, limit=5)


@pytest.mark.asyncio
async def test_bitrefill_agent_get_product_detail(bitrefill_agent):
    """Test agent fetches product detail via BitrefillClient."""
    mock_detail = {"id": "amazon-us-50", "name": "Amazon $50", "price": 50.0}
    with patch.object(
        bitrefill_agent.bitrefill, "get_product_detail", new_callable=AsyncMock
    ) as mock_detail_fn:
        mock_detail_fn.return_value = mock_detail
        result = await bitrefill_agent.get_product_detail("amazon-us-50")
    assert result["id"] == "amazon-us-50"


@pytest.mark.asyncio
async def test_bitrefill_agent_purchase_balance(bitrefill_agent):
    """Test end-to-end purchase using account balance payment."""
    invoice = {"id": "inv-001", "status": "unpaid"}
    pay_resp = {"status": "paid", "order_id": "ord-001"}

    with patch.object(
        bitrefill_agent.bitrefill, "create_invoice", new_callable=AsyncMock
    ) as mock_create, patch.object(
        bitrefill_agent.bitrefill, "pay_invoice", new_callable=AsyncMock
    ) as mock_pay:
        mock_create.return_value = invoice
        mock_pay.return_value = pay_resp

        result = await bitrefill_agent.purchase(
            product_id="amazon-us-50",
            quantity=1,
            payment_method="balance",
        )

    assert result["success"] is True
    assert result["invoice_id"] == "inv-001"
    assert result["payment"]["status"] == "paid"
    assert result["x402"] is None


@pytest.mark.asyncio
async def test_bitrefill_agent_purchase_with_x402(bitrefill_agent):
    """Test purchase with x402 payment settled first."""
    invoice = {"id": "inv-002", "status": "unpaid"}
    pay_resp = {"status": "paid", "order_id": "ord-002"}
    x402_resp = {"success": True, "tx_hash": "0xdeadbeef"}

    with patch.object(
        bitrefill_agent.bitrefill, "create_invoice", new_callable=AsyncMock
    ) as mock_create, patch.object(
        bitrefill_agent.bitrefill, "pay_invoice", new_callable=AsyncMock
    ) as mock_pay, patch.object(
        bitrefill_agent.x402, "transfer", new_callable=AsyncMock
    ) as mock_x402:
        mock_create.return_value = invoice
        mock_pay.return_value = pay_resp
        mock_x402.return_value = x402_resp

        result = await bitrefill_agent.purchase(
            product_id="amazon-us-50",
            quantity=1,
            payment_method="balance",
            use_x402=True,
            x402_amount_usdx=50.0,
            x402_recipient="0xBitrefillEscrow",
        )

    assert result["success"] is True
    assert result["x402"]["tx_hash"] == "0xdeadbeef"
    mock_x402.assert_called_once_with(
        to_address="0xBitrefillEscrow",
        amount=50.0,
        memo="bitrefill:invoice:inv-002",
    )


@pytest.mark.asyncio
async def test_bitrefill_agent_purchase_x402_missing_amount(bitrefill_agent):
    """Test purchase with x402 enabled but missing amount returns error."""
    with patch.object(
        bitrefill_agent.bitrefill, "create_invoice", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = {"id": "inv-003"}
        result = await bitrefill_agent.purchase(
            product_id="amazon-us-50",
            use_x402=True,
            # x402_amount_usdx intentionally omitted
        )
    assert result["success"] is False
    assert "x402_amount_usdx" in result["error"]


@pytest.mark.asyncio
async def test_bitrefill_agent_purchase_x402_missing_recipient(bitrefill_agent):
    """Test purchase with x402 enabled but missing recipient returns error."""
    with patch.object(
        bitrefill_agent.bitrefill, "create_invoice", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = {"id": "inv-005"}
        result = await bitrefill_agent.purchase(
            product_id="amazon-us-50",
            use_x402=True,
            x402_amount_usdx=50.0,
            # x402_recipient intentionally omitted
        )
    assert result["success"] is False
    assert "x402_recipient" in result["error"]


@pytest.mark.asyncio
async def test_bitrefill_agent_purchase_lightning(bitrefill_agent):
    """Test purchase with lightning payment returns awaiting_payment status."""
    invoice = {"id": "inv-004", "status": "unpaid", "payment_uri": "lightning:abc"}

    with patch.object(
        bitrefill_agent.bitrefill, "create_invoice", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = invoice

        result = await bitrefill_agent.purchase(
            product_id="amazon-us-50",
            payment_method="lightning",
        )

    assert result["success"] is True
    assert result["payment"]["status"] == "awaiting_payment"
    assert result["payment"]["payment_method"] == "lightning"


@pytest.mark.asyncio
async def test_bitrefill_agent_get_orders(bitrefill_agent):
    """Test get_orders delegates to BitrefillClient."""
    mock_orders = {"orders": [{"id": "ord-001"}, {"id": "ord-002"}]}
    with patch.object(
        bitrefill_agent.bitrefill, "get_orders", new_callable=AsyncMock
    ) as mock_fn:
        mock_fn.return_value = mock_orders
        result = await bitrefill_agent.get_orders(limit=10)
    assert len(result["orders"]) == 2
    mock_fn.assert_called_once_with(limit=10)


@pytest.mark.asyncio
async def test_bitrefill_agent_unseal_order(bitrefill_agent):
    """Test unseal_order reveals gift card codes."""
    sealed_result = {"order_id": "ord-001", "codes": [{"code": "GIFT-123"}]}
    with patch.object(
        bitrefill_agent.bitrefill, "unseal_order", new_callable=AsyncMock
    ) as mock_fn:
        mock_fn.return_value = sealed_result
        result = await bitrefill_agent.unseal_order("ord-001")
    assert result["codes"][0]["code"] == "GIFT-123"


# ── make_bitrefill_agent Factory Tests ──────────────────────────────────────────

def test_make_bitrefill_agent_from_env():
    """Test factory creates agent from environment variables."""
    env = {
        "BITREFILL_API_ID": "env-api-id",
        "BITREFILL_API_SECRET": "env-api-secret",
        "X402_PRIVATE_KEY": "env-private-key",
    }
    agent = make_bitrefill_agent(env=env)
    assert agent.bitrefill.api_id == "env-api-id"
    assert agent.bitrefill.api_secret == "env-api-secret"
    assert agent.x402.private_key == "env-private-key"


def test_make_bitrefill_agent_empty_env():
    """Test factory creates agent with empty credentials from empty env."""
    agent = make_bitrefill_agent(env={})
    assert agent.bitrefill.api_id == ""
    assert not agent.bitrefill._is_authenticated()


# ── MCP Tool Registration Tests ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bitrefill_mcp_tools_registered():
    """Test Bitrefill tools are registered in the MCP tool registry."""
    from src.mcp.mcp_tools import _LOCAL_TOOLS
    expected_tools = [
        "bitrefill.ping",
        "bitrefill.categories",
        "bitrefill.search",
        "bitrefill.detail",
        "bitrefill.create_invoice",
        "bitrefill.pay_invoice",
        "bitrefill.get_invoices",
        "bitrefill.get_orders",
        "bitrefill.get_order",
        "bitrefill.unseal_order",
        "bitrefill.purchase",
    ]
    for tool in expected_tools:
        assert tool in _LOCAL_TOOLS, f"MCP tool '{tool}' not registered"


@pytest.mark.asyncio
async def test_mcp_toolkit_can_call_bitrefill_search():
    """Test MCPToolkit can dispatch bitrefill.search tool call."""
    toolkit = MCPToolkit(agent_id="test-mcp-001", local_only=True)

    mock_results = {"results": [{"id": "amazon-us-50"}]}
    captured: dict = {}
    from src.mcp import mcp_tools as mcp_module

    original = mcp_module._LOCAL_TOOLS.get("bitrefill.search")
    try:

        async def mock_search(query, country=None, category=None, limit=10):
            captured["query"] = query
            captured["limit"] = limit
            return mock_results

        mcp_module._LOCAL_TOOLS["bitrefill.search"] = mock_search
        result = await toolkit.call_tool("bitrefill.search", query="Amazon", limit=3)
        assert result == mock_results
        assert captured["query"] == "Amazon"
        assert captured["limit"] == 3
    finally:
        if original is not None:
            mcp_module._LOCAL_TOOLS["bitrefill.search"] = original
