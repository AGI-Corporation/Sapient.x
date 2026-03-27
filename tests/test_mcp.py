"""Tests for MCP (Model Context Protocol) toolkit — MCPToolkit in mcp_tools.py."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.mcp.mcp_tools import MCPToolkit, _LOCAL_TOOLS, register_tool


# ── Initialization Tests ───────────────────────────────────────────────────────

def test_mcp_toolkit_creation(mcp_toolkit):
    """Test MCPToolkit initialization attributes."""
    assert mcp_toolkit.agent_id == "test-mcp-001"
    assert mcp_toolkit.local_only is True
    assert mcp_toolkit.route_x_url.startswith("http")


def test_mcp_toolkit_custom_url():
    """Test MCPToolkit accepts a custom Route.X URL."""
    kit = MCPToolkit(agent_id="a1", route_x_url="http://custom:9999", local_only=True)
    assert kit.route_x_url == "http://custom:9999"


# ── Local Tool Registry Tests ──────────────────────────────────────────────────

def test_local_tools_registry_populated():
    """Local tool registry should contain the built-in tools."""
    expected_tools = {
        "parcel.get_state",
        "parcel.list_neighbors",
        "trade.create_offer",
        "trade.get_offers",
        "optimize.run",
        "payment.transfer",
    }
    assert expected_tools.issubset(set(_LOCAL_TOOLS.keys()))


def test_register_tool_decorator():
    """register_tool decorator adds the function to _LOCAL_TOOLS."""
    @register_tool("test.custom_tool")
    async def my_tool(x: int) -> dict:
        return {"x": x}

    assert "test.custom_tool" in _LOCAL_TOOLS
    # Clean up
    del _LOCAL_TOOLS["test.custom_tool"]


# ── call_tool Tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_call_tool_parcel_get_state(mcp_toolkit):
    """call_tool dispatches to the local parcel.get_state tool."""
    result = await mcp_toolkit.call_tool("parcel.get_state", parcel_id="p-001")
    assert result["tool"] == "parcel.get_state"
    assert result["parcel_id"] == "p-001"


@pytest.mark.asyncio
async def test_call_tool_parcel_list_neighbors(mcp_toolkit):
    """call_tool dispatches to parcel.list_neighbors with custom radius."""
    result = await mcp_toolkit.call_tool(
        "parcel.list_neighbors", parcel_id="p-001", radius_meters=250.0
    )
    assert result["tool"] == "parcel.list_neighbors"
    assert result["radius"] == 250.0


@pytest.mark.asyncio
async def test_call_tool_trade_create_offer(mcp_toolkit):
    """call_tool dispatches to trade.create_offer."""
    result = await mcp_toolkit.call_tool(
        "trade.create_offer", seller_id="p-001", asset="land", amount_usdx=50.0
    )
    assert result["tool"] == "trade.create_offer"
    assert result["amount"] == 50.0


@pytest.mark.asyncio
async def test_call_tool_trade_get_offers(mcp_toolkit):
    """call_tool dispatches to trade.get_offers."""
    result = await mcp_toolkit.call_tool("trade.get_offers", parcel_id="p-001")
    assert result["tool"] == "trade.get_offers"


@pytest.mark.asyncio
async def test_call_tool_optimize_run(mcp_toolkit):
    """call_tool dispatches to optimize.run."""
    result = await mcp_toolkit.call_tool(
        "optimize.run", parcel_id="p-001", context={"market": "up"}
    )
    assert result["tool"] == "optimize.run"
    assert result["parcel_id"] == "p-001"


@pytest.mark.asyncio
async def test_call_tool_payment_transfer(mcp_toolkit):
    """call_tool dispatches to payment.transfer."""
    result = await mcp_toolkit.call_tool(
        "payment.transfer", from_id="p-A", to_id="p-B", amount_usdx=15.0
    )
    assert result["tool"] == "payment.transfer"
    assert result["amount"] == 15.0


@pytest.mark.asyncio
async def test_call_tool_unknown_local_only(mcp_toolkit):
    """call_tool returns error dict for unknown tool in local_only mode."""
    result = await mcp_toolkit.call_tool("nonexistent.tool")
    assert result["success"] is False
    assert "not found" in result["error"].lower()


@pytest.mark.asyncio
async def test_call_tool_delegates_to_route_x_when_not_local():
    """call_tool falls back to Route.X when tool is not in local registry."""
    kit = MCPToolkit(agent_id="a1", local_only=False)
    with patch.object(kit, "_route_x_call", new_callable=AsyncMock) as mock_rx:
        mock_rx.return_value = {"result": "ok"}
        result = await kit.call_tool("remote.tool", param="value")
    mock_rx.assert_called_once_with("remote.tool", {"param": "value"})
    assert result == {"result": "ok"}


# ── list_tools Tests ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_tools_local_only(mcp_toolkit):
    """list_tools returns only local tools in local_only mode."""
    tools = await mcp_toolkit.list_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0
    for tool in tools:
        assert tool["source"] == "local"
        assert "name" in tool


@pytest.mark.asyncio
async def test_list_tools_includes_all_registered(mcp_toolkit):
    """list_tools includes all locally registered tool names."""
    tools = await mcp_toolkit.list_tools()
    names = {t["name"] for t in tools}
    assert "parcel.get_state" in names
    assert "trade.create_offer" in names
    assert "payment.transfer" in names


@pytest.mark.asyncio
async def test_list_tools_merges_remote_tools():
    """list_tools merges local + remote tools when not in local_only mode."""
    kit = MCPToolkit(agent_id="a1", local_only=False)
    remote_tools = [{"name": "remote.tool", "description": "A remote tool"}]

    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"tools": remote_tools}}

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )
        tools = await kit.list_tools()

    names = {t["name"] for t in tools}
    assert "remote.tool" in names


@pytest.mark.asyncio
async def test_list_tools_falls_back_on_remote_error():
    """list_tools returns only local tools if the remote call raises an exception."""
    kit = MCPToolkit(agent_id="a1", local_only=False)
    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=Exception("connection refused")
        )
        tools = await kit.list_tools()

    assert len(tools) > 0
    for t in tools:
        assert t["source"] == "local"


# ── send Tests ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_local_only_simulated(mcp_toolkit, capsys):
    """send in local_only mode prints a simulation log and returns success."""
    result = await mcp_toolkit.send(
        to="target-parcel",
        payload={"type": "greeting", "msg": "hello"},
    )
    assert result["success"] is True
    assert result["simulated"] is True
    captured = capsys.readouterr()
    assert "MCP Sim" in captured.out


@pytest.mark.asyncio
async def test_send_real_http():
    """send posts to Route.X and returns response when not local_only."""
    kit = MCPToolkit(agent_id="sender", local_only=False)
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"success": True, "msg_id": "123"}

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )
        result = await kit.send("target", {"data": "payload"})

    assert result["success"] is True


# ── receive Tests ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_receive_local_only_empty(mcp_toolkit):
    """receive returns empty list when inbox is empty in local_only mode."""
    msgs = await mcp_toolkit.receive()
    assert msgs == []


@pytest.mark.asyncio
async def test_receive_drains_inbox_after_inject(mcp_toolkit):
    """receive returns all injected messages and empties the inbox."""
    msg1 = {"from": "a", "data": 1}
    msg2 = {"from": "b", "data": 2}
    mcp_toolkit.inject_message(msg1)
    mcp_toolkit.inject_message(msg2)

    msgs = await mcp_toolkit.receive()
    assert len(msgs) == 2
    assert msg1 in msgs
    assert msg2 in msgs

    # Inbox should now be empty
    msgs2 = await mcp_toolkit.receive()
    assert msgs2 == []


@pytest.mark.asyncio
async def test_receive_real_http():
    """receive fetches messages from Route.X GET endpoint."""
    kit = MCPToolkit(agent_id="agent-x", local_only=False)
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"messages": [{"from": "y", "data": "hi"}]}

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )
        msgs = await kit.receive()

    assert len(msgs) == 1
    assert msgs[0]["from"] == "y"


# ── inject_message Tests ───────────────────────────────────────────────────────

def test_inject_message_adds_to_inbox(mcp_toolkit):
    """inject_message enqueues a message in the internal inbox."""
    assert mcp_toolkit._inbox.empty()
    mcp_toolkit.inject_message({"test": "msg"})
    assert not mcp_toolkit._inbox.empty()


def test_inject_multiple_messages(mcp_toolkit):
    """inject_message can be called multiple times to queue multiple messages."""
    for i in range(5):
        mcp_toolkit.inject_message({"index": i})
    assert mcp_toolkit._inbox.qsize() == 5


# ── _route_x_call Tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_route_x_call_posts_jsonrpc():
    """_route_x_call posts a JSON-RPC payload and returns the result."""
    kit = MCPToolkit(agent_id="rpc-agent", local_only=False)
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"result": {"data": "value"}}

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )
        result = await kit._route_x_call("some.tool", {"key": "val"})

    assert result == {"data": "value"}


@pytest.mark.asyncio
async def test_route_x_call_without_httpx():
    """_route_x_call returns simulated success when httpx is None."""
    kit = MCPToolkit(agent_id="no-httpx", local_only=False)
    with patch("src.mcp.mcp_tools.httpx", None):
        result = await kit._route_x_call("some.tool", {"k": "v"})
    assert result["success"] is True
    assert result["simulated"] is True
