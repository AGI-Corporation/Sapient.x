"""Tests for MCP (Model Context Protocol) toolkit."""

from unittest.mock import AsyncMock, patch

import pytest

# ── MCPToolkit Tests ──────────────────────────────────────────────────────────────


def test_mcp_toolkit_creation(mcp_toolkit):
    """Test MCPToolkit initialization."""
    assert mcp_toolkit is not None
    assert mcp_toolkit.agent_id == "test-mcp-001"
    assert mcp_toolkit.local_only is True


@pytest.mark.asyncio
async def test_mcp_toolkit_list_tools(mcp_toolkit):
    """Test listing available MCP tools."""
    tools = await mcp_toolkit.list_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0

    # Check tool structure
    for tool in tools:
        assert "name" in tool
        assert "description" in tool


@pytest.mark.asyncio
async def test_mcp_send_message(mcp_toolkit):
    """Test sending MCP message to another agent."""
    result = await mcp_toolkit.send_message(
        target_id="agent-002", content={"type": "greeting", "message": "Hello"}
    )

    assert result["success"] is True
    assert "message_id" in result


@pytest.mark.asyncio
async def test_mcp_receive_message(mcp_toolkit):
    """Test receiving MCP messages."""
    # Simulate incoming message
    mock_message = {
        "from": "agent-002",
        "to": "test-mcp-001",
        "payload": {"type": "response", "data": "test"},
        "sent_at": "2026-03-07T20:00:00Z",
    }

    mcp_toolkit.inject_message(mock_message)
    messages = await mcp_toolkit.receive_messages()
    assert len(messages) == 1
    assert messages[0]["from"] == "agent-002"


@pytest.mark.asyncio
async def test_mcp_call_tool(mcp_toolkit):
    """Test calling an MCP tool."""
    result = await mcp_toolkit.call_tool(
        tool_name="parcel_get_place_hierarchy", parameters={"lat": 37.7749, "lon": -122.4194}
    )

    assert result["success"] is True
    assert "data" in result


@pytest.mark.asyncio
async def test_mcp_register_custom_tool(mcp_toolkit):
    """Test registering a custom tool."""

    async def custom_tool(param1: str, param2: int) -> dict:
        return {"result": f"{param1}-{param2}"}

    mcp_toolkit.register_tool(
        name="custom_tool",
        func=custom_tool,
        description="A custom test tool",
        parameters={
            "param1": {"type": "string", "required": True},
            "param2": {"type": "integer", "required": True},
        },
    )

    tools = await mcp_toolkit.list_tools()
    tool_names = [t["name"] for t in tools]
    assert "custom_tool" in tool_names


@pytest.mark.asyncio
async def test_mcp_broadcast_message(mcp_toolkit):
    """Test broadcasting message to multiple agents."""
    target_ids = ["agent-002", "agent-003", "agent-004"]

    with patch.object(mcp_toolkit, "send_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = {"success": True, "message_id": "msg-123"}

        results = await mcp_toolkit.broadcast(
            target_ids=target_ids, content={"type": "announcement", "message": "Update available"}
        )

        assert len(results) == 3
        assert all(r["success"] for r in results)
        assert mock_send.call_count == 3


def test_mcp_validate_message_format(mcp_toolkit):
    """Test message format validation."""
    valid_message = {
        "from": "agent-001",
        "to": "agent-002",
        "payload": {"type": "test"},
        "sent_at": "2026-03-07T20:00:00Z",
    }

    assert mcp_toolkit.validate_message(valid_message) is True

    invalid_message = {"from": "agent-001"}  # Missing required fields
    assert mcp_toolkit.validate_message(invalid_message) is False


@pytest.mark.asyncio
async def test_mcp_tool_error_handling(mcp_toolkit):
    """Test error handling in tool calls."""
    result = await mcp_toolkit.call_tool(tool_name="nonexistent_tool", parameters={})

    assert result["success"] is False
    assert "error" in result
    assert "not found" in result["error"].lower()


@pytest.mark.asyncio
async def test_mcp_message_queue(mcp_toolkit):
    """Test message queueing system."""
    # Send multiple messages
    for i in range(5):
        await mcp_toolkit.send_message(target_id=f"agent-{i:03d}", content={"index": i})

    # Check queue status
    queue_size = mcp_toolkit.get_queue_size()
    assert queue_size == 5


def test_mcp_tool_parameter_validation(mcp_toolkit):
    """Test tool parameter validation."""
    tool_spec = {
        "name": "test_tool",
        "input_schema": {
            "required": ["required_param"],
            "properties": {
                "required_param": {"type": "string"},
                "optional_param": {"type": "integer"},
            },
        },
    }

    # Valid parameters
    valid_params = {"required_param": "value"}
    valid, err = mcp_toolkit.validate_parameters(tool_spec, valid_params)
    assert valid is True

    # Missing required parameter
    invalid_params = {"optional_param": 42}
    valid, err = mcp_toolkit.validate_parameters(tool_spec, invalid_params)
    assert valid is False


@pytest.mark.asyncio
async def test_mcp_connection_status(mcp_toolkit):
    """Test MCP connection status checks."""
    status = await mcp_toolkit.get_connection_status()

    assert "connected" in status
    assert "agent_id" in status
    assert status["agent_id"] == "test-mcp-001"


@pytest.mark.asyncio
async def test_mcp_retry_mechanism(mcp_toolkit):
    """Test automatic retry on failed messages."""
    with patch.object(mcp_toolkit, "send", new_callable=AsyncMock) as mock_send:
        # First two calls fail, third succeeds
        mock_send.side_effect = [
            Exception("Network error"),
            Exception("Network error"),
            {"success": True, "message_id": "msg-123"},
        ]

        result = await mcp_toolkit.send_message(
            target_id="agent-002", content={"test": "data"}, max_retries=3
        )

        assert result["success"] is True
        assert mock_send.call_count == 3
