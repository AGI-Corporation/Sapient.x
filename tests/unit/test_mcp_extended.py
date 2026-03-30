"""Unit tests for MCPToolkit uncovered paths."""

from unittest.mock import AsyncMock, patch

import pytest

from src.mcp.mcp_tools import MCPToolkit, _LOCAL_TOOLS, register_tool


@pytest.fixture
def toolkit():
    return MCPToolkit(agent_id="ext-mcp-001", local_only=True)


# ── inject_message ─────────────────────────────────────────────────────────


def test_inject_message_adds_to_inbox(toolkit):
    msg = {"from": "agent-x", "to": "ext-mcp-001", "payload": {"data": 1}}
    toolkit.inject_message(msg)
    assert toolkit.get_queue_size() == 1


def test_inject_multiple_messages(toolkit):
    for i in range(3):
        toolkit.inject_message({"index": i})
    assert toolkit.get_queue_size() == 3


@pytest.mark.asyncio
async def test_inject_then_receive(toolkit):
    msg = {"from": "a", "to": "b", "payload": {}}
    toolkit.inject_message(msg)
    received = await toolkit.receive()
    assert len(received) == 1
    assert received[0] == msg


# ── local tool exception handling ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_tool_local_exception():
    """When a local tool raises, the error is captured in MCPResult."""
    tk = MCPToolkit(agent_id="err-mcp", local_only=True)

    async def bad_tool(**kwargs):
        raise RuntimeError("tool broken")

    tk.register_tool("bad_tool", bad_tool)
    result = await tk.call_tool("bad_tool", parameters={})
    assert result["success"] is False
    assert "tool broken" in result["error"]


# ── validate_message with 'content' key ───────────────────────────────────


def test_validate_message_content_format(toolkit):
    """Messages using 'content'/'timestamp' keys are also valid."""
    msg = {
        "from": "a",
        "to": "b",
        "content": {"type": "ping"},
        "timestamp": "2026-01-01T00:00:00Z",
    }
    assert toolkit.validate_message(msg) is True


def test_validate_message_content_format_missing_field(toolkit):
    msg = {"from": "a", "to": "b", "content": {"type": "ping"}}  # missing timestamp
    assert toolkit.validate_message(msg) is False


def test_validate_message_payload_format_valid(toolkit):
    msg = {
        "from": "a",
        "to": "b",
        "payload": {},
        "sent_at": "2026-01-01T00:00:00Z",
    }
    assert toolkit.validate_message(msg) is True


def test_validate_message_payload_format_missing_field(toolkit):
    msg = {"from": "a", "payload": {}, "sent_at": "now"}  # missing 'to'
    assert toolkit.validate_message(msg) is False


# ── send retry mechanism ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_retries_on_failure_then_succeeds(toolkit):
    success_response = {
        "success": True,
        "data": None,
        "message_id": "msg-ok",
        "error": None,
    }
    with patch.object(toolkit, "_send_raw", new_callable=AsyncMock) as mock_raw:
        mock_raw.side_effect = [
            Exception("timeout"),
            Exception("timeout"),
            success_response,
        ]
        result = await toolkit.send(to="agent-zzz", payload={"x": 1}, max_retries=3)
    assert result["success"] is True
    assert mock_raw.call_count == 3


@pytest.mark.asyncio
async def test_send_all_retries_exhausted(toolkit):
    with patch.object(toolkit, "_send_raw", new_callable=AsyncMock) as mock_raw:
        mock_raw.side_effect = Exception("permanent failure")
        result = await toolkit.send(to="agent-zzz", payload={}, max_retries=2)
    assert result["success"] is False
    assert "permanent failure" in result["error"]
    assert mock_raw.call_count == 2


# ── call_tool local_only with unknown tool ─────────────────────────────────


@pytest.mark.asyncio
async def test_call_tool_unknown_local_only(toolkit):
    result = await toolkit.call_tool("completely_unknown_tool_xyz", parameters={})
    assert result["success"] is False
    assert "not found" in result["error"].lower()


# ── get_location_data fallback ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_tool_get_location_data_fallback(toolkit):
    """get_location_data is available as a built-in fallback in local_only mode."""
    # Temporarily remove from _LOCAL_TOOLS if present
    original = _LOCAL_TOOLS.pop("get_location_data", None)
    try:
        result = await toolkit.call_tool("get_location_data", parameters={})
        assert result["success"] is True
        assert result["data"]["lat"] == 37.7
    finally:
        if original is not None:
            _LOCAL_TOOLS["get_location_data"] = original


# ── list_tools local_only ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_tools_local_only_includes_get_location_data(toolkit):
    original = _LOCAL_TOOLS.pop("get_location_data", None)
    try:
        tools = await toolkit.list_tools()
        names = [t["name"] for t in tools]
        assert "get_location_data" in names
    finally:
        if original is not None:
            _LOCAL_TOOLS["get_location_data"] = original


@pytest.mark.asyncio
async def test_list_tools_does_not_duplicate_get_location_data():
    """When get_location_data is already in _LOCAL_TOOLS, it is not duplicated."""
    async def loc_tool(**_):
        return {}

    _LOCAL_TOOLS["get_location_data"] = loc_tool
    tk = MCPToolkit(agent_id="dedup-test", local_only=True)
    tools = await tk.list_tools()
    names = [t["name"] for t in tools]
    assert names.count("get_location_data") == 1


# ── register_tool decorator ────────────────────────────────────────────────


def test_register_tool_decorator():
    """The @register_tool decorator adds the function to _LOCAL_TOOLS."""
    @register_tool("test.decorator_tool")
    async def my_tool(**kwargs):
        return {"ok": True}

    assert "test.decorator_tool" in _LOCAL_TOOLS
    _LOCAL_TOOLS.pop("test.decorator_tool", None)


# ── call_tool via registered local tool ───────────────────────────────────


@pytest.mark.asyncio
async def test_call_registered_local_tool(toolkit):
    async def greet(name: str) -> dict:
        return {"greeting": f"Hello {name}"}

    toolkit.register_tool("say_hello", greet)
    result = await toolkit.call_tool("say_hello", parameters={"name": "World"})
    assert result["success"] is True
    assert result["data"]["greeting"] == "Hello World"


# ── validate_parameters edge cases ────────────────────────────────────────


def test_validate_parameters_all_optional(toolkit):
    spec = {
        "parameters": {
            "opt1": {"type": "string", "required": False},
            "opt2": {"type": "int", "required": False},
        }
    }
    assert toolkit.validate_parameters(spec, {}) is True


def test_validate_parameters_empty_spec(toolkit):
    assert toolkit.validate_parameters({}, {"extra": "ok"}) is True


# ── send_message alias ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_message_alias(toolkit):
    """send_message should behave identically to send."""
    result = await toolkit.send_message(target_id="t", content={"a": 1})
    assert result["success"] is True


# ── receive_messages alias ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_receive_messages_alias(toolkit):
    toolkit.inject_message({"test": True})
    msgs = await toolkit.receive_messages()
    assert len(msgs) == 1
