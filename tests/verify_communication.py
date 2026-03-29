import asyncio

import httpx

from src.agents.parcel_agent import ParcelAgent


async def run_verification():
    print("Starting agent-to-agent communication verification...")

    # 1. Initialize two agents
    # Explicitly set local_only=False so we can mock the HTTP network
    agent_a = ParcelAgent(parcel_id="agent-a", owner_address="0xAAAA")
    agent_a.mcp.local_only = False
    agent_b = ParcelAgent(parcel_id="agent-b", owner_address="0xBBBB")
    agent_b.mcp.local_only = False

    # 2. Agent A sends a message to Agent B
    content = {"type": "greeting", "text": "Hello from Agent A!"}
    print(f"Agent A sending message to Agent B: {content}")

    from unittest.mock import AsyncMock, patch

    with patch("httpx.AsyncClient.post") as mock_post, patch("httpx.AsyncClient.get") as mock_get:
        # Mock storage for the "expressor"
        inboxes = {}

        async def side_effect_post(url, json, timeout=None):
            if "/messages" in str(url):
                to_agent = json["to"]
                if to_agent not in inboxes:
                    inboxes[to_agent] = []
                inboxes[to_agent].append(json)
                return AsyncMock(
                    spec=httpx.Response,
                    status_code=200,
                    json=lambda: {"success": True, "message_id": "mock-msg-id"},
                )
            return AsyncMock(spec=httpx.Response, status_code=404)

        async def side_effect_get(url, timeout=None):
            if "/messages/" in str(url):
                agent_id = str(url).split("/")[-1]
                msgs = inboxes.get(agent_id, [])
                inboxes[agent_id] = []
                return AsyncMock(
                    spec=httpx.Response,
                    status_code=200,
                    json=lambda: {"success": True, "messages": msgs},
                )
            return AsyncMock(spec=httpx.Response, status_code=404)

        mock_post.side_effect = side_effect_post
        mock_get.side_effect = side_effect_get

        # Test sending
        send_result = await agent_a.send_message("agent-b", content)
        print(f"Send result: {send_result}")
        assert send_result["success"] is True

        # Test receiving (polling) via MCPToolkit.receive()
        received_envelopes = await agent_b.mcp.receive()
        print(f"Agent B received envelopes: {received_envelopes}")

        assert len(received_envelopes) == 1
        # In the real implementation, the payload is wrapped
        assert received_envelopes[0]["payload"]["text"] == "Hello from Agent A!"
        assert received_envelopes[0]["from"] == "agent-a"

    print("Verification successful!")


if __name__ == "__main__":
    asyncio.run(run_verification())
