"""WebSocket router — Web4AGI.

Real-time bidirectional communication for parcel agents:
  - Agents connect and receive push updates (trade events, market prices, etc.)
  - Supports subscribe/unsubscribe per agent_id
  - Broadcasts to all connected clients
"""

import asyncio
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections keyed by agent_id."""

    def __init__(self) -> None:
        # agent_id -> list of active websocket connections
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, agent_id: str) -> None:
        await websocket.accept()
        self.active_connections.setdefault(agent_id, []).append(websocket)

    def disconnect(self, websocket: WebSocket, agent_id: str | None = None) -> None:
        if agent_id and agent_id in self.active_connections:
            self.active_connections[agent_id] = [
                ws for ws in self.active_connections[agent_id] if ws is not websocket
            ]
            if not self.active_connections[agent_id]:
                del self.active_connections[agent_id]
        else:
            # Remove from all lists when agent_id is unknown
            for aid in list(self.active_connections):
                self.active_connections[aid] = [
                    ws for ws in self.active_connections[aid] if ws is not websocket
                ]

    async def send_personal_message(self, message: dict[str, Any], agent_id: str) -> None:
        """Send a message to all connections for a specific agent."""
        for ws in list(self.active_connections.get(agent_id, [])):
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(ws, agent_id)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        tasks = []
        for agent_id, connections in list(self.active_connections.items()):
            for ws in list(connections):
                tasks.append(ws.send_json(message))
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Clean up failed connections
            idx = 0
            for agent_id, connections in list(self.active_connections.items()):
                for ws in list(connections):
                    if idx < len(results) and isinstance(results[idx], Exception):
                        self.disconnect(ws, agent_id)
                    idx += 1


manager = ConnectionManager()


@router.websocket("/ws/{agent_id}")
async def websocket_endpoint(websocket: WebSocket, agent_id: str) -> None:
    """WebSocket endpoint for a single parcel agent."""
    await manager.connect(websocket, agent_id)
    try:
        await websocket.send_json(
            {"type": "connected", "agent_id": agent_id, "message": "WebSocket connected"}
        )
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "unknown")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "subscribe":
                # Acknowledge subscription (agent already registered on connect)
                await websocket.send_json({"type": "subscribed", "agent_id": data.get("agent_id", agent_id)})

            elif msg_type == "broadcast":
                # Broadcast a message to all connected clients
                await manager.broadcast(data.get("payload", data))

            elif msg_type == "send":
                # Direct message to another agent
                target = data.get("to")
                if target:
                    await manager.send_personal_message(data.get("payload", data), target)

            else:
                await websocket.send_json({"type": "echo", "received": data})

    except WebSocketDisconnect:
        manager.disconnect(websocket, agent_id)
    except Exception:
        manager.disconnect(websocket, agent_id)
