"""NANDARegistry — Web4AGI

Implements the NANDA (Network for Agentic and Networked Discovery of Agents)
protocol for decentralized agent registration and discovery.

Each parcel agent registers an AgentFact that describes its:
  - Capabilities (trade, lease, optimize, communicate)
  - Location / region tag
  - MCP tool endpoints
  - Wallet address
  - Protocol support (x402, MCP, NANDA)

Reference: https://nanda.network
"""

import uuid
from datetime import UTC, datetime
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None


NANDA_REGISTRY_URL = "https://registry.nanda.network/v1"


class AgentFact:
    """Represents a registered agent's discoverable metadata."""

    def __init__(
        self,
        agent_id: str,
        parcel_id: str,
        owner_address: str,
        capabilities: list[str],
        location: dict[str, float],
        region_tag: str = "",
        protocols: list[str] | None = None,
        mcp_endpoint: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.parcel_id = parcel_id
        self.owner_address = owner_address
        self.capabilities = capabilities
        self.location = location
        self.region_tag = region_tag
        self.protocols = protocols or ["x402", "mcp", "nanda"]
        self.mcp_endpoint = mcp_endpoint
        self.metadata = metadata or {}
        self.registered_at = datetime.now(UTC).isoformat()
        self.updated_at = self.registered_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "parcel_id": self.parcel_id,
            "owner_address": self.owner_address,
            "capabilities": self.capabilities,
            "location": self.location,
            "region_tag": self.region_tag,
            "protocols": self.protocols,
            "mcp_endpoint": self.mcp_endpoint,
            "metadata": self.metadata,
            "registered_at": self.registered_at,
            "updated_at": self.updated_at,
        }


class NANDARegistry:
    """Local + remote NANDA agent registry.

    In local_only mode (default) acts as an in-process registry.
    When a remote URL is configured, it mirrors registrations to the
    NANDA network so other agents can discover this one.
    """

    def __init__(
        self,
        registry_url: str = NANDA_REGISTRY_URL,
        region_tag: str = "us-ca-sf-frontier-corridor",
        local_only: bool = True,
    ) -> None:
        self.registry_url = registry_url.rstrip("/")
        self.region_tag = region_tag
        self.local_only = local_only
        self._local: dict[str, AgentFact] = {}

    # ── Registration ───────────────────────────────────────────────────────────

    async def register(
        self,
        parcel_id: str,
        owner_address: str,
        capabilities: list[str],
        location: dict[str, float],
        region_tag: str = "",
        protocols: list[str] | None = None,
        mcp_endpoint: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AgentFact:
        """Register a parcel agent and return the AgentFact."""
        agent_id = f"nanda-{uuid.uuid4().hex[:12]}"
        fact = AgentFact(
            agent_id=agent_id,
            parcel_id=parcel_id,
            owner_address=owner_address,
            capabilities=capabilities,
            location=location,
            region_tag=region_tag or self.region_tag,
            protocols=protocols,
            mcp_endpoint=mcp_endpoint,
            metadata=metadata or {},
        )
        self._local[agent_id] = fact

        if not self.local_only and httpx is not None:
            await self._remote_register(fact)

        return fact

    async def _remote_register(self, fact: AgentFact) -> None:
        """Push registration to the remote NANDA registry."""
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                await client.post(f"{self.registry_url}/agents", json=fact.to_dict())
            except Exception:
                pass  # Best-effort; local registry always succeeds

    # ── Deregistration ────────────────────────────────────────────────────────

    async def deregister(self, agent_id: str) -> bool:
        if agent_id not in self._local:
            return False
        del self._local[agent_id]
        return True

    # ── Discovery ─────────────────────────────────────────────────────────────

    async def find(
        self,
        capability: str | None = None,
        region_tag: str | None = None,
        protocol: str | None = None,
        limit: int = 50,
    ) -> list[AgentFact]:
        """Find agents matching the given filters."""
        agents = list(self._local.values())

        if capability:
            agents = [a for a in agents if capability in a.capabilities]
        if region_tag:
            agents = [a for a in agents if a.region_tag == region_tag]
        if protocol:
            agents = [a for a in agents if protocol in a.protocols]

        if not self.local_only and httpx is not None:
            remote = await self._remote_find(capability, region_tag, protocol)
            seen = {a.agent_id for a in agents}
            agents += [a for a in remote if a.agent_id not in seen]

        return agents[:limit]

    async def _remote_find(
        self,
        capability: str | None,
        region_tag: str | None,
        protocol: str | None,
    ) -> list[AgentFact]:
        """Query the remote NANDA registry."""
        params: dict[str, str] = {}
        if capability:
            params["capability"] = capability
        if region_tag:
            params["region_tag"] = region_tag
        if protocol:
            params["protocol"] = protocol

        async with httpx.AsyncClient(timeout=10) as client:
            try:
                resp = await client.get(f"{self.registry_url}/agents", params=params)
                resp.raise_for_status()
                items = resp.json().get("agents", [])
                return [AgentFact(**item) for item in items]
            except Exception:
                return []

    async def get(self, agent_id: str) -> AgentFact | None:
        """Retrieve a specific agent fact by ID."""
        return self._local.get(agent_id)

    async def update(self, agent_id: str, **kwargs: Any) -> AgentFact | None:
        """Update fields on an existing AgentFact."""
        fact = self._local.get(agent_id)
        if fact is None:
            return None
        for key, value in kwargs.items():
            if hasattr(fact, key):
                setattr(fact, key, value)
        fact.updated_at = datetime.now(UTC).isoformat()
        return fact

    def list_all(self) -> list[AgentFact]:
        return list(self._local.values())

    @property
    def agent_count(self) -> int:
        return len(self._local)
