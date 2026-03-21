# Sapient.x — Requirements Traceability Matrix

## V1: Core Infrastructure
- **[FOUNDATION-01]** Core `ParcelAgent` state management and local state persistence.
- **[FOUNDATION-02]** `TradeAgent` orchestration for multi-agent trade negotiations.
- **[FINANCE-01]** X402 client implementation with HMAC-SHA256 deterministic signing.
- **[FINANCE-02]** USDx stablecoin support with deposit and transfer logic.
- **[MCP-01]** Model Context Protocol (MCP) toolkit for local and remote tool discovery.
- **[MCP-02]** Standardized messaging envelope (`from`, `to`, `payload`, `sent_at`) via Route.X.

## V2: Spatial & Registry Integrations
- **[SPATIAL-01]** Stikk loyalty spot discovery via MCP tools.
- **[SPATIAL-02]** Underground utility risk assessment integration for autonomous agents.
- **[REGISTRY-01]** NANDA Agent Registry for capability-based agent discovery.
- **[REGISTRY-02]** Autonomous agent self-registration and discovery logic.

## V3: Optimization & Scale (Current)
- **[OPTIMIZE-01]** LangGraph-driven parcel optimization workflows using Sentient Foundation models.
- **[OPTIMIZE-02]** Parallel execution of trade offers and broadcasts.
- **[API-01]** Standardized FastAPI REST endpoints for all core modules.
- **[UX-01]** Global error handling and standardized response models.

## Out of Scope
- Direct blockchain interaction (currently simulated via X402 protocol logic).
- Non-USDx asset support.
- Human-to-agent chat interface (agents communicate only with other agents).
