# Sapient.x — Open Agentic Spatial Operating System

Sapient.x is a metaverse-native operating system that empowers autonomous digital agents to manage, trade, and optimize virtual land (parcels).

## Vision
To create a fully autonomous economy in the metaverse where digital agents act as stewards of spatial assets, governed by the X402 financial protocol and the Model Context Protocol (MCP).

## Core Pillars
- **Spatial Fabric:** Integrated OpenStreetMap data, underground utility metrics, and "Stikk" loyalty points.
- **Agent Orchestration:** Autonomous ParcelAgents and TradeAgents built with LangGraph and AI reasoning.
- **Interoperability:** Native support for MCP tool discovery and inter-agent communication.
- **Financial Layer:** Deterministic wallet addresses and signed transactions using the X402 protocol (HMAC-SHA256).
- **NANDA Registry:** Capability-based discovery for agent-to-agent interaction.

## Architecture
- **API:** FastAPI REST endpoints for external visibility and management.
- **Agents:** Python-based autonomous entities with local state and X402 wallets.
- **Protocol:** JSON-RPC based MCP tools for system-level actions (spatial, financial, messaging).
- **Orchestration:** LangGraph-driven optimization workflows for complex decision-making.

## Tech Stack
- **Languages:** Python 3.10+, TypeScript/Express (for core routers/servers).
- **Frameworks:** FastAPI, Pydantic, LangGraph.
- **Protocols:** MCP (Model Context Protocol), X402 (Financial Protocol).
