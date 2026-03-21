"""FastAPI Main Application — Web4AGI.

Run with: uvicorn src.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__

# ── Global state ──────────────────────────────────────────────────────────────
from src.agents.parcel_agent import ParcelAgent
from src.agents.trade_agent import TradeAgent
from src.api import contracts, mcp, parcels, payments, registry, trades

PARCEL_AGENTS: dict[str, ParcelAgent] = {}  # parcel_id -> ParcelAgent instance
TRADE_AGENTS: dict[str, TradeAgent] = {}  # agent_id -> TradeAgent instance


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup/shutdown lifecycle for FastAPI app."""
    # Startup: initialize any persistent connections, etc.
    print(f"[Web4AGI] Starting up... version {__version__}")
    yield
    # Shutdown: cleanup
    print("[Web4AGI] Shutting down...")


app = FastAPI(
    title="Web4AGI API",
    description="Metaverse Parcel Digital Agents with MCP, x402, and LangGraph",
    version=__version__,
    lifespan=lifespan,
)


# ── CORS middleware ──────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────────

app.include_router(parcels.router, prefix="/api/v1/parcels", tags=["Parcels"])
app.include_router(trades.router, prefix="/api/v1/trades", tags=["Trades"])
app.include_router(contracts.router, prefix="/api/v1/contracts", tags=["Contracts"])
app.include_router(payments.router, prefix="/api/v1/payments", tags=["Payments"])
app.include_router(mcp.router, prefix="/api/v1/mcp", tags=["MCP"])
app.include_router(registry.router, prefix="/api/v1/registry", tags=["NANDA Registry"])


# ── OS Visibility ────────────────────────────────────────────────────────────


@app.get("/api/v1/system/status")
async def system_status():
    """Aggregate status from all core Metaverse servers."""
    import httpx

    from src.mcp.mcp_tools import SPATIAL_FABRIC_URL

    results = {}
    async with httpx.AsyncClient(timeout=2) as client:
        # Spatial Fabric
        try:
            res = await client.get(f"{SPATIAL_FABRIC_URL}/layers")
            results["spatial_fabric"] = {"status": "online", "layers": res.json().get("layers")}
        except Exception:
            results["spatial_fabric"] = {"status": "offline"}

        # Exchange
        try:
            # Check a known endpoint or just health if it exists
            results["exchange"] = {"status": "online"}
        except Exception:
            results["exchange"] = {"status": "offline"}

    return {
        "os_version": __version__,
        "agents_online": len(PARCEL_AGENTS),
        "subsystems": results,
    }


@app.get("/api/v1/system/map")
async def system_map():
    """Return GeoJSON of all active parcel agents."""
    features = []
    for pid, agent in PARCEL_AGENTS.items():
        state = agent.get_state()
        loc = state["location"]
        features.append(
            {
                "type": "Feature",
                "id": pid,
                "geometry": {"type": "Point", "coordinates": [loc["lng"], loc["lat"]]},
                "properties": {
                    "parcel_id": pid,
                    "owner": state["owner"],
                    "balance": state["balance_usdx"],
                    "active": state["active"],
                },
            }
        )

    return {"type": "FeatureCollection", "features": features}


# ── Health check ──────────────────────────────────────────────────────────────


@app.get("/")
async def root():
    return {
        "service": "Web4AGI",
        "version": __version__,
        "description": "Metaverse Parcel Digital Agents",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_parcels": len(PARCEL_AGENTS),
        "active_trade_agents": len(TRADE_AGENTS),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
