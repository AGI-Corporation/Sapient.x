"""FastAPI Main Application — Web4AGI.

Run with: uvicorn src.main:app --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import parcels, trades, contracts, payments, mcp
from src.api import agents as agents_router
from src.api import auth as auth_router
from src.api import trades_v2, contracts_v2
from src import __version__


# ── Global state ──────────────────────────────────────────────────────────────

PARCEL_AGENTS = {}  # parcel_id -> ParcelAgent instance
TRADE_AGENTS = {}   # agent_id -> TradeAgent instance


@asynccontextmanager
async def lifespan(app: FastAPI):
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

# Alternative API paths used by integration tests
app.include_router(agents_router.router, prefix="/api/agents", tags=["Agents"])
app.include_router(auth_router.router, prefix="/api/auth", tags=["Auth"])
app.include_router(trades_v2.router, prefix="/api/trades", tags=["Trades v2"])
app.include_router(contracts_v2.router, prefix="/api/contracts", tags=["Contracts v2"])


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
