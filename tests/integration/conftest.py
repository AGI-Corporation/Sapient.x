"""Integration test conftest — minimal FastAPI test application.

Provides a test_app fixture with all routes expected by test_api.py.
"""
import uuid
from typing import Any, Dict

import pytest
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.testclient import TestClient

# ── In-memory stores ─────────────────────────────────────────────────────────

_agents: Dict[str, Dict[str, Any]] = {}
_trades: Dict[str, Dict[str, Any]] = {}
_contracts: Dict[str, Dict[str, Any]] = {}

# ── Auth helpers ─────────────────────────────────────────────────────────────

_VALID_USERS = {"testuser": "testpass123"}
_VALID_TOKEN = "valid_token_123"


def _get_current_user(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    token = authorization.removeprefix("Bearer ").strip()
    if token != _VALID_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


# ── App factory ───────────────────────────────────────────────────────────────

def build_test_app() -> FastAPI:
    app = FastAPI()

    # ── Agents ────────────────────────────────────────────────────────────────

    @app.post("/api/agents", status_code=201)
    def create_agent(body: Dict[str, Any]):
        agent_id = f"agent_{uuid.uuid4().hex[:6]}"
        record = {"id": agent_id, "parcel_id": body.get("parcel_id", ""), **body}
        _agents[agent_id] = record
        return record

    @app.get("/api/agents", dependencies=[Depends(_get_current_user)])
    def list_agents():
        return list(_agents.values())

    @app.get("/api/agents/{agent_id}")
    def get_agent(agent_id: str):
        if agent_id not in _agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id!r} not found")
        return _agents[agent_id]

    @app.patch("/api/agents/{agent_id}")
    def update_agent(agent_id: str, body: Dict[str, Any]):
        if agent_id not in _agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id!r} not found")
        _agents[agent_id].update(body)
        return _agents[agent_id]

    @app.delete("/api/agents/{agent_id}", status_code=204)
    def delete_agent(agent_id: str):
        _agents.pop(agent_id, None)

    # ── Trades ────────────────────────────────────────────────────────────────

    @app.post("/api/trades", status_code=201)
    def create_trade(body: Dict[str, Any]):
        amount = body.get("amount", 0)
        if amount > 100000:
            raise HTTPException(status_code=400, detail="Insufficient balance for this trade")
        trade_id = f"trade_{uuid.uuid4().hex[:6]}"
        record = {"id": trade_id, "status": "pending", **body}
        _trades[trade_id] = record
        return record

    @app.get("/api/trades/{trade_id}")
    def get_trade(trade_id: str):
        if trade_id not in _trades:
            raise HTTPException(status_code=404, detail=f"Trade {trade_id!r} not found")
        return _trades[trade_id]

    @app.post("/api/trades/{trade_id}/cancel")
    def cancel_trade(trade_id: str):
        if trade_id not in _trades:
            raise HTTPException(status_code=404, detail=f"Trade {trade_id!r} not found")
        _trades[trade_id]["status"] = "cancelled"
        return _trades[trade_id]

    # ── Contracts ─────────────────────────────────────────────────────────────

    @app.post("/api/contracts", status_code=201)
    def create_contract(body: Dict[str, Any]):
        contract_id = f"contract_{uuid.uuid4().hex[:6]}"
        record = {"id": contract_id, "status": "pending", "signatures": [], **body}
        _contracts[contract_id] = record
        return record

    @app.get("/api/contracts/{contract_id}")
    def get_contract(contract_id: str):
        if contract_id not in _contracts:
            raise HTTPException(status_code=404, detail=f"Contract {contract_id!r} not found")
        return _contracts[contract_id]

    @app.post("/api/contracts/{contract_id}/sign")
    def sign_contract(contract_id: str, body: Dict[str, Any]):
        if contract_id not in _contracts:
            raise HTTPException(status_code=404, detail=f"Contract {contract_id!r} not found")
        _contracts[contract_id]["signatures"].append(body)
        return _contracts[contract_id]

    @app.post("/api/contracts/{contract_id}/execute")
    def execute_contract(contract_id: str):
        if contract_id not in _contracts:
            raise HTTPException(status_code=404, detail=f"Contract {contract_id!r} not found")
        _contracts[contract_id]["status"] = "executed"
        return _contracts[contract_id]

    # ── Auth ──────────────────────────────────────────────────────────────────

    @app.post("/api/auth/login")
    def login(body: Dict[str, Any]):
        username = body.get("username", "")
        password = body.get("password", "")
        if _VALID_USERS.get(username) != password:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {"access_token": _VALID_TOKEN, "token_type": "bearer"}

    return app


@pytest.fixture(scope="module")
def test_app():
    return build_test_app()


@pytest.fixture(scope="module")
def api_client(test_app):
    return TestClient(test_app)
