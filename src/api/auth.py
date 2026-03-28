"""Auth API router — Web4AGI (/api/auth)."""

import secrets
from typing import Any, Dict
from fastapi import APIRouter, HTTPException

router = APIRouter()

# Simple credential store (demo only — not for production)
_VALID_CREDENTIALS = {
    "testuser": "testpass123",
    "admin": "admin123",
}

_ISSUED_TOKENS: Dict[str, str] = {}  # token → username


@router.post("/login")
async def login(body: Dict[str, Any]) -> Any:
    username = body.get("username", "")
    password = body.get("password", "")
    expected = _VALID_CREDENTIALS.get(username)
    if expected is None or expected != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = secrets.token_hex(32)
    _ISSUED_TOKENS[token] = username
    return {"access_token": token, "token_type": "bearer", "username": username}


@router.post("/logout")
async def logout(body: Dict[str, Any]) -> Any:
    token = body.get("token", "")
    _ISSUED_TOKENS.pop(token, None)
    return {"success": True}


def verify_token(authorization: str) -> str:
    """Verify a Bearer token and return the username, or raise 401."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ", 1)[1]
    username = _ISSUED_TOKENS.get(token)
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return username
