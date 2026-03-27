"""Payment and wallet operations for Web4AGI.

This package provides x402 protocol integration for wallet management,
USDC/USDT transactions, and smart contract interactions.
"""
from src.payments.x402_client import X402Client

__all__ = ["X402Client"]
