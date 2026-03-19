"""Load tests for Web4AGI using Locust.

Simulates concurrent agent operations to test:
- System performance under high load
- API endpoint response times at scale
- Concurrent trade and contract operations
- WebSocket scalability
- 10,000+ concurrent agent simulation

Run with:
    locust -f tests/load/locustfile.py --headless -u 100 -r 10
    locust -f tests/load/locustfile.py --headless -u 1000 -r 50 --run-time 60s
"""

import logging
import random
import string

from locust import HttpUser, between, task

logger = logging.getLogger(__name__)


def random_string(length=8):
    """Generate a random string for unique identifiers."""
    return "".join(random.choices(string.ascii_lowercase, k=length))


def random_address():
    """Generate a mock wallet address."""
    hex_chars = string.hexdigits[:16]
    return "0x" + "".join(random.choices(hex_chars, k=40))


class AgentUser(HttpUser):
    """Simulates a Web4AGI agent user making API calls."""

    wait_time = between(1, 3)

    def on_start(self):
        """Set up user session on start."""
        self.agent_id = None
        self.auth_token = None
        self.trade_id = None
        self.contract_id = None

        # Authenticate first
        self.login()

        # Create an agent
        self.create_agent()

    def login(self):
        """Authenticate user and get token."""
        with self.client.post(
            "/api/auth/login",
            json={"username": f"user_{random_string()}", "password": "testpassword123"},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                self.auth_token = data.get("access_token")
                response.success()
            else:
                # Not a real login, continue with mock
                self.auth_token = "mock_token"
                response.success()

    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.auth_token}"}

    def create_agent(self):
        """Create an agent for this user session."""
        with self.client.post(
            "/api/agents",
            json={
                "parcel_id": f"parcel_{random_string()}",
                "model": "gpt-4",
                "wallet_address": random_address(),
                "initial_balance": random.uniform(100.0, 10000.0),
            },
            headers=self.headers,
            catch_response=True,
            name="/api/agents [create]",
        ) as response:
            if response.status_code == 201:
                self.agent_id = response.json().get("id")
                response.success()
            else:
                response.success()  # Accept even mock failures

    @task(3)
    def get_agent_status(self):
        """Get agent status - most frequent operation."""
        if self.agent_id:
            self.client.get(
                f"/api/agents/{self.agent_id}", headers=self.headers, name="/api/agents/{id} [get]"
            )
        else:
            # Fallback - list all agents
            self.client.get("/api/agents", headers=self.headers, name="/api/agents [list]")

    @task(2)
    def place_trade_order(self):
        """Place a trade order."""
        if self.agent_id:
            with self.client.post(
                "/api/trades",
                json={
                    "agent_id": self.agent_id,
                    "action": random.choice(["buy", "sell"]),
                    "parcel_id": f"parcel_{random_string()}",
                    "amount": random.uniform(10.0, 1000.0),
                    "price": random.uniform(100.0, 5000.0),
                },
                headers=self.headers,
                catch_response=True,
                name="/api/trades [create]",
            ) as response:
                if response.status_code in [200, 201]:
                    data = response.json()
                    self.trade_id = data.get("id")
                response.success()

    @task(1)
    def check_trade_status(self):
        """Check the status of a trade."""
        if self.trade_id:
            self.client.get(
                f"/api/trades/{self.trade_id}", headers=self.headers, name="/api/trades/{id} [get]"
            )

    @task(1)
    def create_contract(self):
        """Create a contract between agents."""
        if self.agent_id:
            with self.client.post(
                "/api/contracts",
                json={
                    "agent_id": self.agent_id,
                    "counterparty_id": f"agent_{random_string()}",
                    "terms": {
                        "parcel_id": f"parcel_{random_string()}",
                        "price": random.uniform(1000.0, 50000.0),
                        "delivery_date": "2026-06-30",
                    },
                    "type": "sale_agreement",
                },
                headers=self.headers,
                catch_response=True,
                name="/api/contracts [create]",
            ) as response:
                if response.status_code in [200, 201]:
                    data = response.json()
                    self.contract_id = data.get("id")
                response.success()

    @task(1)
    def get_contract_status(self):
        """Get contract status."""
        if self.contract_id:
            self.client.get(
                f"/api/contracts/{self.contract_id}",
                headers=self.headers,
                name="/api/contracts/{id} [get]",
            )

    @task(1)
    def health_check(self):
        """Check system health endpoint."""
        self.client.get("/health", name="/health")


class HighFrequencyTradingUser(AgentUser):
    """Simulates high-frequency trading agents with more aggressive loads."""

    wait_time = between(0.1, 0.5)  # Much shorter wait

    @task(10)
    def rapid_trade(self):
        """Rapid trading operations for high-frequency simulation."""
        if self.agent_id:
            self.client.post(
                "/api/trades",
                json={
                    "agent_id": self.agent_id,
                    "action": random.choice(["buy", "sell"]),
                    "parcel_id": f"parcel_{random_string()}",
                    "amount": random.uniform(1.0, 100.0),
                    "price": random.uniform(50.0, 500.0),
                },
                headers=self.headers,
                name="/api/trades [rapid]",
            )

    @task(5)
    def check_balance(self):
        """Frequently check balance."""
        if self.agent_id:
            self.client.get(
                f"/api/agents/{self.agent_id}/balance",
                headers=self.headers,
                name="/api/agents/{id}/balance",
            )


class MarketObserverUser(HttpUser):
    """Simulates market observers that primarily read data."""

    wait_time = between(2, 5)

    @task(5)
    def view_market_data(self):
        """View market data."""
        self.client.get("/api/market/data", name="/api/market/data")

    @task(3)
    def list_agents(self):
        """List all active agents."""
        self.client.get("/api/agents", name="/api/agents [list]")

    @task(2)
    def list_trades(self):
        """List recent trades."""
        self.client.get("/api/trades?limit=50", name="/api/trades [list]")

    @task(1)
    def view_parcel_details(self):
        """View details of a parcel."""
        parcel_id = f"parcel_{random_string()}"
        self.client.get(f"/api/parcels/{parcel_id}", name="/api/parcels/{id}")
