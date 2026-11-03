"""Unit and integration tests for FastAPI ingestion endpoint, authentication, and lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import sys
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient
import redis
import fakeredis

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ML_SRC = _REPO_ROOT / "ml" / "src"
_INGESTION_SRC = _REPO_ROOT / "services" / "ingestion" / "src"
for p in (_ML_SRC, _INGESTION_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ingestion import config, deps
from ingestion.main import app
from ingestion.persistence import MemoryStore


@pytest.fixture
def client_env():
    """Configure TestClient with isolated MemoryStore and FakeRedis."""
    known_account = uuid4()
    known_counterparty = uuid4()
    mem_store = MemoryStore(known_accounts=[known_account, known_counterparty])
    fake_redis = fakeredis.FakeRedis()

    deps._store = mem_store
    deps._redis = fake_redis

    test_client = TestClient(app)
    try:
        yield test_client, mem_store, fake_redis, known_account, known_counterparty
    finally:
        deps.reset_deps()
        fake_redis.flushdb()


# ============================================================================
# 1. Authentication Tests
# ============================================================================


def test_missing_service_key_returns_401(client_env):
    """Verify request without X-Service-Key is rejected with 401."""
    client, _, _, acc_id, _ = client_env
    payload = {
        "id": str(uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "account_id": str(acc_id),
        "amount": "100.00",
        "currency": "USD",
        "channel": "card",
    }
    res = client.post("/api/v1/transactions", json=payload)
    assert res.status_code == 401
    assert "Invalid or missing X-Service-Key" in res.json()["detail"]


def test_invalid_service_key_returns_401(client_env):
    """Verify request with wrong X-Service-Key is rejected with 401."""
    client, _, _, acc_id, _ = client_env
    payload = {
        "id": str(uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "account_id": str(acc_id),
        "amount": "100.00",
        "currency": "USD",
        "channel": "card",
    }
    headers = {"X-Service-Key": "wrong-secret-key"}
    res = client.post("/api/v1/transactions", json=payload, headers=headers)
    assert res.status_code == 401
    assert "Invalid or missing X-Service-Key" in res.json()["detail"]


# ============================================================================
# 2. Validation & Unknown Entity Tests
# ============================================================================


def test_invalid_request_body_returns_422_before_persistence(client_env):
    """Verify malformed payload (e.g. float amount) returns 422 before persistence."""
    client, store, _, acc_id, _ = client_env
    headers = {"X-Service-Key": config.SERVICE_KEY}
    payload = {
        "id": str(uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "account_id": str(acc_id),
        "amount": 100.50,  # Invalid JSON float!
        "currency": "USD",
        "channel": "card",
    }
    res = client.post("/api/v1/transactions", json=payload, headers=headers)
    assert res.status_code == 422
    assert len(store.transactions) == 0


def test_unknown_entity_returns_422_and_does_not_call_redis(client_env):
    """Verify unknown account raises 422 and does not execute Redis Lua update."""
    client, store, r, _, _ = client_env
    unknown_account = uuid4()
    txn_id = uuid4()
    headers = {"X-Service-Key": config.SERVICE_KEY}

    payload = {
        "id": str(txn_id),
        "ts": datetime.now(timezone.utc).isoformat(),
        "account_id": str(unknown_account),
        "amount": "50.00",
        "currency": "USD",
        "channel": "card",
    }
    res = client.post("/api/v1/transactions", json=payload, headers=headers)
    assert res.status_code == 422
    assert "Unknown entity" in res.json()["detail"]
    assert len(store.transactions) == 0
    # Redis seen key must NOT have been written
    assert r.exists(f"seen:txn:{txn_id}") == 0


# ============================================================================
# 3. Successful Ingestion & Duplicate Tests
# ============================================================================


def test_newly_inserted_transaction_calls_redis_and_returns_201(client_env):
    """Verify new transaction is stored, updates Redis, and returns HTTP 201."""
    client, store, r, acc_id, _ = client_env
    txn_id = uuid4()
    headers = {"X-Service-Key": config.SERVICE_KEY}
    payload = {
        "id": str(txn_id),
        "ts": datetime.now(timezone.utc).isoformat(),
        "account_id": str(acc_id),
        "device_id": str(uuid4()),
        "ip_id": str(uuid4()),
        "amount": "125.50",
        "currency": "USD",
        "channel": "card",
    }
    res = client.post("/api/v1/transactions", json=payload, headers=headers)
    assert res.status_code == 201
    data = res.json()
    assert data["id"] == str(txn_id)
    assert data["duplicate"] is False
    assert data["decision"] is None

    # Check state in store and Redis
    assert txn_id in store.transactions
    assert r.exists(f"seen:txn:{txn_id}") == 1


def test_duplicate_transaction_still_calls_redis_and_returns_200(client_env):
    """Verify duplicate transaction returns HTTP 200, duplicate=True, and calls Redis Lua."""
    client, store, r, acc_id, _ = client_env
    txn_id = uuid4()
    headers = {"X-Service-Key": config.SERVICE_KEY}
    payload = {
        "id": str(txn_id),
        "ts": datetime.now(timezone.utc).isoformat(),
        "account_id": str(acc_id),
        "amount": "100.00",
        "currency": "USD",
        "channel": "card",
    }

    # First call: 201 Created
    res1 = client.post("/api/v1/transactions", json=payload, headers=headers)
    assert res1.status_code == 201
    assert res1.json()["duplicate"] is False

    # Second call (same payload): 200 OK, duplicate=True
    res2 = client.post("/api/v1/transactions", json=payload, headers=headers)
    assert res2.status_code == 200
    assert res2.json()["duplicate"] is True
    assert res2.json()["decision"] is None


# ============================================================================
# 4. Critical Retry & Idempotency Sequence
# ============================================================================


class FailingRedis(fakeredis.FakeRedis):
    """Fakeredis subclass that simulates a failure on command execution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.should_fail: bool = False

    def eval(self, *args, **kwargs):
        if self.should_fail:
            raise redis.ConnectionError("Simulated Redis network disconnection")
        return super().eval(*args, **kwargs)

    def evalsha(self, *args, **kwargs):
        if self.should_fail:
            raise redis.ConnectionError("Simulated Redis network disconnection")
        return super().evalsha(*args, **kwargs)


def test_critical_retry_and_idempotency_lifecycle(client_env):
    """Simulate exact failure and retry lifecycle:

    1. First request: PostgreSQL commits transaction; Redis raises ConnectionError.
    2. Endpoint returns 503 (features_pending).
    3. Second request: retry with exact same transaction payload.
    4. PostgreSQL detects duplicate (does not insert new row, does not re-increment edges).
    5. Redis update succeeds.
    6. Endpoint returns 200 duplicate=True.
    """
    client, store, _, acc_id, _ = client_env
    failing_redis = FailingRedis()
    deps._redis = failing_redis

    txn_id = uuid4()
    dev_id = uuid4()
    headers = {"X-Service-Key": config.SERVICE_KEY}
    payload = {
        "id": str(txn_id),
        "ts": datetime.now(timezone.utc).isoformat(),
        "account_id": str(acc_id),
        "device_id": str(dev_id),
        "amount": "250.00",
        "currency": "USD",
        "channel": "card",
    }

    # Step 1: Redis is failing
    failing_redis.should_fail = True
    res1 = client.post("/api/v1/transactions", json=payload, headers=headers)
    assert res1.status_code == 503
    assert "online features pending" in res1.json()["detail"].lower()

    # Verify transaction was persisted in database and edge weight is 1.0
    assert txn_id in store.transactions
    edge_key = ("account", str(acc_id), "device", str(dev_id), "uses_device")
    assert store.edges[edge_key] == 1.0

    # Step 2: Caller retries exact same request after Redis is restored
    failing_redis.should_fail = False
    res2 = client.post("/api/v1/transactions", json=payload, headers=headers)
    assert res2.status_code == 200
    assert res2.json()["duplicate"] is True
    assert res2.json()["id"] == str(txn_id)

    # Verify database state was preserved (still only 1 transaction, edge weight still 1.0)
    assert len(store.transactions) == 1
    assert store.edges[edge_key] == 1.0

    # Verify Redis features were applied
    assert failing_redis.exists(f"seen:txn:{txn_id}") == 1


def test_duplicate_with_conflicting_amount_returns_409(client_env):
    """Verify duplicate transaction ID with different amount returns 409 and does not call Redis."""
    client, store, r, acc_id, _ = client_env
    txn_id = uuid4()
    dev_id = uuid4()
    headers = {"X-Service-Key": config.SERVICE_KEY}
    payload = {
        "id": str(txn_id),
        "ts": datetime.now(timezone.utc).isoformat(),
        "account_id": str(acc_id),
        "device_id": str(dev_id),
        "amount": "100.00",
        "currency": "USD",
        "channel": "card",
    }

    # 1. First request creates the transaction
    res1 = client.post("/api/v1/transactions", json=payload, headers=headers)
    assert res1.status_code == 201
    assert r.exists(f"seen:txn:{txn_id}") == 1

    # Clear seen key to verify whether Redis is called on conflicting retry
    r.delete(f"seen:txn:{txn_id}")
    edge_key = ("account", str(acc_id), "device", str(dev_id), "uses_device")
    assert store.edges[edge_key] == 1.0

    # 2. Duplicate ID with conflicting amount
    conflicting_payload = dict(payload, amount="999.00")
    res2 = client.post("/api/v1/transactions", json=conflicting_payload, headers=headers)
    assert res2.status_code == 409
    assert "conflicting payload" in res2.json()["detail"].lower()

    # Verify Redis was NOT called for the conflicting transaction
    assert r.exists(f"seen:txn:{txn_id}") == 0

    # Verify stored transaction amount and edge weights are unchanged
    stored_t = store.get(txn_id)
    assert stored_t is not None
    assert stored_t.amount == Decimal("100.00")
    assert store.edges[edge_key] == 1.0


def test_duplicate_with_conflicting_account_returns_409(client_env):
    """Verify duplicate transaction ID with different account returns 409."""
    client, store, r, acc_id, other_acc_id = client_env
    txn_id = uuid4()
    headers = {"X-Service-Key": config.SERVICE_KEY}
    payload = {
        "id": str(txn_id),
        "ts": datetime.now(timezone.utc).isoformat(),
        "account_id": str(acc_id),
        "amount": "100.00",
        "currency": "USD",
        "channel": "card",
    }

    res1 = client.post("/api/v1/transactions", json=payload, headers=headers)
    assert res1.status_code == 201

    r.delete(f"seen:txn:{txn_id}")

    # Retry with different account ID
    conflicting_payload = dict(payload, account_id=str(other_acc_id))
    res2 = client.post("/api/v1/transactions", json=conflicting_payload, headers=headers)
    assert res2.status_code == 409
    assert "conflicting payload" in res2.json()["detail"].lower()
    assert r.exists(f"seen:txn:{txn_id}") == 0
    assert store.get(txn_id).account_id == acc_id


def test_duplicate_with_conflicting_timestamp_returns_409(client_env):
    """Verify duplicate transaction ID with different timestamp returns 409."""
    client, store, r, acc_id, _ = client_env
    txn_id = uuid4()
    headers = {"X-Service-Key": config.SERVICE_KEY}
    dt1 = datetime(2026, 10, 31, 10, 0, 0, tzinfo=timezone.utc)
    dt2 = datetime(2026, 10, 31, 10, 5, 0, tzinfo=timezone.utc)

    payload = {
        "id": str(txn_id),
        "ts": dt1.isoformat(),
        "account_id": str(acc_id),
        "amount": "100.00",
        "currency": "USD",
        "channel": "card",
    }

    res1 = client.post("/api/v1/transactions", json=payload, headers=headers)
    assert res1.status_code == 201

    r.delete(f"seen:txn:{txn_id}")

    # Retry with different timestamp
    conflicting_payload = dict(payload, ts=dt2.isoformat())
    res2 = client.post("/api/v1/transactions", json=conflicting_payload, headers=headers)
    assert res2.status_code == 409
    assert "conflicting payload" in res2.json()["detail"].lower()
    assert r.exists(f"seen:txn:{txn_id}") == 0


def test_failure_retry_with_conflicting_and_identical_payloads(client_env):
    """Verify sequence:
    1. First request: PostgreSQL inserts transaction, Redis fails -> 503.
    2. Conflicting retry: returns 409, does NOT call Redis.
    3. Identical retry: returns 200, Redis called and succeeds.
    4. Database state has 1 row and 1 edge weight.
    """
    client, store, _, acc_id, _ = client_env
    failing_redis = FailingRedis()
    deps._redis = failing_redis

    txn_id = uuid4()
    dev_id = uuid4()
    headers = {"X-Service-Key": config.SERVICE_KEY}
    payload = {
        "id": str(txn_id),
        "ts": datetime.now(timezone.utc).isoformat(),
        "account_id": str(acc_id),
        "device_id": str(dev_id),
        "amount": "300.00",
        "currency": "USD",
        "channel": "card",
    }

    # Step 1: Redis fails on initial insert
    failing_redis.should_fail = True
    res1 = client.post("/api/v1/transactions", json=payload, headers=headers)
    assert res1.status_code == 503

    edge_key = ("account", str(acc_id), "device", str(dev_id), "uses_device")
    assert store.edges[edge_key] == 1.0

    # Step 2: Conflicting retry with different amount
    conflicting_payload = dict(payload, amount="400.00")
    res_conflict = client.post("/api/v1/transactions", json=conflicting_payload, headers=headers)
    assert res_conflict.status_code == 409

    # Step 3: Identical retry after Redis is healthy
    failing_redis.should_fail = False
    res_retry = client.post("/api/v1/transactions", json=payload, headers=headers)
    assert res_retry.status_code == 200
    assert res_retry.json()["duplicate"] is True

    # Step 4: Verify single transaction row and edge weight is still 1.0
    assert len(store.transactions) == 1
    assert store.edges[edge_key] == 1.0
    assert store.get(txn_id).amount == Decimal("300.00")
    assert failing_redis.exists(f"seen:txn:{txn_id}") == 1


# ============================================================================
# 5. Observability Endpoints: /health, /ready, /metrics
# ============================================================================


def test_health_endpoint(client_env):
    """Verify /health returns 200 {"status": "ok"}."""
    client, _, _, _, _ = client_env
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_ready_endpoint_healthy(client_env):
    """Verify /ready returns 200 {"status": "ready"} when store and Redis are healthy."""
    client, _, _, _, _ = client_env
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json() == {"status": "ready"}


def test_ready_endpoint_unhealthy_when_store_down(client_env):
    """Verify /ready returns 503 when database is unavailable."""
    client, store, _, _, _ = client_env
    store.set_healthy(False)
    res = client.get("/ready")
    assert res.status_code == 503
    data = res.json()
    assert data["status"] == "not_ready"
    assert data["postgres"] == "unavailable"


def test_ready_endpoint_unhealthy_when_redis_down(client_env):
    """Verify /ready returns 503 when Redis is unavailable."""
    client, _, _, _, _ = client_env
    # Replace Redis with broken client
    broken_redis = fakeredis.FakeRedis()
    broken_redis.ping = lambda: (_ for _ in ()).throw(redis.ConnectionError("Redis down"))
    deps._redis = broken_redis

    res = client.get("/ready")
    assert res.status_code == 503
    data = res.json()
    assert data["status"] == "not_ready"
    assert data["redis"] == "unavailable"


def test_metrics_endpoint(client_env):
    """Verify /metrics returns Prometheus exposition format with expected metrics."""
    client, _, _, _, _ = client_env
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "text/plain" in res.headers["content-type"]
    content = res.text
    assert "af_transactions_total" in content
    assert "af_online_feature_seconds" in content
