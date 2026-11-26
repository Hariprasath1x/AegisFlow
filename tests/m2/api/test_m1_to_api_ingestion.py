"""API tests verifying M1-style transaction generation through POST /api/v1/transactions."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import fakeredis
from fastapi.testclient import TestClient

from ingestion import config
from ingestion.main import app
from ingestion.persistence import MemoryStore
from scripts.ingest_transactions import (
    build_m1_api_transactions,
    map_m1_channel_to_api,
)


class _TestClientHttpBridge:
    """Test-only HTTP bridge forwarding urllib requests to FastAPI TestClient."""

    def __init__(self, client: TestClient) -> None:
        self._client = client

    def __call__(self, req: Any, timeout: float = 10.0) -> Any:
        from urllib.parse import urlsplit

        path = urlsplit(req.full_url).path
        headers = {k: v for k, v in req.header_items()}
        if req.get_method() == "GET":
            resp = self._client.get(path, headers=headers)
        else:
            resp = self._client.post(path, content=req.data, headers=headers)

        class _FakeHTTPResponse:
            def __init__(self, status_code: int, raw: bytes) -> None:
                self.status = status_code
                self._raw = raw

            def read(self) -> bytes:
                return self._raw

            def __enter__(self) -> "_FakeHTTPResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

        return _FakeHTTPResponse(resp.status_code, resp.content)


def test_map_m1_channel_to_api_valid_and_invalid() -> None:
    assert map_m1_channel_to_api("card") == "card"
    assert map_m1_channel_to_api("atm") == "card"
    assert map_m1_channel_to_api("wire") == "transfer"
    assert map_m1_channel_to_api("ach") == "transfer"
    assert map_m1_channel_to_api("upi") == "wallet"

    try:
        map_m1_channel_to_api("crypto")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unsupported channel")


def test_build_m1_api_transactions_is_monotonic_and_transfer_safe() -> None:
    entities = [
        {"account_id": uuid4(), "account_number": f"AF-M1-DEMO-ACCOUNT-{i:04d}", "counterparty_id": uuid4()}
        for i in range(6)
    ]
    ref_end = datetime(2026, 11, 26, 5, 0, 0, tzinfo=timezone.utc)
    txns = build_m1_api_transactions(entities, count=30, end_time=ref_end, window_minutes=30)

    assert len(txns) == 30
    prev_keys_by_acct: dict[str, tuple[str, str]] = {}
    channels_seen: set[str] = set()

    account_ids = {str(e["account_id"]) for e in entities}
    for row in txns:
        UUID(row["id"])
        assert isinstance(row["amount"], str)
        assert Decimal(row["amount"]) > 0
        assert row["channel"] in {"card", "transfer", "wallet"}
        channels_seen.add(row["channel"])
        if row["channel"] == "transfer":
            assert row["counterparty_id"] in account_ids
            assert row["counterparty_id"] != row["account_id"]

        order_key = (row["ts"], row["id"])
        acct = row["account_id"]
        if acct in prev_keys_by_acct:
            assert order_key > prev_keys_by_acct[acct]
        prev_keys_by_acct[acct] = order_key

    assert channels_seen == {"card", "transfer", "wallet"}


def test_m1_stream_end_to_end_persists_db_edges_and_redis_velocity() -> None:
    from scripts.ingest_transactions import submit_transactions

    entities = [
        {"account_id": uuid4(), "account_number": f"AF-M1-DEMO-ACCOUNT-{i:04d}", "counterparty_id": uuid4()}
        for i in range(5)
    ]
    txns = build_m1_api_transactions(
        entities,
        count=15,
        end_time=datetime.now(timezone.utc),
        window_minutes=20,
        batch_id="test-m1-api-batch",
    )

    store = MemoryStore()
    fake_redis = fakeredis.FakeRedis()
    client = TestClient(app)

    bridge = _TestClientHttpBridge(client)
    with patch("ingestion.deps.store", return_value=store), patch(
        "ingestion.deps.redis_client", return_value=fake_redis
    ), patch("urllib.request.urlopen", side_effect=bridge):
        first = submit_transactions(
            txns,
            api_url="http://localhost:8001",
            service_key=config.SERVICE_KEY,
        )
        replay = submit_transactions(
            txns,
            api_url="http://localhost:8001",
            service_key=config.SERVICE_KEY,
        )

    assert first.submitted == 15
    assert first.accepted_new == 15
    assert first.accepted_duplicate == 0
    assert first.rejected == 0
    assert all(store.get(UUID(row["id"])) is not None for row in txns)

    assert replay.submitted == 15
    assert replay.accepted_new == 0
    assert replay.accepted_duplicate == 15
    assert replay.rejected == 0

    vel_1h_keys = list(fake_redis.scan_iter("vel:acct:*:1h"))
    feat_keys = list(fake_redis.scan_iter("feat:acct:*"))
    assert len(vel_1h_keys) == 5
    assert len(feat_keys) == 5

    # Verify each account's 1h velocity ZSET holds all 3 generated events (15 / 5 accounts).
    for key in vel_1h_keys:
        assert fake_redis.zcard(key) == 3

    sample_feat = fake_redis.hgetall(feat_keys[0])
    assert int(sample_feat[b"w_count"]) == 3
    assert json.loads(json.dumps(txns[0]))["channel"] in {"card", "transfer", "wallet"}
