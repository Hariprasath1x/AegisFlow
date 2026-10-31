"""Unit tests for online feature computation via Redis and Lua.

Tests the atomic Lua script, argument and key construction, idempotency guard,
velocity sliding windows, distinct counts, sharing counts, Welford's algorithm,
calendar features, graph features, event-time ordering safeguards, and canonical
19-feature output ordering.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal
import math
import os
from pathlib import Path
import sys
from typing import Generator
import pytest
import redis
import fakeredis

# Ensure ml/src and services/ingestion/src are in sys.path
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ML_SRC = _REPO_ROOT / "ml" / "src"
_INGESTION_SRC = _REPO_ROOT / "services" / "ingestion" / "src"
if str(_ML_SRC) not in sys.path:
    sys.path.insert(0, str(_ML_SRC))
if str(_INGESTION_SRC) not in sys.path:
    sys.path.insert(0, str(_INGESTION_SRC))

from aegis_ml.features import definitions as D
import ingestion.online_features as OF


@pytest.fixture
def redis_client() -> Generator[redis.Redis | fakeredis.FakeRedis, None, None]:
    """Provide clean Redis client (real Redis if running on REDIS_URL, else FakeRedis)."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    client: redis.Redis | fakeredis.FakeRedis
    try:
        real_client = redis.Redis.from_url(redis_url, decode_responses=False)
        real_client.ping()
        client = real_client
    except Exception:
        client = fakeredis.FakeRedis()

    client.flushdb()
    try:
        yield client
    finally:
        client.flushdb()


# ============================================================================
# 1. Key and Argument Construction Contracts
# ============================================================================


def test_keys_for_contract():
    """Verify keys_for returns exactly 10 keys in canonical order with placeholders."""
    t = {
        "txn_id": "txn_100",
        "account_id": "acc_200",
        "device_id": "dev_300",
        "ip_address": "10.0.0.1",
    }
    keys = OF.keys_for(t)
    assert len(keys) == 10
    assert keys == [
        "seen:txn:txn_100",
        "feat:acct:acc_200",
        "vel:acct:acc_200:1m",
        "vel:acct:acc_200:1h",
        "vel:acct:acc_200:24h",
        "vel:acct:acc_200:7d",
        "dist:acct:acc_200:dev",
        "dist:acct:acc_200:ip",
        "share:dev:dev_300",
        "share:ip:10.0.0.1",
    ]

    # Without device and IP -> placeholder keys
    t_sparse = {
        "id": "txn_101",
        "account_id": "acc_201",
        "device_id": None,
        "ip": "",
    }
    keys_sparse = OF.keys_for(t_sparse)
    assert len(keys_sparse) == 10
    assert keys_sparse[8] == "share:dev:-"
    assert keys_sparse[9] == "share:ip:-"

    # Validation errors on missing required keys
    with pytest.raises(ValueError, match="account_id"):
        OF.keys_for({"txn_id": "txn_102"})

    with pytest.raises(ValueError, match="one of"):
        OF.keys_for({"account_id": "acc_202"})


def test_args_for_contract():
    """Verify args_for returns exactly 15 arguments with proper types and constants."""
    dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    t = {
        "txn_id": "txn_100",
        "account_id": "acc_200",
        "timestamp": dt,
        "amount": Decimal("125.50"),
        "device_id": "dev_300",
        "ip_address": "192.168.1.1",
    }
    args = OF.args_for(t)
    assert len(args) == 15
    assert args[0] == "txn_100"
    assert args[1] == f"{D.epoch_seconds(dt):.6f}"
    assert args[2] == 12550
    assert args[3] == 1  # has_device
    assert args[4] == 1  # has_ip
    assert args[5] == "acc_200"
    assert args[6] == "dev_300"
    assert args[7] == "192.168.1.1"
    assert args[8] == D.KEY_TTL_S
    assert args[9] == D.WINDOWS["1m"]
    assert args[10] == D.WINDOWS["1h"]
    assert args[11] == D.WINDOWS["24h"]
    assert args[12] == D.WINDOWS["7d"]
    assert args[13] == D.DISTINCT_WINDOW_S
    assert args[14] == D.SHARE_WINDOW_S

    # Sparse inputs (no device / IP, amount_minor provided directly)
    t_sparse = {
        "id": "txn_101",
        "account_id": "acc_201",
        "ts": 1700000000.0,
        "amount_minor": 5000,
    }
    args_sparse = OF.args_for(t_sparse)
    assert len(args_sparse) == 15
    assert args_sparse[0] == "txn_101"
    assert args_sparse[1] == "1700000000.000000"
    assert args_sparse[2] == 5000
    assert args_sparse[3] == 0  # has_device
    assert args_sparse[4] == 0  # has_ip
    assert args_sparse[6] == ""
    assert args_sparse[7] == ""


# ============================================================================
# 2. Idempotency Guard
# ============================================================================


def test_idempotency_guard(redis_client):
    """Verify duplicate transactions return None and do not double-count."""
    t = {
        "txn_id": "txn_idemp_1",
        "account_id": "acc_idemp_1",
        "timestamp": "2024-01-01T10:00:00Z",
        "amount": 100.00,
        "device_id": "dev_idemp_1",
        "ip_address": "10.0.0.1",
    }
    res1 = OF.update(redis_client, t)
    assert res1 is not None
    assert res1["txn_count_1m"] == 1.0
    assert res1["amount_sum_minor_1m"] == 10000.0

    # Repeat exact same transaction
    res2 = OF.update(redis_client, t)
    assert res2 is None, "Duplicate transaction must return None"

    # Verify Redis state was not mutated by the duplicate
    vel_key = f"vel:acct:{t['account_id']}:1m"
    assert redis_client.zcard(vel_key) == 1
    assert redis_client.exists(f"seen:txn:{t['txn_id']}") == 1


# ============================================================================
# 3. Velocity Sliding Windows & Expiration
# ============================================================================


def test_velocity_sliding_windows_and_expiration(redis_client):
    """Verify velocity features count and sum across 1m, 1h, 24h, 7d windows."""
    base_t = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    acc = "acc_vel_1"

    # t1 at T
    t1 = {
        "txn_id": "txn_v1",
        "account_id": acc,
        "timestamp": base_t,
        "amount_minor": 1000,
    }
    r1 = OF.update(redis_client, t1)
    assert r1["txn_count_1m"] == 1.0
    assert r1["amount_sum_minor_1m"] == 1000.0
    assert r1["txn_count_7d"] == 1.0
    assert r1["amount_sum_minor_7d"] == 1000.0

    # t2 at T + 30s (inside 1m)
    t2 = {
        "txn_id": "txn_v2",
        "account_id": acc,
        "timestamp": base_t + timedelta(seconds=30),
        "amount_minor": 2000,
    }
    r2 = OF.update(redis_client, t2)
    assert r2["txn_count_1m"] == 2.0
    assert r2["amount_sum_minor_1m"] == 3000.0
    assert r2["txn_count_1h"] == 2.0
    assert r2["amount_sum_minor_1h"] == 3000.0

    # t3 at T + 65s (t1 expired from 1m, inside 1h)
    t3 = {
        "txn_id": "txn_v3",
        "account_id": acc,
        "timestamp": base_t + timedelta(seconds=65),
        "amount_minor": 3000,
    }
    r3 = OF.update(redis_client, t3)
    assert r3["txn_count_1m"] == 2.0  # t2 and t3
    assert r3["amount_sum_minor_1m"] == 5000.0  # 2000 + 3000 (1000 expired)
    assert r3["txn_count_1h"] == 3.0  # t1, t2, t3
    assert r3["amount_sum_minor_1h"] == 6000.0  # 1000 + 2000 + 3000

    # t4 at T + 3700s (> 1h + 65s: t1, t2, t3 expired from 1h, inside 24h)
    t4 = {
        "txn_id": "txn_v4",
        "account_id": acc,
        "timestamp": base_t + timedelta(seconds=3700),
        "amount_minor": 4000,
    }
    r4 = OF.update(redis_client, t4)
    assert r4["txn_count_1m"] == 1.0
    assert r4["amount_sum_minor_1m"] == 4000.0
    assert r4["txn_count_1h"] == 1.0
    assert r4["amount_sum_minor_1h"] == 4000.0
    assert r4["txn_count_24h"] == 4.0
    assert r4["amount_sum_minor_24h"] == 10000.0


def test_velocity_boundary_condition(redis_client):
    """Verify window interval (ts - w, ts] strictly excludes boundary ts - w."""
    base_t = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    acc = "acc_bound_1"

    t1 = {
        "txn_id": "txn_b1",
        "account_id": acc,
        "timestamp": base_t,
        "amount_minor": 5000,
    }
    OF.update(redis_client, t1)

    # t2 arrived at exactly base_t + 60s
    # Window interval: (base_t + 60s - 60s, base_t + 60s] = (base_t, base_t + 60s]
    # base_t is the lower boundary and MUST be excluded.
    t2 = {
        "txn_id": "txn_b2",
        "account_id": acc,
        "timestamp": base_t + timedelta(seconds=60),
        "amount_minor": 7000,
    }
    r2 = OF.update(redis_client, t2)
    assert r2["txn_count_1m"] == 1.0, "Boundary item at ts - w must be excluded"
    assert r2["amount_sum_minor_1m"] == 7000.0


# ============================================================================
# 4. Distinct Accounts (24h Devices and IPs)
# ============================================================================


def test_distinct_devices_and_ips_24h(redis_client):
    """Verify 24h distinct device and IP counts per account."""
    base_t = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    acc = "acc_dist_1"

    # 1. First device and IP
    t1 = {
        "txn_id": "txn_d1",
        "account_id": acc,
        "timestamp": base_t,
        "amount_minor": 1000,
        "device_id": "dev_A",
        "ip_address": "1.1.1.1",
    }
    r1 = OF.update(redis_client, t1)
    assert r1["distinct_devices_24h"] == 1.0
    assert r1["distinct_ips_24h"] == 1.0

    # 2. Second device, same IP (10m later)
    t2 = {
        "txn_id": "txn_d2",
        "account_id": acc,
        "timestamp": base_t + timedelta(minutes=10),
        "amount_minor": 1000,
        "device_id": "dev_B",
        "ip_address": "1.1.1.1",
    }
    r2 = OF.update(redis_client, t2)
    assert r2["distinct_devices_24h"] == 2.0
    assert r2["distinct_ips_24h"] == 1.0

    # 3. Repeated dev_A, new IP (20m later)
    t3 = {
        "txn_id": "txn_d3",
        "account_id": acc,
        "timestamp": base_t + timedelta(minutes=20),
        "amount_minor": 1000,
        "device_id": "dev_A",
        "ip_address": "2.2.2.2",
    }
    r3 = OF.update(redis_client, t3)
    assert r3["distinct_devices_24h"] == 2.0  # dev_A refreshed, dev_B still live
    assert r3["distinct_ips_24h"] == 2.0

    # 4. Missing device & IP in subsequent txn does not wipe active counts
    t4 = {
        "txn_id": "txn_d4",
        "account_id": acc,
        "timestamp": base_t + timedelta(minutes=30),
        "amount_minor": 1000,
    }
    r4 = OF.update(redis_client, t4)
    assert r4["distinct_devices_24h"] == 2.0
    assert r4["distinct_ips_24h"] == 2.0

    # 5. After 24 hours + 15 minutes:
    # dev_B (at +10m) has expired (> 24h).
    # dev_A (refreshed at +20m) is still active!
    t5 = {
        "txn_id": "txn_d5",
        "account_id": acc,
        "timestamp": base_t + timedelta(hours=24, minutes=15),
        "amount_minor": 1000,
    }
    r5 = OF.update(redis_client, t5)
    assert r5["distinct_devices_24h"] == 1.0  # only dev_A remains


# ============================================================================
# 5. Entity Sharing (7d Device and IP Sharing)
# ============================================================================


def test_entity_sharing_7d(redis_client):
    """Verify distinct accounts linked to the same device or IP within 7 days."""
    base_t = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    dev = "shared_dev_X"
    ip = "192.168.0.100"

    # Account 1 on device X
    t1 = {
        "txn_id": "txn_s1",
        "account_id": "acc_1",
        "timestamp": base_t,
        "amount_minor": 1000,
        "device_id": dev,
        "ip_address": ip,
    }
    r1 = OF.update(redis_client, t1)
    assert r1["device_accounts_7d"] == 1.0
    assert r1["ip_accounts_7d"] == 1.0

    # Account 2 on same device X (1 day later)
    t2 = {
        "txn_id": "txn_s2",
        "account_id": "acc_2",
        "timestamp": base_t + timedelta(days=1),
        "amount_minor": 1000,
        "device_id": dev,
        "ip_address": ip,
    }
    r2 = OF.update(redis_client, t2)
    assert r2["device_accounts_7d"] == 2.0
    assert r2["ip_accounts_7d"] == 2.0

    # Account 1 on same device X again (2 days later) -> does not double-count acc_1
    t3 = {
        "txn_id": "txn_s3",
        "account_id": "acc_1",
        "timestamp": base_t + timedelta(days=2),
        "amount_minor": 1000,
        "device_id": dev,
        "ip_address": ip,
    }
    r3 = OF.update(redis_client, t3)
    assert r3["device_accounts_7d"] == 2.0

    # Transaction without device does not touch placeholder keys in Redis
    t_sparse = {
        "txn_id": "txn_s4",
        "account_id": "acc_3",
        "timestamp": base_t + timedelta(days=3),
        "amount_minor": 1000,
    }
    r_sparse = OF.update(redis_client, t_sparse)
    assert r_sparse["device_accounts_7d"] == 0.0
    assert r_sparse["ip_accounts_7d"] == 0.0
    assert redis_client.exists("share:dev:-") == 0
    assert redis_client.exists("share:ip:-") == 0

    # After 7 days + 1 day: acc_2 (at day 1) has expired from device sharing.
    # acc_1 (refreshed at day 2) is still within 7 days!
    t5 = {
        "txn_id": "txn_s5",
        "account_id": "acc_4",
        "timestamp": base_t + timedelta(days=8, hours=1),
        "amount_minor": 1000,
        "device_id": dev,
        "ip_address": ip,
    }
    r5 = OF.update(redis_client, t5)
    assert r5["device_accounts_7d"] == 2.0  # acc_1 and acc_4


# ============================================================================
# 6. Welford's Algorithm and Account Recency
# ============================================================================


def test_welford_zscore_and_recency(redis_client):
    """Verify amount_zscore (calculated before update, 0 when n < 2) and seconds_since_prev."""
    base_t = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    acc = "acc_welford_1"

    # First txn: n=0 before -> zscore=0.0, seconds_since_prev=-1.0
    t1 = {
        "txn_id": "txn_w1",
        "account_id": acc,
        "timestamp": base_t,
        "amount_minor": 5000,
    }
    r1 = OF.update(redis_client, t1)
    assert r1["amount_zscore"] == 0.0
    assert r1["seconds_since_prev"] == -1.0

    # Second txn: n=1 before -> zscore=0.0 (requires n >= 2), seconds_since_prev=30.0
    t2 = {
        "txn_id": "txn_w2",
        "account_id": acc,
        "timestamp": base_t + timedelta(seconds=30),
        "amount_minor": 7000,
    }
    r2 = OF.update(redis_client, t2)
    assert r2["amount_zscore"] == 0.0
    assert r2["seconds_since_prev"] == 30.0

    # Third txn: n=2 before
    # Past transactions: 5000, 7000 -> mean = 6000, sample variance = 2,000,000, std = sqrt(2,000,000)
    # Incoming amount: 9000
    # Expected zscore = (9000 - 6000) / sqrt(2,000,000) = 3000 / 1414.21356 = 2.1213203435596424
    expected_std = math.sqrt(2_000_000.0)
    expected_z = (9000.0 - 6000.0) / expected_std

    t3 = {
        "txn_id": "txn_w3",
        "account_id": acc,
        "timestamp": base_t + timedelta(seconds=45),
        "amount_minor": 9000,
    }
    r3 = OF.update(redis_client, t3)
    assert math.isclose(r3["amount_zscore"], expected_z, rel_tol=1e-9)
    assert r3["seconds_since_prev"] == 15.0


def test_welford_zero_variance(redis_client):
    """Verify zero variance (identical past amounts) yields 0.0 z-score without division by zero."""
    base_t = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    acc = "acc_zero_var"

    for i in range(1, 4):
        t = {
            "txn_id": f"txn_zv_{i}",
            "account_id": acc,
            "timestamp": base_t + timedelta(seconds=i * 10),
            "amount_minor": 5000,
        }
        res = OF.update(redis_client, t)
        assert res["amount_zscore"] == 0.0


# ============================================================================
# 7. Calendar Features
# ============================================================================


def test_calendar_features(redis_client):
    """Verify calendar features hour_of_day and day_of_week in UTC."""
    # 2024-03-15 was a Friday (weekday = 4)
    dt_utc = datetime(2024, 3, 15, 14, 30, 0, tzinfo=timezone.utc)
    t = {
        "txn_id": "txn_cal_1",
        "account_id": "acc_cal_1",
        "timestamp": dt_utc,
        "amount_minor": 1000,
    }
    res = OF.update(redis_client, t)
    assert res["hour_of_day"] == 14.0
    assert res["day_of_week"] == 4.0

    # Test timezone offset conversion: 20:00 +05:30 is 14:30 UTC
    tz_ist = timezone(timedelta(hours=5, minutes=30))
    dt_ist = datetime(2024, 3, 15, 20, 0, 0, tzinfo=tz_ist)
    t_ist = {
        "txn_id": "txn_cal_2",
        "account_id": "acc_cal_1",
        "timestamp": dt_ist,
        "amount_minor": 1000,
    }
    res_ist = OF.update(redis_client, t_ist)
    assert res_ist["hour_of_day"] == 14.0
    assert res_ist["day_of_week"] == 4.0


# ============================================================================
# 8. Graph Features Integration & Fallbacks
# ============================================================================


def test_graph_features_reading_and_fallbacks(redis_client):
    """Verify graph features read from Redis hash gfeat:acct:{id} with canonical fallbacks."""
    acc_populated = "acc_graph_pop"
    acc_missing = "acc_graph_missing"

    # 1. Pre-populate snapshot in Redis
    redis_client.hset(
        f"gfeat:acct:{acc_populated}",
        mapping={
            "component_size": "25",
            "dist_to_fraud": "2.0",
            "risk_rank": "0.785",
        },
    )

    t_pop = {
        "txn_id": "txn_gp1",
        "account_id": acc_populated,
        "timestamp": "2024-01-01T12:00:00Z",
        "amount_minor": 1000,
    }
    r_pop = OF.update(redis_client, t_pop)
    assert r_pop["graph_component_size"] == 25.0
    assert r_pop["graph_dist_to_fraud"] == 2.0
    assert r_pop["graph_risk_rank"] == 0.785

    # 2. Account without pre-computed graph features falls back to canonical defaults
    t_miss = {
        "txn_id": "txn_gm1",
        "account_id": acc_missing,
        "timestamp": "2024-01-01T12:00:00Z",
        "amount_minor": 1000,
    }
    r_miss = OF.update(redis_client, t_miss)
    assert r_miss["graph_component_size"] == D.BY_NAME["graph_component_size"].default
    assert r_miss["graph_dist_to_fraud"] == D.BY_NAME["graph_dist_to_fraud"].default
    assert r_miss["graph_risk_rank"] == D.BY_NAME["graph_risk_rank"].default
    assert r_miss["graph_component_size"] == 1.0
    assert r_miss["graph_dist_to_fraud"] == -1.0
    assert r_miss["graph_risk_rank"] == 0.0


# ============================================================================
# 9. Event Ordering Safeguard
# ============================================================================


def test_event_ordering_safeguard(redis_client):
    """Verify out-of-order transactions raise ValueError and clean up seen tombstone."""
    base_t = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    acc = "acc_order_1"

    t1 = {
        "txn_id": "txn_o1",
        "account_id": acc,
        "timestamp": base_t,
        "amount_minor": 1000,
    }
    OF.update(redis_client, t1)

    t2 = {
        "txn_id": "txn_o2",
        "account_id": acc,
        "timestamp": base_t + timedelta(seconds=10),
        "amount_minor": 1000,
    }
    OF.update(redis_client, t2)

    # Arriving earlier than last_ts (T + 5s < T + 10s) -> raises ValueError
    t_retro = {
        "txn_id": "txn_o0",
        "account_id": acc,
        "timestamp": base_t + timedelta(seconds=5),
        "amount_minor": 1000,
    }
    with pytest.raises(ValueError, match="[Oo]ut-of-order"):
        OF.update(redis_client, t_retro)

    # Verify seen tombstone was deleted so it does not falsely claim processed
    assert redis_client.exists(f"seen:txn:{t_retro['txn_id']}") == 0

    # Same timestamp tie-breaker:
    # Incoming txn with same ts but lower/equal id -> raises ValueError
    t_tie_invalid = {
        "txn_id": "txn_o1_tie",  # "txn_o1_tie" < "txn_o2"
        "account_id": acc,
        "timestamp": base_t + timedelta(seconds=10),
        "amount_minor": 1000,
    }
    with pytest.raises(ValueError, match="[Oo]ut-of-order"):
        OF.update(redis_client, t_tie_invalid)

    # Incoming txn with same ts but higher id -> succeeds
    t_tie_valid = {
        "txn_id": "txn_o3",  # "txn_o3" > "txn_o2"
        "account_id": acc,
        "timestamp": base_t + timedelta(seconds=10),
        "amount_minor": 1000,
    }
    r_tie = OF.update(redis_client, t_tie_valid)
    assert r_tie is not None


# ============================================================================
# 10. Key TTL Contract
# ============================================================================


def test_key_ttl_contract(redis_client):
    """Verify all modified keys receive positive TTL bounded by KEY_TTL_S."""
    t = {
        "txn_id": "txn_ttl_1",
        "account_id": "acc_ttl_1",
        "timestamp": "2024-01-01T12:00:00Z",
        "amount_minor": 2500,
        "device_id": "dev_ttl_1",
        "ip_address": "172.16.0.1",
    }
    OF.update(redis_client, t)

    keys = OF.keys_for(t)
    for k in keys:
        ttl = redis_client.ttl(k)
        assert ttl > 0, f"Key {k} must have a positive TTL"
        assert ttl <= D.KEY_TTL_S, f"Key {k} TTL {ttl} must not exceed KEY_TTL_S"


# ============================================================================
# 11. Canonical 19-Feature Output Contract
# ============================================================================


def test_floating_point_precision_and_serialization(redis_client):
    """Verify %.17g serialization maintains double-precision floating point accuracy."""
    base_t = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    acc = "acc_prec_1"

    t1 = {
        "txn_id": "txn_p1",
        "account_id": acc,
        "timestamp": base_t,
        "amount": Decimal("1234567.89"),
    }
    r1 = OF.update(redis_client, t1)
    assert r1["amount_sum_minor_1m"] == 123456789.0


def test_read_graph_corrupted_fields(redis_client):
    """Verify read_graph gracefully falls back to canonical defaults on malformed field values."""
    acc = "acc_corrupt_graph"
    redis_client.hset(
        f"gfeat:acct:{acc}",
        mapping={
            "component_size": "not_a_number",
            "dist_to_fraud": "inf",
            "risk_rank": "nan",
        },
    )
    res = OF.read_graph(redis_client, acc)
    assert res["graph_component_size"] == D.BY_NAME["graph_component_size"].default
    assert res["graph_dist_to_fraud"] == D.BY_NAME["graph_dist_to_fraud"].default
    assert res["graph_risk_rank"] == D.BY_NAME["graph_risk_rank"].default


def test_out_of_order_preserves_clean_state(redis_client):
    """Verify out-of-order transaction aborts atomically without mutating Redis state."""
    base_t = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    acc = "acc_abort_clean"

    t1 = {
        "txn_id": "txn_clean_1",
        "account_id": acc,
        "timestamp": base_t + timedelta(seconds=100),
        "amount_minor": 5000,
    }
    r1 = OF.update(redis_client, t1)
    assert r1["txn_count_1m"] == 1.0

    # Old transaction attempting to insert at base_t (out-of-order)
    t_old = {
        "txn_id": "txn_clean_0",
        "account_id": acc,
        "timestamp": base_t + timedelta(seconds=50),
        "amount_minor": 9999,
    }
    with pytest.raises(ValueError, match="[Oo]ut-of-order"):
        OF.update(redis_client, t_old)

    # State must still reflect only txn_clean_1
    vel_key = f"vel:acct:{acc}:1m"
    assert redis_client.zcard(vel_key) == 1
    feat_key = f"feat:acct:{acc}"
    assert redis_client.hget(feat_key, "last_id").decode("utf-8") == "txn_clean_1"
    assert redis_client.hget(feat_key, "w_count").decode("utf-8") == "1"


def test_args_for_validation_errors():
    """Verify args_for raises ValueError on missing or non-finite inputs."""
    with pytest.raises(ValueError, match="timestamp"):
        OF.args_for({"txn_id": "t1", "account_id": "a1", "amount": 10.0})

    with pytest.raises(ValueError, match="amount"):
        OF.args_for({"txn_id": "t1", "account_id": "a1", "timestamp": "2024-01-01T00:00:00Z"})

    with pytest.raises(ValueError, match="finite"):
        OF.args_for({
            "txn_id": "t1",
            "account_id": "a1",
            "timestamp": "2024-01-01T00:00:00Z",
            "amount": float("nan"),
        })


def test_canonical_feature_output_contract(redis_client):
    """Verify update returns exactly 19 canonical features in exact D.FEATURE_LIST order."""
    t = {
        "txn_id": "txn_canonical_1",
        "account_id": "acc_canonical_1",
        "timestamp": "2024-05-10T08:15:30Z",
        "amount": 75.20,
        "device_id": "dev_canon_1",
        "ip_address": "192.168.1.50",
    }
    res = OF.update(redis_client, t)
    assert res is not None
    assert len(res) == 19
    assert list(res.keys()) == D.FEATURE_LIST
    for name, val in res.items():
        assert isinstance(val, float), f"Feature {name} value {val!r} must be float"
        assert math.isfinite(val), f"Feature {name} value {val!r} must be finite"


