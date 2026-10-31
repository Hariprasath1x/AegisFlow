"""Offline/Online Feature Parity Proof for AegisFlow M2.

Proves exact, tolerance-free numerical and structural equality between:
1. Offline point-in-time feature computation via aegis_ml.features.offline.
2. Online feature computation via services.ingestion.online_features and Redis Lua.

Runs over a deterministic 5,000-transaction synthetic stream exercising all
window boundaries, timestamp ties, missing entities, and entity sharing.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Generator
from urllib.parse import urlparse
import pytest
import pandas as pd
import redis
import fakeredis

# Ensure ml/src, services/ingestion/src, and tests are on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ML_SRC = _REPO_ROOT / "ml" / "src"
_INGESTION_SRC = _REPO_ROOT / "services" / "ingestion" / "src"
_TESTS_DIR = _REPO_ROOT / "tests"
for p in (_ML_SRC, _INGESTION_SRC, _TESTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from aegis_ml.features import definitions as D
from aegis_ml.features.offline import build, prepare
import ingestion.online_features as OF
from m2.support import stream


# ============================================================================
# Redis Configuration & Safety Utilities
# ============================================================================


def validate_scratch_redis_url(url: str) -> int:
    """Validate that PARITY_REDIS_URL specifies an isolated non-zero scratch database index.

    Fails safely if URL is missing a database index or points to production / default database 0.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("redis", "rediss"):
        raise ValueError(
            f"Invalid Redis URL scheme in PARITY_REDIS_URL: {url!r}. Must start with redis:// or rediss://"
        )

    path = parsed.path.strip("/")
    if not path:
        raise ValueError(
            f"PARITY_REDIS_URL must explicitly specify a database index (e.g. redis://localhost:6379/15). Got: {url!r}"
        )

    try:
        db_index = int(path)
    except ValueError:
        raise ValueError(f"PARITY_REDIS_URL database index must be an integer, got: {path!r}")

    if db_index <= 0:
        raise ValueError(
            f"PARITY_REDIS_URL must use an isolated scratch database (db > 0). "
            f"Database {db_index} is forbidden to prevent accidental data loss in database 0."
        )

    return db_index


@pytest.fixture
def fakeredis_client() -> Generator[fakeredis.FakeRedis, None, None]:
    """Provide a clean FakeRedis instance with database cleanup."""
    client = fakeredis.FakeRedis()
    client.flushdb()
    try:
        yield client
    finally:
        client.flushdb()


@pytest.fixture
def real_redis_client() -> Generator[redis.Redis, None, None]:
    """Provide a connection to real Redis when PARITY_REDIS_URL is configured on a scratch db."""
    redis_url = os.getenv("PARITY_REDIS_URL")
    if not redis_url:
        pytest.skip("PARITY_REDIS_URL environment variable is not set; skipping real Redis parity test.")

    db_index = validate_scratch_redis_url(redis_url)
    client = redis.Redis.from_url(redis_url, decode_responses=False)

    try:
        client.ping()
    except Exception as exc:
        pytest.fail(f"PARITY_REDIS_URL provided ({redis_url}) but connection failed: {exc}")

    # Safety check on connected database
    connection_kwargs = client.get_connection_kwargs()
    conn_db = connection_kwargs.get("db", 0)
    if conn_db <= 0 or conn_db != db_index:
        pytest.fail(f"Connected to unexpected database {conn_db}; expected scratch database {db_index}")

    # Isolated scratch DB cleanup
    client.flushdb()
    try:
        yield client
    finally:
        client.flushdb()


# ============================================================================
# 1. Stream Validation Tests
# ============================================================================


def test_stream_really_exercises_the_edges():
    """Verify that the 5,000-row synthetic stream exercises all critical boundary and edge conditions."""
    df = stream(n=5000, accounts=300, devices=150, ips=100, seed=7)

    # 1. Exactly 5,000 transactions
    assert len(df) == 5000, f"Expected 5,000 transactions, got {len(df)}"

    # 2. Timestamp ties exist (same ts, different accounts or same account)
    num_ties = int(df["ts"].duplicated().sum())
    assert num_ties > 0, "Stream must include duplicate timestamps"
    assert num_ties >= 50, f"Stream should have substantial timestamp ties, got {num_ties}"

    # 3. Missing device IDs exist
    num_missing_devices = int(df["device_id"].isna().sum())
    assert num_missing_devices > 0, "Stream must include transactions with missing device_id"
    assert num_missing_devices >= 100, f"Expected >= 100 missing devices, got {num_missing_devices}"

    # 4. Missing IP IDs exist
    num_missing_ips = int(df["ip_address"].isna().sum())
    assert num_missing_ips > 0, "Stream must include transactions with missing ip_address"
    assert num_missing_ips >= 100, f"Expected >= 100 missing IPs, got {num_missing_ips}"

    # 5. Shared device IDs exist (same device used by multiple accounts)
    dev_account_counts = df.dropna(subset=["device_id"]).groupby("device_id")["account_id"].nunique()
    assert (dev_account_counts > 1).any(), "Stream must include devices shared across multiple accounts"

    # 6. Shared IP IDs exist (same IP used by multiple accounts)
    ip_account_counts = df.dropna(subset=["ip_address"]).groupby("ip_address")["account_id"].nunique()
    assert (ip_account_counts > 1).any(), "Stream must include IPs shared across multiple accounts"

    # 7. Exact 60-second boundary exists for an account
    acc_groups = df.groupby("account_id")
    has_exact_60s = False
    has_exact_3600s = False
    has_just_inside_60s = False
    has_just_outside_60s = False
    has_just_inside_3600s = False
    has_just_outside_3600s = False

    for _, group in acc_groups:
        if len(group) >= 2:
            ts_diffs = group["ts"].diff().dt.total_seconds().dropna().to_numpy()
            if any(abs(d - 60.0) < 1e-6 for d in ts_diffs):
                has_exact_60s = True
            if any(abs(d - 3600.0) < 1e-6 for d in ts_diffs):
                has_exact_3600s = True
            if any(abs(d - 59.0) < 1e-6 for d in ts_diffs):
                has_just_inside_60s = True
            if any(abs(d - 61.0) < 1e-6 for d in ts_diffs):
                has_just_outside_60s = True
            if any(abs(d - 3599.0) < 1e-6 for d in ts_diffs):
                has_just_inside_3600s = True
            if any(abs(d - 3601.0) < 1e-6 for d in ts_diffs):
                has_just_outside_3600s = True

    assert has_exact_60s, "Stream must contain an account with an exact 60.0s transaction boundary"
    assert has_exact_3600s, "Stream must contain an account with an exact 3600.0s transaction boundary"
    assert has_just_inside_60s, "Stream must contain an account with a transaction 59.0s (just inside 1m) apart"
    assert has_just_outside_60s, "Stream must contain an account with a transaction 61.0s (just outside 1m) apart"
    assert has_just_inside_3600s, "Stream must contain an account with a transaction 3599.0s (just inside 1h) apart"
    assert has_just_outside_3600s, "Stream must contain an account with a transaction 3601.0s (just outside 1h) apart"


# ============================================================================
# 2. Parity Execution Helper
# ============================================================================


def _execute_and_verify_parity(
    redis_client: redis.Redis | fakeredis.FakeRedis,
    stream_df: pd.DataFrame,
    backend_name: str,
) -> tuple[float, float]:
    """Execute complete 5,000-row parity verification between offline builder and online Redis Lua.

    Returns:
        (offline_duration_s, online_duration_s)
    """
    n_txns = len(stream_df)
    assert n_txns == 5000, f"Parity proof requires exactly 5,000 transactions, got {n_txns}"

    # 1. Compute offline features
    t0_off = time.perf_counter()
    offline_features_df = build(stream_df)
    offline_duration = time.perf_counter() - t0_off

    # 2. Prepare transactions for sequential online processing
    prepared_txns = prepare(stream_df)
    assert len(prepared_txns) == n_txns

    # 3. Process transactions sequentially through atomic online update
    online_feature_rows: list[dict[str, float]] = []
    t0_on = time.perf_counter()

    for idx, row in prepared_txns.iterrows():
        t_dict = row.to_dict()
        res = OF.update(redis_client, t_dict)
        if res is None:
            pytest.fail(
                f"[{backend_name}] Transaction at row {idx} (id={t_dict.get('id')}, "
                f"account={t_dict.get('account_id')}, ts={t_dict.get('ts')}) unexpectedly returned None"
            )
        online_feature_rows.append(res)

    online_duration = time.perf_counter() - t0_on

    # 4. Construct DataFrame from online results
    online_features_df = pd.DataFrame(online_feature_rows, index=prepared_txns.index)

    # 5. Extract the canonical 19 features in exact order
    offline_matrix = offline_features_df.loc[online_features_df.index, D.FEATURE_LIST].astype(float)
    online_matrix = online_features_df[D.FEATURE_LIST].astype(float)

    # 6. Prove exact equality without tolerances
    try:
        pd.testing.assert_frame_equal(
            offline_matrix,
            online_matrix,
            check_exact=True,
        )
    except AssertionError as err:
        # Diagnostic inspection on mismatch
        diff_mask = offline_matrix != online_matrix
        mismatch_cols = diff_mask.any()
        first_col = mismatch_cols[mismatch_cols].index[0]
        first_row_idx = diff_mask[first_col].idxmax()
        txn_meta = prepared_txns.loc[first_row_idx]
        off_val = offline_matrix.loc[first_row_idx, first_col]
        on_val = online_matrix.loc[first_row_idx, first_col]

        pytest.fail(
            f"[{backend_name}] Parity failure at row {first_row_idx} (id={txn_meta.get('id')}, "
            f"account={txn_meta.get('account_id')}, ts={txn_meta.get('ts')}) on feature '{first_col}': "
            f"offline={off_val!r} vs online={on_val!r}.\nOriginal error: {err}"
        )

    return offline_duration, online_duration


# ============================================================================
# 3. Parity Proof Test Suites
# ============================================================================


def test_offline_online_parity_fakeredis(fakeredis_client):
    """Prove exact offline/online parity over 5,000 transactions using Fakeredis."""
    stream_df = stream(n=5000, accounts=300, devices=150, ips=100, seed=7)
    off_dur, on_dur = _execute_and_verify_parity(fakeredis_client, stream_df, "Fakeredis")
    print(
        f"\n[Fakeredis Parity Proof PASSED] 5,000 transactions, 19 features, exact equality. "
        f"Offline: {off_dur:.2f}s, Online: {on_dur:.2f}s"
    )


def test_offline_online_parity_real_redis(real_redis_client):
    """Prove exact offline/online parity over 5,000 transactions using Real Redis."""
    info = real_redis_client.info()
    redis_version = info.get("redis_version", "unknown")
    print(f"\n[Real Redis Connected] Server version: {redis_version}")

    stream_df = stream(n=5000, accounts=300, devices=150, ips=100, seed=7)
    off_dur, on_dur = _execute_and_verify_parity(
        real_redis_client, stream_df, f"Real Redis {redis_version}"
    )

    # Verify TTL contract and ZSet state on sample keys
    sample_txn = stream_df.iloc[-1]
    sample_keys = OF.keys_for(sample_txn.to_dict())
    for k in sample_keys:
        ttl = real_redis_client.ttl(k)
        assert ttl > 0, f"Sample key {k} must have positive TTL in real Redis"
        assert ttl <= D.KEY_TTL_S

    print(
        f"\n[Real Redis Parity Proof PASSED] Server {redis_version}, 5,000 transactions, 19 features, exact equality. "
        f"Offline: {off_dur:.2f}s, Online: {on_dur:.2f}s"
    )


def test_scratch_database_safety_guardrails():
    """Verify that validate_scratch_redis_url strictly rejects db 0 and unisolated URLs."""
    # Rejects missing db index
    with pytest.raises(ValueError, match="database index"):
        validate_scratch_redis_url("redis://localhost:6379")

    with pytest.raises(ValueError, match="database index"):
        validate_scratch_redis_url("redis://localhost:6379/")

    # Rejects database 0
    with pytest.raises(ValueError, match="forbidden"):
        validate_scratch_redis_url("redis://localhost:6379/0")

    # Rejects negative database
    with pytest.raises(ValueError, match="forbidden"):
        validate_scratch_redis_url("redis://localhost:6379/-1")

    # Rejects non-integer database
    with pytest.raises(ValueError, match="integer"):
        validate_scratch_redis_url("redis://localhost:6379/scratch")

    # Rejects invalid scheme
    with pytest.raises(ValueError, match="scheme"):
        validate_scratch_redis_url("http://localhost:6379/15")

    # Accepts valid isolated scratch database
    assert validate_scratch_redis_url("redis://localhost:6379/15") == 15
    assert validate_scratch_redis_url("rediss://secure-host:6380/1") == 1

