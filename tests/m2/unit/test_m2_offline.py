"""Unit tests for M2 offline point-in-time feature computation."""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import numpy as np
import pandas as pd
import pytest

# Ensure ml/src is on sys.path
_ML_SRC = Path(__file__).resolve().parents[3] / "ml" / "src"
if str(_ML_SRC) not in sys.path:
    sys.path.insert(0, str(_ML_SRC))

from aegis_ml.features.definitions import FEATURE_LIST, WINDOWS
from aegis_ml.features.offline import (
    prepare,
    build_offline_features,
    _windowed_distinct,
    _join_graph_snapshots,
)


# =====================================================================
# 1. TIMESTAMP VALIDATION TESTS
# =====================================================================

def test_prepare_accepts_utc_and_non_utc_aware_timestamps():
    """Verify timezone-aware timestamps are accepted and normalized to UTC."""
    t_utc = datetime(2026, 3, 15, 10, 0, 0, 123456, tzinfo=timezone.utc)
    t_ist = datetime(2026, 3, 15, 15, 30, 0, 123456, tzinfo=timezone(timedelta(hours=5, minutes=30)))

    df = pd.DataFrame({
        "account_id": ["A1", "A1"],
        "ts": [t_utc, t_ist],
        "amount": [10.0, 20.0],
    })

    prep = prepare(df)
    assert prep["ts"].iloc[0] == pd.Timestamp("2026-03-15 10:00:00.123456+00:00")
    assert prep["ts"].iloc[1] == pd.Timestamp("2026-03-15 10:00:00.123456+00:00")
    assert prep["ts"].iloc[0].microsecond == 123456


def test_prepare_rejects_naive_timestamps():
    """Verify timezone-naive timestamps (datetime, Timestamp, or string) are rejected."""
    # 1. Naive datetime object
    df_naive = pd.DataFrame({
        "account_id": ["A1"],
        "ts": [datetime(2026, 3, 15, 10, 0, 0)],
        "amount": [10.0],
    })
    with pytest.raises(ValueError, match="Timezone-naive.*rejected"):
        prepare(df_naive)

    # 2. Naive ISO string
    df_str = pd.DataFrame({
        "account_id": ["A1"],
        "ts": ["2026-03-15T10:00:00"],
        "amount": [10.0],
    })
    with pytest.raises(ValueError, match="Timezone-naive.*rejected"):
        prepare(df_str)

    # 3. Naive pandas datetime64 Series
    df_series = pd.DataFrame({
        "account_id": ["A1"],
        "ts": pd.to_datetime(["2026-03-15 10:00:00"]),
        "amount": [10.0],
    })
    with pytest.raises(ValueError, match="Timezone-naive.*rejected"):
        prepare(df_series)


def test_prepare_rejects_missing_and_invalid_timestamps():
    """Verify missing (None, NaN, NaT) and invalid timestamp strings are rejected."""
    # Missing None / NaT
    df_missing = pd.DataFrame({
        "account_id": ["A1", "A1"],
        "ts": [datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc), None],
        "amount": [10.0, 20.0],
    })
    with pytest.raises(ValueError, match="missing or null values"):
        prepare(df_missing)

    # Invalid timestamp string
    df_invalid = pd.DataFrame({
        "account_id": ["A1"],
        "ts": ["not-a-timestamp"],
        "amount": [10.0],
    })
    with pytest.raises(ValueError, match="Invalid timestamp string"):
        prepare(df_invalid)


# =====================================================================
# 2. AMOUNT_MINOR VALIDATION TESTS
# =====================================================================

def test_prepare_validates_existing_amount_minor():
    """Verify existing amount_minor integer values are preserved."""
    df = pd.DataFrame({
        "account_id": ["A1", "A2"],
        "ts": [
            datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 15, 10, 1, tzinfo=timezone.utc),
        ],
        "amount_minor": [1050, 25000],
    })
    prep = prepare(df)
    assert list(prep["amount_minor"]) == [1050, 25000]
    assert prep["amount_minor"].dtype == np.int64


def test_prepare_rejects_invalid_existing_amount_minor():
    """Verify NaN, non-finite, fractional, or overflowing existing amount_minor are rejected."""
    base_ts = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)

    # 1. NaN / Null
    df_nan = pd.DataFrame({"account_id": ["A1"], "ts": [base_ts], "amount_minor": [np.nan]})
    with pytest.raises(ValueError, match="missing or null values"):
        prepare(df_nan)

    # 2. Non-finite (Infinity)
    df_inf = pd.DataFrame({"account_id": ["A1"], "ts": [base_ts], "amount_minor": [float("inf")]})
    with pytest.raises(ValueError, match="Non-finite"):
        prepare(df_inf)

    # 3. Fractional float
    df_frac = pd.DataFrame({"account_id": ["A1"], "ts": [base_ts], "amount_minor": [1050.75]})
    with pytest.raises(ValueError, match="Fractional"):
        prepare(df_frac)

    # 4. Fractional Decimal
    df_dec = pd.DataFrame({"account_id": ["A1"], "ts": [base_ts], "amount_minor": [Decimal("1050.5")]})
    with pytest.raises(ValueError, match="Fractional"):
        prepare(df_dec)

    # 5. Fractional String
    df_str = pd.DataFrame({"account_id": ["A1"], "ts": [base_ts], "amount_minor": ["1050.5"]})
    with pytest.raises(ValueError, match="Fractional"):
        prepare(df_str)

    # 6. Out of 64-bit signed integer range
    df_overflow = pd.DataFrame({"account_id": ["A1"], "ts": [base_ts], "amount_minor": [10**25]})
    with pytest.raises(ValueError, match="out of 64-bit signed integer range"):
        prepare(df_overflow)


def test_prepare_converts_amount_when_amount_minor_missing():
    """Verify conversion from major currency units via amount column."""
    df = pd.DataFrame({
        "account_id": ["A1", "A2"],
        "ts": [
            datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 15, 10, 1, tzinfo=timezone.utc),
        ],
        "amount": [Decimal("123.45"), 10.50],
    })
    prep = prepare(df)
    assert list(prep["amount_minor"]) == [12345, 1050]


# =====================================================================
# 3. GRAPH SNAPSHOT DETERMINISM & POINT-IN-TIME JOIN TESTS
# =====================================================================

def test_graph_snapshots_point_in_time_semantics():
    """Verify backward-only join: computed_at <= txn.ts, latest eligible chosen, no future leakage."""
    # Transactions at 10:00, 11:00, 12:00
    t1 = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 3, 15, 11, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)

    txns = pd.DataFrame({
        "id": ["T1", "T2", "T3"],
        "account_id": ["ACC_1", "ACC_1", "ACC_2"],
        "ts": [t1, t2, t3],
        "amount_minor": [1000, 2000, 3000],
    })
    prep_txns = prepare(txns)

    # Snapshots for ACC_1:
    # S1 at 09:00 (before T1 and T2)
    # S2 at 11:00 (exact match for T2, after T1)
    # S3 at 13:00 (future relative to all transactions -> must NEVER be consumed)
    snapshots = pd.DataFrame({
        "account_id": ["ACC_1", "ACC_1", "ACC_1"],
        "snapshot_at": [
            datetime(2026, 3, 15, 9, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 15, 11, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 15, 13, 0, tzinfo=timezone.utc),
        ],
        "component_size": [5.0, 12.0, 99.0],
        "dist_to_fraud": [3.0, 1.0, 0.0],
        "risk_rank": [0.10, 0.45, 0.99],
    })

    res = _join_graph_snapshots(prep_txns, snapshots)

    # T1 (at 10:00) should consume S1 (at 09:00)
    assert res["graph_component_size"][0] == 5.0
    assert res["graph_dist_to_fraud"][0] == 3.0
    assert res["graph_risk_rank"][0] == 0.10

    # T2 (at 11:00) should consume S2 (at 11:00 exact match)
    assert res["graph_component_size"][1] == 12.0
    assert res["graph_dist_to_fraud"][1] == 1.0
    assert res["graph_risk_rank"][1] == 0.45

    # T3 (ACC_2, no snapshots exist) should receive defaults
    assert res["graph_component_size"][2] == 1.0
    assert res["graph_dist_to_fraud"][2] == -1.0
    assert res["graph_risk_rank"][2] == 0.0


def test_graph_snapshots_duplicate_timestamp_tiebreaker():
    """Verify deterministic selection if duplicate snapshots share identical (account, timestamp)."""
    t = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)
    txns = prepare(pd.DataFrame({
        "id": ["T1"],
        "account_id": ["ACC_1"],
        "ts": [t],
        "amount_minor": [1000],
    }))

    # Two snapshots at same timestamp with snapshot_id tie-breaker
    snapshots = pd.DataFrame({
        "account_id": ["ACC_1", "ACC_1"],
        "snapshot_at": [t, t],
        "snapshot_id": ["SNAP_A", "SNAP_B"],
        "component_size": [10.0, 20.0],
        "dist_to_fraud": [2.0, 1.0],
        "risk_rank": [0.2, 0.4],
    })

    res = _join_graph_snapshots(txns, snapshots)
    # Highest snapshot_id ('SNAP_B') is chosen deterministically
    assert res["graph_component_size"][0] == 20.0


# =====================================================================
# 4. WINDOWED DISTINCT & TWO-POINTER TESTS
# =====================================================================

def test_windowed_distinct_behavior():
    """Verify repeated keys, null values, current inclusion, and expiration."""
    # Timestamps at 0s, 10s, 50s, 70s
    # Window size: 60s -> (ts - 60, ts]
    timestamps = np.array([0.0, 10.0, 50.0, 70.0])
    keys = ["dev_A", "dev_A", "dev_B", "dev_A"]

    # At 0s: window (-60, 0] -> ["dev_A"] -> distinct = 1
    # At 10s: window (-50, 10] -> ["dev_A", "dev_A"] -> distinct = 1
    # At 50s: window (-10, 50] -> ["dev_A", "dev_A", "dev_B"] -> distinct = 2
    # At 70s: window (10, 70] -> 0s and 10s are evicted! (0 <= 10, 10 <= 10)
    # Remaining in (10, 70]: 50s ("dev_B") and 70s ("dev_A") -> distinct = 2
    distinct = _windowed_distinct(timestamps, keys, 60.0)
    assert list(distinct) == [1.0, 1.0, 2.0, 2.0]


def test_windowed_distinct_null_handling():
    """Verify null/None keys are not counted."""
    timestamps = np.array([0.0, 10.0, 20.0])
    keys = [None, "dev_A", None]

    distinct = _windowed_distinct(timestamps, keys, 60.0)
    assert list(distinct) == [0.0, 1.0, 1.0]


# =====================================================================
# 5. WELFORD Z-SCORE & RECENCY TESTS
# =====================================================================

def test_welford_zscore_and_recency():
    """Verify z-score uses profile before current txn, and recency tracks seconds since prev."""
    t0 = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=120)
    t2 = t1 + timedelta(seconds=60)

    # Account with 3 transactions: amounts 100, 100, 200
    df = pd.DataFrame({
        "id": ["T1", "T2", "T3"],
        "account_id": ["ACC_1", "ACC_1", "ACC_1"],
        "ts": [t0, t1, t2],
        "amount_minor": [10000, 10000, 20000],
    })

    out = build_offline_features(df)

    # 1. Recency
    # First txn: -1.0
    # Second txn: 120s
    # Third txn: 60s
    assert out["seconds_since_prev"].iloc[0] == -1.0
    assert out["seconds_since_prev"].iloc[1] == 120.0
    assert out["seconds_since_prev"].iloc[2] == 60.0

    # 2. Amount Z-score
    # Txn 1: n=0 -> zscore = 0.0
    # Txn 2: n=1 -> zscore = 0.0
    # Txn 3: previous profile has amounts [10000, 10000] -> mean=10000, var=0 -> zscore = 0.0
    assert out["amount_zscore"].iloc[0] == 0.0
    assert out["amount_zscore"].iloc[1] == 0.0
    assert out["amount_zscore"].iloc[2] == 0.0

    # Now add fourth transaction with amount 50000:
    # Profile has [10000, 10000, 20000] -> mean = 13333.33, var = 33333333.33, std = 5773.50
    t3 = t2 + timedelta(seconds=30)
    df4 = pd.concat([df, pd.DataFrame({
        "id": ["T4"],
        "account_id": ["ACC_1"],
        "ts": [t3],
        "amount_minor": [50000],
    })]).reset_index(drop=True)

    out4 = build_offline_features(df4)
    # Z-score for T4: (50000 - 13333.33) / 5773.50 = 6.35
    assert out4["amount_zscore"].iloc[3] == pytest.approx(6.35085, rel=1e-3)


# =====================================================================
# 6. END-TO-END CANONICAL FEATURES INTEGRATION TEST
# =====================================================================

def test_build_offline_features_full_integration():
    """Verify all 19 canonical features are generated in exact FEATURE_LIST order."""
    t0 = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)  # Monday 14:30 UTC
    t1 = t0 + timedelta(seconds=30)

    df = pd.DataFrame({
        "txn_id": ["T1", "T2"],
        "account_id": ["ACC_1", "ACC_1"],
        "occurred_at": [t0, t1],
        "amount": [Decimal("50.00"), Decimal("75.00")],
        "device_id": ["DEV_1", "DEV_1"],
        "ip_address": ["192.168.1.1", "192.168.1.1"],
    })

    snapshots = pd.DataFrame({
        "account_id": ["ACC_1"],
        "snapshot_at": [t0 - timedelta(hours=1)],
        "component_size": [8.0],
        "dist_to_fraud": [2.0],
        "risk_rank": [0.15],
    })

    features_df = build_offline_features(df, snapshots)

    # Confirm all 19 features exist in the exact canonical order
    for f_name in FEATURE_LIST:
        assert f_name in features_df.columns

    # Verify column slice matches FEATURE_LIST exactly
    sub_cols = [c for c in features_df.columns if c in FEATURE_LIST]
    assert sub_cols == FEATURE_LIST

    # Velocity:
    # At T1: count_1m = 1, amount_1m = 5000
    # At T2 (30s later): count_1m = 2, amount_1m = 12500
    assert features_df["txn_count_1m"].iloc[0] == 1.0
    assert features_df["amount_sum_minor_1m"].iloc[0] == 5000.0
    assert features_df["txn_count_1m"].iloc[1] == 2.0
    assert features_df["amount_sum_minor_1m"].iloc[1] == 12500.0

    # Distinct:
    assert features_df["distinct_devices_24h"].iloc[0] == 1.0
    assert features_df["distinct_ips_24h"].iloc[0] == 1.0
    assert features_df["device_accounts_7d"].iloc[0] == 1.0
    assert features_df["ip_accounts_7d"].iloc[0] == 1.0

    # Calendar:
    assert features_df["hour_of_day"].iloc[0] == 14.0
    assert features_df["day_of_week"].iloc[0] == 0.0  # Monday

    # Graph:
    assert features_df["graph_component_size"].iloc[0] == 8.0
    assert features_df["graph_dist_to_fraud"].iloc[0] == 2.0
    assert features_df["graph_risk_rank"].iloc[0] == 0.15

