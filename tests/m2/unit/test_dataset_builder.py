"""Unit tests for AegisFlow M2 Supervised Dataset Builder.

Verifies:
  - Collision-safe account ID normalization (P0-3)
  - Graph snapshot normalization contract (P0-2)
  - Feature builder ordering and identity contract (P0-1)
  - Complete outcome table invariants (P1-1)
  - Multi-incident target resolution rules (Three-state target model)
  - Point-in-time correctness, anti-leakage, and maturity buffers
  - Shuffled input invariance
  - Full canonical synthetic manifest integration
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import uuid

import numpy as np
import pandas as pd
import pytest

from aegis_ml.features.dataset_builder import (
    VALID_FRAUD_TYPOLOGIES,
    build_supervised_training_dataset,
    normalize_account_id,
    normalize_timestamp_series,
    normalize_txn_id,
    normalize_utc,
    resolve_graph_snapshots,
    verify_invariants,
)
from aegis_ml.features.definitions import FEATURE_LIST
from aegis_ml.scenarios.constants import (
    MATURITY_BOUNDARY,
    OBSERVATION_CUTOFF,
    SEED,
)
from aegis_ml.scenarios.generator import ManifestGenerator


# =====================================================================
# 1. ACCOUNT ID & TIMESTAMP NORMALIZATION (P0-3)
# =====================================================================

def test_normalize_account_id_valid_cases():
    """Verify type-aware, collision-safe account ID canonicalization."""
    # 1. UUID instances -> canonical lowercase string
    u = uuid.UUID("12345678-1234-5678-1234-567812345678")
    assert normalize_account_id(u) == "12345678-1234-5678-1234-567812345678"

    # 2. String UUIDs with mixed case -> canonical lowercase string
    assert normalize_account_id("12345678-ABCD-5678-1234-567812345678") == "12345678-abcd-5678-1234-567812345678"

    # 3. String accounts with distinct cases -> case strictly preserved (no collision)
    assert normalize_account_id("Account-A") == "Account-A"
    assert normalize_account_id("account-a") == "account-a"
    assert normalize_account_id("Account-A") != normalize_account_id("account-a")

    # 4. Integer IDs -> stringified directly
    assert normalize_account_id(12345) == "12345"

    # 5. Integral float -> stringified integer
    assert normalize_account_id(100.0) == "100"


def test_normalize_account_id_invalid_cases():
    """Verify invalid account IDs are rejected."""
    # Null / NaN / empty
    with pytest.raises(ValueError, match="cannot be null or NaN"):
        normalize_account_id(None)
    with pytest.raises(ValueError, match="cannot be null or NaN"):
        normalize_account_id(np.nan)
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_account_id("   ")

    # Non-integral float
    with pytest.raises(ValueError, match="Floating-point account_id.*invalid"):
        normalize_account_id(123.45)


def test_normalize_utc_valid_and_invalid():
    """Verify timezone-aware normalization and naive rejection."""
    t_utc = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert normalize_utc(t_utc) == t_utc

    # Timezone conversion from IST
    ist = timezone(timedelta(hours=5, minutes=30))
    t_ist = datetime(2026, 3, 1, 17, 30, 0, tzinfo=ist)
    assert normalize_utc(t_ist) == t_utc

    # Naive timestamp rejected
    with pytest.raises(ValueError, match="Timezone-naive.*rejected"):
        normalize_utc(datetime(2026, 3, 1, 12, 0, 0))

    # Null / NaT rejected
    with pytest.raises(ValueError, match="cannot be null"):
        normalize_utc(None)


# =====================================================================
# 2. GRAPH SNAPSHOT NORMALIZATION (P0-2)
# =====================================================================

def test_resolve_graph_snapshots_aliases_and_precedence():
    """Verify dual column resolution and schema contracts."""
    # 1. None or empty handled cleanly
    assert resolve_graph_snapshots(None) is None
    empty_df = pd.DataFrame()
    assert len(resolve_graph_snapshots(empty_df)) == 0

    # 2. Canonical column names
    df_canonical = pd.DataFrame([{
        "account_id": "acc-1",
        "snapshot_at": datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc),
        "component_size": 2.0,
    }])
    res = resolve_graph_snapshots(df_canonical)
    assert "account_id" in res.columns
    assert "snapshot_at" in res.columns

    # 3. Alias column names (entity_id and computed_at)
    df_aliases = pd.DataFrame([{
        "entity_id": "acc-2",
        "computed_at": datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc),
        "component_size": 3.0,
    }])
    res_alias = resolve_graph_snapshots(df_aliases)
    assert "account_id" in res_alias.columns
    assert "snapshot_at" in res_alias.columns
    assert res_alias["account_id"].iloc[0] == "acc-2"

    # 4. Dual column precedence ('account_id' > 'entity_id', 'snapshot_at' > 'computed_at')
    df_dual = pd.DataFrame([{
        "account_id": "primary-acc",
        "entity_id": "secondary-acc",
        "snapshot_at": datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc),
        "computed_at": datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    }])
    res_dual = resolve_graph_snapshots(df_dual)
    assert res_dual["account_id"].iloc[0] == "primary-acc"
    assert res_dual["snapshot_at"].iloc[0] == pd.Timestamp("2026-02-01 00:00:00+00:00")

    # 5. Missing required columns raise ValueError
    with pytest.raises(ValueError, match="missing required entity column"):
        resolve_graph_snapshots(pd.DataFrame([{"snapshot_at": "2026-01-01T00:00:00Z"}]))
    with pytest.raises(ValueError, match="missing required timestamp column"):
        resolve_graph_snapshots(pd.DataFrame([{"account_id": "acc-1"}]))


# =====================================================================
# 3. OUTCOME TABLE INVARIANTS (P1-1)
# =====================================================================

def test_verify_invariants_valid_and_invalid():
    """Verify mathematical and relational invariants on outcome tables."""
    txns = pd.DataFrame([
        {"txn_id": "t-1", "account_id": "a-1"},
        {"txn_id": "t-2", "account_id": "a-1"},
    ])

    valid_outcomes = pd.DataFrame([
        {"txn_id": "t-1", "account_id": "a-1", "target": 1, "target_typology": "card_not_present", "primary_outcome": "positive_fraud", "reason": "r1"},
        {"txn_id": "t-2", "account_id": "a-1", "target": 0, "target_typology": "legitimate_clean", "primary_outcome": "negative_clean", "reason": "r2"},
    ])
    # Should succeed without error
    verify_invariants(txns, valid_outcomes)

    # 1. Row count mismatch
    with pytest.raises(ValueError, match="Outcome row count.*must match"):
        verify_invariants(txns, valid_outcomes.iloc[:1])

    # 2. Duplicate txn_id
    dup_outcomes = valid_outcomes.copy()
    dup_outcomes.loc[1, "txn_id"] = "t-1"
    with pytest.raises(ValueError, match="strictly unique"):
        verify_invariants(txns, dup_outcomes)

    # 3. Target 1 with incompatible outcome
    bad_target_outcome = valid_outcomes.copy()
    bad_target_outcome.loc[0, "primary_outcome"] = "negative_clean"
    with pytest.raises(ValueError, match="All rows with target=1 must have primary_outcome='positive_fraud'"):
        verify_invariants(txns, bad_target_outcome)

    # 4. Target 0 with incompatible typology
    bad_clean_typology = valid_outcomes.copy()
    bad_clean_typology.loc[1, "target_typology"] = "card_not_present"
    with pytest.raises(ValueError, match="must have target_typology='legitimate_clean'"):
        verify_invariants(txns, bad_clean_typology)


# =====================================================================
# 4. TARGET RESOLUTION RULES & THREE-STATE MODEL
# =====================================================================

def test_direct_transaction_fraud_resolution():
    """Direct transaction fraud label resolves to target=1 (positive_fraud)."""
    t_time = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    txns = pd.DataFrame([{
        "txn_id": "txn-fraud-1",
        "account_id": "acc-1",
        "occurred_at": t_time,
        "amount": 25000.0,
    }])
    labels = pd.DataFrame([{
        "label_id": uuid.uuid4(),
        "txn_id": "txn-fraud-1",
        "account_id": None,
        "label_target": "transaction",
        "is_fraud": True,
        "fraud_type": "card_not_present",
        "confidence": 0.95,
        "labeled_at": t_time + timedelta(days=2),
        "incident_start_at": t_time,
        "incident_end_at": t_time + timedelta(days=1),
    }])

    pit_df, sup_df, out_df = build_supervised_training_dataset(txns, labels)

    assert len(sup_df) == 1
    assert sup_df["target"].iloc[0] == 1
    assert sup_df["target_typology"].iloc[0] == "card_not_present"
    assert sup_df["primary_outcome"].iloc[0] == "positive_fraud"
    assert sup_df["reason"].iloc[0] == "direct_transaction_fraud"


def test_direct_transaction_clean_resolution():
    """Direct transaction clean label resolves to target=0 (negative_clean)."""
    t_time = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    txns = pd.DataFrame([{
        "txn_id": "txn-clean-1",
        "account_id": "acc-clean",
        "occurred_at": t_time,
        "amount": 500.0,
    }])
    labels = pd.DataFrame([{
        "label_id": uuid.uuid4(),
        "txn_id": "txn-clean-1",
        "account_id": None,
        "label_target": "transaction",
        "is_fraud": False,
        "fraud_type": None,
        "confidence": 1.0,
        "labeled_at": t_time + timedelta(days=1),
    }])

    pit_df, sup_df, out_df = build_supervised_training_dataset(txns, labels)

    assert len(sup_df) == 1
    assert sup_df["target"].iloc[0] == 0
    assert sup_df["target_typology"].iloc[0] == "legitimate_clean"
    assert sup_df["primary_outcome"].iloc[0] == "negative_clean"


def test_direct_clean_vs_account_fraud_conflict_fails_closed():
    """Direct clean conflicting with active account fraud resolves to target=None (direct_conflict)."""
    t_time = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    txns = pd.DataFrame([{
        "txn_id": "txn-conf-1",
        "account_id": "acc-victim",
        "occurred_at": t_time,
        "amount": 1000.0,
    }])
    labels = pd.DataFrame([
        # Direct clean label on the transaction
        {
            "label_id": uuid.uuid4(),
            "txn_id": "txn-conf-1",
            "account_id": None,
            "label_target": "transaction",
            "is_fraud": False,
            "fraud_type": None,
            "confidence": 1.0,
            "labeled_at": t_time + timedelta(days=1),
        },
        # Conflicting account-level fraud incident active at t_time
        {
            "label_id": uuid.uuid4(),
            "txn_id": None,
            "account_id": "acc-victim",
            "label_target": "account",
            "is_fraud": True,
            "fraud_type": "account_takeover",
            "confidence": 0.90,
            "labeled_at": t_time + timedelta(days=2),
            "incident_start_at": t_time - timedelta(days=1),
            "incident_end_at": t_time + timedelta(days=1),
        },
    ])

    pit_df, sup_df, out_df = build_supervised_training_dataset(txns, labels)

    assert len(sup_df) == 0  # Excluded from supervised training loss
    assert len(pit_df) == 1
    assert pd.isna(pit_df["target"].iloc[0])
    assert pit_df["primary_outcome"].iloc[0] == "direct_conflict"
    assert pit_df["reason"].iloc[0] == "direct_clean_vs_account_fraud_conflict"


def test_active_account_overlap_contradiction():
    """Simultaneous active fraud and clean incidents on account resolve to target=None (temporal_contradiction)."""
    t_time = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    txns = pd.DataFrame([{
        "txn_id": "txn-overlap-1",
        "account_id": "acc-overlap",
        "occurred_at": t_time,
        "amount": 1000.0,
    }])
    labels = pd.DataFrame([
        {
            "label_id": uuid.uuid4(),
            "txn_id": None,
            "account_id": "acc-overlap",
            "label_target": "account",
            "is_fraud": True,
            "fraud_type": "structuring",
            "confidence": 0.90,
            "labeled_at": t_time + timedelta(days=1),
            "incident_start_at": t_time - timedelta(days=1),
            "incident_end_at": t_time + timedelta(days=1),
        },
        {
            "label_id": uuid.uuid4(),
            "txn_id": None,
            "account_id": "acc-overlap",
            "label_target": "account",
            "is_fraud": False,
            "fraud_type": None,
            "confidence": 1.0,
            "labeled_at": t_time + timedelta(days=1),
            "incident_start_at": t_time - timedelta(days=1),
            "incident_end_at": t_time + timedelta(days=1),
        },
    ])

    pit_df, sup_df, out_df = build_supervised_training_dataset(txns, labels)

    assert len(sup_df) == 0
    assert pd.isna(pit_df["target"].iloc[0])
    assert pit_df["primary_outcome"].iloc[0] == "temporal_contradiction"


def test_account_fraud_window_without_direct_label_defaults_unresolved():
    """Revision 4 invariant: transaction in account fraud window without direct label defaults to Y=None."""
    t_time = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    txns = pd.DataFrame([{
        "txn_id": "txn-mule-1",
        "account_id": "acc-mule",
        "occurred_at": t_time,
        "amount": 80000.0,
    }])
    labels = pd.DataFrame([{
        "label_id": uuid.uuid4(),
        "txn_id": None,
        "account_id": "acc-mule",
        "label_target": "account",
        "is_fraud": True,
        "fraud_type": "money_mule",
        "confidence": 0.95,
        "labeled_at": t_time + timedelta(days=2),
        "incident_start_at": t_time - timedelta(days=1),
        "incident_end_at": t_time + timedelta(days=1),
    }])

    # 1. Default policy (propagate_account_fraud=False) -> Y=None, unresolved
    pit_df, sup_df, out_df = build_supervised_training_dataset(txns, labels, propagate_account_fraud=False)
    assert len(sup_df) == 0
    assert pd.isna(pit_df["target"].iloc[0])
    assert pit_df["primary_outcome"].iloc[0] == "unresolved_account_fraud_window"
    assert pit_df["reason"].iloc[0] == "account_incident_lacks_direct_txn_label"

    # 2. Configurable propagation (propagate_account_fraud=True) -> Y=1
    pit_df_prop, sup_df_prop, out_df_prop = build_supervised_training_dataset(txns, labels, propagate_account_fraud=True)
    assert len(sup_df_prop) == 1
    assert sup_df_prop["target"].iloc[0] == 1
    assert sup_df_prop["target_typology"].iloc[0] == "money_mule"


def test_account_clean_propagation_and_maturity_buffer():
    """Account clean label propagates Y=0, but transactions in maturity buffer are excluded."""
    eval_time = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
    # Txn 1: Mature transaction (30 days before cutoff)
    t1_time = eval_time - timedelta(days=30)
    # Txn 2: Immature transaction (within 7-day maturity buffer)
    t2_time = eval_time - timedelta(days=3)

    txns = pd.DataFrame([
        {"txn_id": "txn-mature", "account_id": "acc-clean", "occurred_at": t1_time, "amount": 100.0},
        {"txn_id": "txn-immature", "account_id": "acc-clean", "occurred_at": t2_time, "amount": 200.0},
    ])
    labels = pd.DataFrame([{
        "label_id": uuid.uuid4(),
        "txn_id": None,
        "account_id": "acc-clean",
        "label_target": "account",
        "is_fraud": False,
        "fraud_type": None,
        "confidence": 1.0,
        "labeled_at": eval_time,
    }])

    pit_df, sup_df, out_df = build_supervised_training_dataset(
        txns, labels, evaluation_time=eval_time, maturity_buffer=timedelta(days=7)
    )

    # Mature transaction resolved to clean control
    row_mature = pit_df[pit_df["txn_id"] == "txn-mature"].iloc[0]
    assert row_mature["target"] == 0
    assert row_mature["primary_outcome"] == "negative_clean"

    # Immature transaction excluded by maturity buffer
    row_immature = pit_df[pit_df["txn_id"] == "txn-immature"].iloc[0]
    assert pd.isna(row_immature["target"])
    assert row_immature["primary_outcome"] == "immature"
    assert "maturity_buffer" in row_immature["reason"]

    # Supervised split contains only the mature record
    assert len(sup_df) == 1
    assert sup_df["txn_id"].iloc[0] == "txn-mature"


def test_unreviewed_transaction_exclusion():
    """Unlabeled transactions must never be imputed as negative; target=None."""
    t_time = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    txns = pd.DataFrame([{
        "txn_id": "txn-unlabeled",
        "account_id": "acc-unknown",
        "occurred_at": t_time,
        "amount": 5000.0,
    }])
    labels = pd.DataFrame()  # Zero labels

    pit_df, sup_df, out_df = build_supervised_training_dataset(txns, labels)

    assert len(sup_df) == 0  # Never imputed as negative control
    assert len(pit_df) == 1
    assert pd.isna(pit_df["target"].iloc[0])
    assert pit_df["primary_outcome"].iloc[0] == "unreviewed_or_subthreshold"


# =====================================================================
# 5. INTEGRITY, SHUFFLE INVARIANCE & FULL MANIFEST INTEGRATION (P0-1)
# =====================================================================

def test_shuffled_input_order_invariance():
    """Verify input transaction shuffle produces identical feature values and targets (P0-1)."""
    t_base = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    txns = pd.DataFrame([
        {"txn_id": "t-1", "account_id": "A1", "occurred_at": t_base + timedelta(hours=1), "amount": 100.0},
        {"txn_id": "t-2", "account_id": "A1", "occurred_at": t_base + timedelta(hours=2), "amount": 200.0},
        {"txn_id": "t-3", "account_id": "A2", "occurred_at": t_base + timedelta(hours=3), "amount": 300.0},
    ])
    labels = pd.DataFrame([
        {
            "label_id": uuid.uuid4(),
            "txn_id": "t-2",
            "account_id": None,
            "label_target": "transaction",
            "is_fraud": True,
            "fraud_type": "card_not_present",
            "confidence": 0.95,
            "labeled_at": t_base + timedelta(days=1),
        }
    ])

    # 1. Forward run
    pit_fwd, sup_fwd, _ = build_supervised_training_dataset(txns, labels)

    # 2. Shuffled reverse run
    txns_shuffled = txns.iloc[::-1].reset_index(drop=True)
    pit_rev, sup_rev, _ = build_supervised_training_dataset(txns_shuffled, labels)

    # Assert canonical ordering and identical features & targets
    assert pit_fwd["txn_id"].tolist() == pit_rev["txn_id"].tolist()
    assert (pit_fwd["target"].fillna(-999).values == pit_rev["target"].fillna(-999).values).all()
    for col in FEATURE_LIST:
        np.testing.assert_array_equal(pit_fwd[col].values, pit_rev[col].values)


def test_duplicate_txn_id_rejected():
    """Verify duplicate txn_id in input transactions raises ValueError."""
    t_time = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    dup_txns = pd.DataFrame([
        {"txn_id": "t-dup", "account_id": "A1", "occurred_at": t_time, "amount": 100.0},
        {"txn_id": "t-dup", "account_id": "A1", "occurred_at": t_time + timedelta(minutes=1), "amount": 200.0},
    ])
    labels = pd.DataFrame()

    with pytest.raises(ValueError, match="duplicate txn_id"):
        build_supervised_training_dataset(dup_txns, labels)


def test_full_synthetic_manifest_integration():
    """End-to-end integration test running on the canonical 6-scenario manifest."""
    manifest = ManifestGenerator(seed=SEED).generate()

    all_txns = [t.model_dump() for sc in manifest.scenarios for t in sc.transactions]
    all_labels = [l.model_dump() for sc in manifest.scenarios for l in sc.labels]
    all_accounts = [a.model_dump() for sc in manifest.scenarios for a in sc.accounts]

    txns_df = pd.DataFrame(all_txns)
    labels_df = pd.DataFrame(all_labels)
    accounts_df = pd.DataFrame(all_accounts)

    pit_df, sup_df, out_df = build_supervised_training_dataset(
        transactions=txns_df,
        labels=labels_df,
        accounts=accounts_df,
        evaluation_time=OBSERVATION_CUTOFF,
        maturity_buffer=timedelta(days=7),
    )

    # 1. Total row conservation
    assert len(pit_df) == 405
    assert len(out_df) == 405
    assert len(sup_df) == 174

    # 2. Target distributions
    assert (sup_df["target"] == 1).sum() == 41
    assert (sup_df["target"] == 0).sum() == 133
    assert pit_df["target"].isna().sum() == 231

    # 3. Supervised split data hygiene
    assert sup_df["target"].notna().all()
    assert set(sup_df["target"].unique()) == {0, 1}
    assert sup_df["target"].dtype == int

    # 4. Feature completeness: All 19 canonical features must have non-null values
    for feat in FEATURE_LIST:
        assert feat in pit_df.columns, f"Missing canonical feature: {feat}"
        assert pit_df[feat].notna().all(), f"Feature {feat} contains NaN values"

    # 5. Invariant validation
    verify_invariants(txns_df, out_df)


# =====================================================================
# 6. CORRECTIVE REGRESSION TESTS (T-01 - T-05, F-04, F-05)
# =====================================================================

def test_t01_direct_fraud_vs_direct_clean_conflict_fails_closed():
    """T-01 / F-01: Direct fraud vs direct clean on same txn must fail closed to direct_conflict."""
    t_time = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    txns = pd.DataFrame([{
        "txn_id": "txn-dual-direct",
        "account_id": "acc-1",
        "occurred_at": t_time,
        "amount": 5000.0,
    }])
    labels = pd.DataFrame([
        {
            "label_id": uuid.uuid4(),
            "txn_id": "txn-dual-direct",
            "account_id": None,
            "label_target": "transaction",
            "is_fraud": True,
            "fraud_type": "card_not_present",
            "confidence": 0.95,
            "labeled_at": t_time + timedelta(days=1),
        },
        {
            "label_id": uuid.uuid4(),
            "txn_id": "txn-dual-direct",
            "account_id": None,
            "label_target": "transaction",
            "is_fraud": False,
            "fraud_type": None,
            "confidence": 1.0,
            "labeled_at": t_time + timedelta(days=1),
        },
    ])

    pit_df, sup_df, out_df = build_supervised_training_dataset(txns, labels)

    # Must be excluded from supervised training loss
    assert len(sup_df) == 0
    # Must be retained in pit_features with null target and direct_conflict outcome
    assert len(pit_df) == 1
    assert pd.isna(pit_df["target"].iloc[0])
    assert pd.isna(pit_df["target_typology"].iloc[0])
    assert pit_df["primary_outcome"].iloc[0] == "direct_conflict"
    assert pit_df["reason"].iloc[0] == "direct_fraud_vs_direct_clean_conflict"


def test_t02_future_label_exclusion():
    """T-02: Labels with labeled_at > evaluation_time must be excluded from resolution."""
    eval_time = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    t_time = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    txns = pd.DataFrame([{
        "txn_id": "txn-future-label",
        "account_id": "acc-1",
        "occurred_at": t_time,
        "amount": 1000.0,
    }])
    # Label timestamped after the evaluation cutoff
    labels = pd.DataFrame([{
        "label_id": uuid.uuid4(),
        "txn_id": "txn-future-label",
        "account_id": None,
        "label_target": "transaction",
        "is_fraud": True,
        "fraud_type": "card_not_present",
        "confidence": 0.95,
        "labeled_at": eval_time + timedelta(days=1),  # In the future relative to eval_time
    }])

    pit_df, sup_df, out_df = build_supervised_training_dataset(
        txns, labels, evaluation_time=eval_time
    )

    # Future label ignored; transaction not labeled as positive fraud
    assert len(sup_df) == 0
    assert pd.isna(pit_df["target"].iloc[0])
    assert pit_df["primary_outcome"].iloc[0] == "unreviewed_or_subthreshold"


def test_t03_reject_timezone_naive_timestamps():
    """T-03 / F-02: Strict timezone-awareness validation on txns, labels, snapshots, and incident windows."""
    # 1. Naive transaction timestamp
    txns_naive = pd.DataFrame([{
        "txn_id": "t-1",
        "account_id": "a-1",
        "occurred_at": "2026-02-01 10:00:00",  # Naive string
        "amount": 100.0,
    }])
    with pytest.raises(ValueError, match="Timezone-naive.*rejected"):
        build_supervised_training_dataset(txns_naive, pd.DataFrame())

    # 2. Naive label labeled_at timestamp
    t_aware = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    txns_aware = pd.DataFrame([{
        "txn_id": "t-1",
        "account_id": "a-1",
        "occurred_at": t_aware,
        "amount": 100.0,
    }])
    labels_naive_labeled = pd.DataFrame([{
        "label_id": uuid.uuid4(),
        "txn_id": "t-1",
        "account_id": None,
        "label_target": "transaction",
        "is_fraud": True,
        "fraud_type": "card_not_present",
        "confidence": 0.95,
        "labeled_at": "2026-02-02 12:00:00",  # Naive
    }])
    with pytest.raises(ValueError, match="Timezone-naive.*rejected"):
        build_supervised_training_dataset(txns_aware, labels_naive_labeled)

    # 3. Naive incident_start_at / incident_end_at
    labels_naive_incident = pd.DataFrame([{
        "label_id": uuid.uuid4(),
        "txn_id": None,
        "account_id": "a-1",
        "label_target": "account",
        "is_fraud": True,
        "fraud_type": "money_mule",
        "confidence": 0.95,
        "labeled_at": t_aware,
        "incident_start_at": "2026-01-01 00:00:00",  # Naive
        "incident_end_at": t_aware,
    }])
    with pytest.raises(ValueError, match="Timezone-naive.*rejected"):
        build_supervised_training_dataset(txns_aware, labels_naive_incident)

    # 4. Naive graph snapshot timestamp
    snaps_naive = pd.DataFrame([{
        "account_id": "a-1",
        "snapshot_at": "2026-02-01 00:00:00",  # Naive
        "component_size": 2.0,
    }])
    with pytest.raises(ValueError, match="Timezone-naive.*rejected"):
        resolve_graph_snapshots(snaps_naive)

    with pytest.raises(ValueError, match="Timezone-naive.*rejected"):
        build_supervised_training_dataset(txns_aware, pd.DataFrame(), graph_snapshots=snaps_naive)


def test_t04_account_opening_horizon():
    """T-04: Transactions occurring before account opened_on cannot be labeled clean."""
    t_open = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
    t_before = t_open - timedelta(days=5)
    t_after = t_open + timedelta(days=5)

    txns = pd.DataFrame([
        {"txn_id": "t-before", "account_id": "acc-open", "occurred_at": t_before, "amount": 100.0},
        {"txn_id": "t-after", "account_id": "acc-open", "occurred_at": t_after, "amount": 200.0},
    ])
    accounts = pd.DataFrame([
        {"account_id": "acc-open", "opened_on": t_open}
    ])
    labels = pd.DataFrame([{
        "label_id": uuid.uuid4(),
        "txn_id": None,
        "account_id": "acc-open",
        "label_target": "account",
        "is_fraud": False,
        "fraud_type": None,
        "confidence": 1.0,
        "labeled_at": t_open + timedelta(days=30),
    }])

    eval_time = t_open + timedelta(days=30)
    pit_df, sup_df, out_df = build_supervised_training_dataset(
        txns, labels, accounts=accounts, evaluation_time=eval_time, maturity_buffer=timedelta(days=7)
    )

    # Transaction before opening -> excluded from supervised, reason: occurred_before_account_opened
    row_before = pit_df[pit_df["txn_id"] == "t-before"].iloc[0]
    assert pd.isna(row_before["target"])
    assert row_before["primary_outcome"] == "unreviewed_or_subthreshold"
    assert row_before["reason"] == "occurred_before_account_opened"

    # Transaction after opening -> resolved as negative clean control
    row_after = pit_df[pit_df["txn_id"] == "t-after"].iloc[0]
    assert row_after["target"] == 0
    assert row_after["primary_outcome"] == "negative_clean"


def test_t05_target_domain_invariant_enforcement():
    """T-05 / F-03: verify_invariants strictly rejects target values outside {0, 1, None/NaN}."""
    txns = pd.DataFrame([{"txn_id": "t-1", "account_id": "a-1"}])

    base_outcome = {
        "txn_id": "t-1",
        "account_id": "a-1",
        "target_typology": None,
        "primary_outcome": "unreviewed_or_subthreshold",
        "reason": "r",
    }

    # 1. Invalid integer: 2
    out_2 = pd.DataFrame([{**base_outcome, "target": 2}])
    with pytest.raises(ValueError, match="out of domain"):
        verify_invariants(txns, out_2)

    # 2. Invalid integer: -1
    out_neg = pd.DataFrame([{**base_outcome, "target": -1}])
    with pytest.raises(ValueError, match="out of domain"):
        verify_invariants(txns, out_neg)

    # 3. Invalid float: 0.5
    out_half = pd.DataFrame([{**base_outcome, "target": 0.5}])
    with pytest.raises(ValueError, match="out of domain"):
        verify_invariants(txns, out_half)

    # 4. Invalid non-numeric string: 'fraud'
    out_str = pd.DataFrame([{**base_outcome, "target": "fraud"}])
    with pytest.raises(ValueError, match="invalid; must be numeric"):
        verify_invariants(txns, out_str)

    # 5. Invalid boolean: True
    out_bool = pd.DataFrame([{**base_outcome, "target": True}])
    with pytest.raises(ValueError, match="Boolean target.*invalid"):
        verify_invariants(txns, out_bool)

    # 6. Valid targets (0, 1, None, np.nan) continue to pass
    out_valid_1 = pd.DataFrame([{
        "txn_id": "t-1",
        "account_id": "a-1",
        "target": 1,
        "target_typology": "card_not_present",
        "primary_outcome": "positive_fraud",
        "reason": "r",
    }])
    verify_invariants(txns, out_valid_1)

    out_valid_0 = pd.DataFrame([{
        "txn_id": "t-1",
        "account_id": "a-1",
        "target": 0,
        "target_typology": "legitimate_clean",
        "primary_outcome": "negative_clean",
        "reason": "r",
    }])
    verify_invariants(txns, out_valid_0)

    out_valid_none = pd.DataFrame([{
        "txn_id": "t-1",
        "account_id": "a-1",
        "target": None,
        "target_typology": None,
        "primary_outcome": "unreviewed_or_subthreshold",
        "reason": "r",
    }])
    verify_invariants(txns, out_valid_none)


def test_f04_transaction_id_validation_and_trimming():
    """F-04: Null, NaN, pd.NA, and whitespace-only txn_id rejected; whitespace trimmed."""
    t_time = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    base_txn = {"account_id": "a-1", "occurred_at": t_time, "amount": 100.0}

    # 1. Null / None
    with pytest.raises(ValueError, match="txn_id cannot be null"):
        build_supervised_training_dataset(pd.DataFrame([{**base_txn, "txn_id": None}]), pd.DataFrame())

    # 2. NaN
    with pytest.raises(ValueError, match="txn_id cannot be null"):
        build_supervised_training_dataset(pd.DataFrame([{**base_txn, "txn_id": np.nan}]), pd.DataFrame())

    # 3. pd.NA
    with pytest.raises(ValueError, match="txn_id cannot be null"):
        build_supervised_training_dataset(pd.DataFrame([{**base_txn, "txn_id": pd.NA}]), pd.DataFrame())

    # 4. Empty string
    with pytest.raises(ValueError, match="txn_id cannot be empty"):
        build_supervised_training_dataset(pd.DataFrame([{**base_txn, "txn_id": ""}]), pd.DataFrame())

    # 5. Whitespace-only string
    with pytest.raises(ValueError, match="txn_id cannot be empty"):
        build_supervised_training_dataset(pd.DataFrame([{**base_txn, "txn_id": "   \t  "}]), pd.DataFrame())

    # 6. Valid ID with surrounding whitespace -> trimmed and preserved
    txns_padded = pd.DataFrame([{**base_txn, "txn_id": "  txn-valid-trimmed  "}])
    pit_df, _, out_df = build_supervised_training_dataset(txns_padded, pd.DataFrame())
    assert pit_df["txn_id"].iloc[0] == "txn-valid-trimmed"
    assert out_df["txn_id"].iloc[0] == "txn-valid-trimmed"


def test_f05_simultaneous_timeless_account_fraud_and_clean_global_contradiction():
    """F-05: Account with simultaneous timeless fraud and timeless clean receives global_contradiction."""
    t_time = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    txns = pd.DataFrame([{
        "txn_id": "txn-timeless-conflict",
        "account_id": "acc-contradictory",
        "occurred_at": t_time,
        "amount": 2500.0,
    }])
    labels = pd.DataFrame([
        # Timeless account fraud
        {
            "label_id": uuid.uuid4(),
            "txn_id": None,
            "account_id": "acc-contradictory",
            "label_target": "account",
            "is_fraud": True,
            "fraud_type": "structuring",
            "confidence": 0.95,
            "labeled_at": t_time + timedelta(days=1),
            "incident_start_at": None,
            "incident_end_at": None,
        },
        # Timeless account clean
        {
            "label_id": uuid.uuid4(),
            "txn_id": None,
            "account_id": "acc-contradictory",
            "label_target": "account",
            "is_fraud": False,
            "fraud_type": None,
            "confidence": 1.0,
            "labeled_at": t_time + timedelta(days=1),
            "incident_start_at": None,
            "incident_end_at": None,
        },
    ])

    pit_df, sup_df, out_df = build_supervised_training_dataset(txns, labels)

    # Excluded from supervised training
    assert len(sup_df) == 0
    # Must receive global_contradiction
    assert len(pit_df) == 1
    assert pd.isna(pit_df["target"].iloc[0])
    assert pd.isna(pit_df["target_typology"].iloc[0])
    assert pit_df["primary_outcome"].iloc[0] == "global_contradiction"
    assert pit_df["reason"].iloc[0] == "account_quarantined_by_contradiction"


def test_f02_optional_incident_boundaries_nat_handling():
    """F-02 edge case: NaT-only optional boundaries accepted; mixed naive rejected; mixed aware accepted."""
    t_aware = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    txns = pd.DataFrame([{
        "txn_id": "txn-nat-test",
        "account_id": "acc-nat",
        "occurred_at": t_aware,
        "amount": 100.0,
    }])

    # 1. All-null optional incident_start_at with dtype datetime64[ns] is accepted
    labels_start_nat = pd.DataFrame([{
        "label_id": uuid.uuid4(),
        "txn_id": None,
        "account_id": "acc-nat",
        "label_target": "account",
        "is_fraud": False,
        "fraud_type": None,
        "confidence": 1.0,
        "labeled_at": t_aware + timedelta(days=1),
        "incident_start_at": pd.NaT,
        "incident_end_at": t_aware + timedelta(days=2),
    }])
    labels_start_nat["incident_start_at"] = labels_start_nat["incident_start_at"].astype("datetime64[ns]")
    pit_df1, _, _ = build_supervised_training_dataset(txns, labels_start_nat)
    assert len(pit_df1) == 1

    # 2. All-null optional incident_end_at with dtype datetime64[ns] is accepted
    labels_end_nat = pd.DataFrame([{
        "label_id": uuid.uuid4(),
        "txn_id": None,
        "account_id": "acc-nat",
        "label_target": "account",
        "is_fraud": False,
        "fraud_type": None,
        "confidence": 1.0,
        "labeled_at": t_aware + timedelta(days=1),
        "incident_start_at": t_aware - timedelta(days=1),
        "incident_end_at": pd.NaT,
    }])
    labels_end_nat["incident_end_at"] = labels_end_nat["incident_end_at"].astype("datetime64[ns]")
    pit_df2, _, _ = build_supervised_training_dataset(txns, labels_end_nat)
    assert len(pit_df2) == 1

    # 3. Direct Series test: All-null datetime64[ns] returns UTC-aware NaT Series
    s_all_nat = pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]")
    res_all_nat = normalize_timestamp_series(s_all_nat, allow_null=True)
    assert str(res_all_nat.dtype) == "datetime64[ns, UTC]"
    assert res_all_nat.isna().all()

    # 4. Mixed Series containing NaT and non-null naive timestamp is rejected
    # 4a. In datetime64[ns] dtype
    s_mixed_naive_dt = pd.Series([pd.NaT, pd.Timestamp("2026-02-01 10:00:00")], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="Timezone-naive datetime64 series is rejected"):
        normalize_timestamp_series(s_mixed_naive_dt, allow_null=True)

    # 4b. In object dtype (naive string)
    s_mixed_naive_str = pd.Series([pd.NaT, "2026-02-01 10:00:00"])
    with pytest.raises(ValueError, match="Timezone-naive timestamp.*rejected"):
        normalize_timestamp_series(s_mixed_naive_str, allow_null=True)

    # 5. Mixed Series containing NaT and timezone-aware timestamp is accepted and normalized to UTC
    s_mixed_aware = pd.Series([pd.NaT, "2026-02-01 10:00:00+00:00"])
    res_mixed_aware = normalize_timestamp_series(s_mixed_aware, allow_null=True)
    assert str(res_mixed_aware.dtype) == "datetime64[ns, UTC]"
    assert pd.isna(res_mixed_aware.iloc[0])
    assert res_mixed_aware.iloc[1] == pd.Timestamp("2026-02-01 10:00:00+00:00")
