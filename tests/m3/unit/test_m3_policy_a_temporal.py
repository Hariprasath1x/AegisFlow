"""Comprehensive tests for AegisFlow M3 Policy A and Design B temporal protocol.

Verifies:
A. Synthetic delay generation bounds [3, 30] and properties.
B. Strict maturity eligibility boundaries (inclusive <=, exactly 30d, older, newer).
C. No outcome-dependent admission (fast reporting cannot change eligibility).
D. No test-period cutoff leakage (training cutoff invariant to test transactions).
E. Exact Design B reconciliation on canonical dataset (590,540 rows, 204,091 mature train, Fit/Val/Cal).
F. Feature and index integrity (preserves 19 canonical M2 features, matrix alignments).
"""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import pytest

from aegis_ml.training.prepare_ieee import canonicalise
from aegis_ml.training.splits import (
    TRAIN_WEEKS,
    TEST_WEEKS,
    check_no_time_leak,
    label_delay_mask,
    strict_maturity_mask,
    time_split,
    training_cutoff,
)
from aegis_ml.training import train_fraud
from aegis_ml.features.definitions import FEATURE_LIST


def sample_transactions() -> pd.DataFrame:
    """Create a minimal valid raw transactions dataframe."""
    return pd.DataFrame(
        {
            "TransactionID": [3000000, 3000001, 3000044, 3000045, 3000099],
            "TransactionDT": [0, 3600, 86400, 172800, 259200],
            "TransactionAmt": [10.50, 20.00, 150.25, 45.00, 1000.00],
            "ProductCD": ["W", "H", "C", "W", "R"],
            "card1": [1001, 1002, 1003, 1004, 1005],
            "card4": ["visa", "mastercard", None, "discover", "visa"],
            "card6": ["debit", "credit", "credit", None, "debit"],
            "addr1": [315, None, 181, 204, None],
            "P_emaildomain": ["gmail.com", None, "yahoo.com", "anonymous.com", "hotmail.com"],
            "isFraud": [0, 1, 0, 1, 1],
        }
    )


# ==============================================================================
# Suite A: Synthetic Delay Generation
# ==============================================================================

def test_synthetic_delay_bounds_and_properties() -> None:
    """Suite A: Generator (suffix % 28) + 3 bounds delay strictly in [3, 30] days."""
    df = sample_transactions()
    # Construct a dataframe with suffixes covering all residues modulo 28
    extra_rows = []
    for r in range(28):
        extra_rows.append(
            {
                "TransactionID": 3001000 + r,
                "TransactionDT": 3600 * (r + 1),
                "TransactionAmt": 100.0 + float(r),
                "ProductCD": "W",
                "card1": 1000 + r,
                "card4": "visa",
                "card6": "debit",
                "addr1": 315,
                "P_emaildomain": "gmail.com",
                "isFraud": 1,
            }
        )
    df_full = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)
    res = canonicalise(df_full)

    # 1. Legitimate reporting timestamps remain missing (NaT)
    legit = res[res["is_fraud"] == 0]
    assert len(legit) > 0
    assert legit["label_reported_at"].isna().all()

    # 2. Fraud reporting timestamps are present
    fraud = res[res["is_fraud"] == 1]
    assert len(fraud) > 0
    assert fraud["label_reported_at"].notna().all()

    # 3. Delays are bounded strictly in [3, 30] days
    delays_days = (fraud["label_reported_at"] - fraud["ts"]).dt.total_seconds() / 86400.0
    assert (delays_days >= 3.0).all()
    assert (delays_days <= 30.0).all()
    assert delays_days.min() == 3.0
    assert delays_days.max() == 30.0

    # 4. No delay exceeds 30 days
    assert (delays_days > 30.0).sum() == 0

    # 5. Ground truth labels remain identical (matched by transaction_id)
    id_to_fraud = dict(zip(df_full["TransactionID"].astype(str), df_full["isFraud"]))
    for tx_id, f in zip(res["transaction_id"], res["is_fraud"]):
        assert id_to_fraud[str(tx_id)] == f


# ==============================================================================
# Suite B: Strict Maturity Eligibility
# ==============================================================================

def test_strict_maturity_boundaries_both_classes() -> None:
    """Suite B: Eligibility depends strictly on transaction timestamp relative to (as_of - 30d)."""
    as_of = pd.Timestamp("2026-04-01 00:00:00", tz="UTC")
    cutoff = as_of - pd.Timedelta(days=30)  # 2026-03-02 00:00:00 UTC

    # Fixture timestamps:
    # 1. Exactly 30 days old (at cutoff) -> eligible
    # 2. 35 days old (before cutoff) -> eligible
    # 3. 29.99999 days old (1 second after cutoff) -> ineligible
    # 4. 10 days old (well after cutoff) -> ineligible
    ts_list = [
        cutoff,
        cutoff - pd.Timedelta(days=5),
        cutoff + pd.Timedelta(seconds=1),
        as_of - pd.Timedelta(days=10),
    ]

    for label in (0, 1):
        mask = strict_maturity_mask(ts_list, as_of=as_of, horizon_days=30)
        assert mask[0] is np.True_ or mask[0] == True  # exactly 30 days old -> eligible
        assert mask[1] is np.True_ or mask[1] == True  # 35 days old -> eligible
        assert mask[2] is np.False_ or mask[2] == False  # 29.99999 days old -> ineligible
        assert mask[3] is np.False_ or mask[3] == False  # 10 days old -> ineligible

        # Test via label_delay_mask consistency
        reported_at = [cutoff + pd.Timedelta(days=3)] * 4 if label == 1 else [pd.NaT] * 4
        m_delay = label_delay_mask(ts_list, reported_at, as_of=as_of, horizon_days=30)
        np.testing.assert_array_equal(mask, m_delay)


# ==============================================================================
# Suite C: No Outcome-Dependent Admission
# ==============================================================================

def test_no_outcome_dependent_admission_under_policy_a() -> None:
    """Suite C: Reporting speed cannot change training eligibility. Immature transactions excluded."""
    as_of = pd.Timestamp("2026-04-01 00:00:00", tz="UTC")
    cutoff = as_of - pd.Timedelta(days=30)

    # Scenarios:
    # 0: Immature fraud with early-arrived report (ts=10d ago, rep=2d ago) -> MUST BE EXCLUDED
    # 1: Immature fraud with future report (ts=10d ago, rep=5d in future) -> MUST BE EXCLUDED
    # 2: Immature legitimate with no report (ts=10d ago, rep=NaT) -> MUST BE EXCLUDED
    # 3: Mature fraud with report arriving exactly at observation cutoff (ts=30d ago, rep=as_of) -> ELIGIBLE
    # 4: Mature fraud with early report (ts=30d ago, rep=as_of - 20d) -> ELIGIBLE
    # 5: Mature legitimate with no report (ts=30d ago, rep=NaT) -> ELIGIBLE
    ts = pd.Series([
        as_of - pd.Timedelta(days=10),
        as_of - pd.Timedelta(days=10),
        as_of - pd.Timedelta(days=10),
        cutoff,
        cutoff,
        cutoff,
    ])
    reported_at = pd.Series([
        as_of - pd.Timedelta(days=2),
        as_of + pd.Timedelta(days=5),
        pd.NaT,
        as_of,
        as_of - pd.Timedelta(days=20),
        pd.NaT,
    ])

    mask = label_delay_mask(ts, reported_at, as_of=as_of, horizon_days=30)

    # Immature transactions (0, 1, 2) are ALL excluded
    assert mask[0] == False
    assert mask[1] == False
    assert mask[2] == False

    # Mature transactions (3, 4, 5) are ALL eligible
    assert mask[3] == True
    assert mask[4] == True
    assert mask[5] == True


# ==============================================================================
# Suite D: No Test-Period Cutoff Leakage
# ==============================================================================

def test_no_test_period_cutoff_leakage() -> None:
    """Suite D: Training cutoff and membership are invariant to test-period timestamps."""
    start = pd.Timestamp("2026-01-05 00:00:00", tz="UTC")

    # 12 weeks of training data (day 0 to day 83)
    train_rows = []
    for d in range(80):
        train_rows.append({
            "transaction_id": f"tx_tr_{d}",
            "ts": start + pd.Timedelta(days=d),
            "account_id": f"acc_{d % 10}",
            "amount": 50.0,
            "channel": "web",
            "card_network": "visa",
            "card_type": "debit",
            "device_type": "desktop",
            "device_id": f"dev_{d % 5}",
            "ip_address": "10.0.0.1",
            "email_domain": "example.com",
            "region": "US",
            "is_fraud": 1 if d % 10 == 0 else 0,
            "label_reported_at": start + pd.Timedelta(days=d + 5) if d % 10 == 0 else pd.NaT,
        })
    df_train = pd.DataFrame(train_rows)

    # Baseline test period (day 84 to day 97: 2 weeks)
    test_rows_base = []
    for d in range(85, 98):
        test_rows_base.append({
            "transaction_id": f"tx_te_{d}",
            "ts": start + pd.Timedelta(days=d),
            "account_id": f"acc_{d % 10}",
            "amount": 50.0,
            "channel": "web",
            "card_network": "visa",
            "card_type": "debit",
            "device_type": "desktop",
            "device_id": f"dev_{d % 5}",
            "ip_address": "10.0.0.1",
            "email_domain": "example.com",
            "region": "US",
            "is_fraud": 0,
            "label_reported_at": pd.NaT,
        })
    df_base = pd.concat([df_train, pd.DataFrame(test_rows_base)], ignore_index=True)

    params = {
        "train_weeks": 12,
        "test_weeks": 2,
        "embargo_hours": 24,
        "label_horizon_days": 30,
        "val_fraction": 0.10,
        "calibration_fraction": 0.15,
    }

    # Cutoff derived exclusively from training boundary
    cutoff1 = training_cutoff(df_base["ts"], params["train_weeks"], params["embargo_hours"])
    prep1 = train_fraud.prepare(params, df=df_base)

    # Now extend the test set with transactions far into the future (day 150)
    future_test_row = {
        "transaction_id": "tx_te_future_150d",
        "ts": start + pd.Timedelta(days=150),
        "account_id": "acc_0",
        "amount": 1000.0,
        "channel": "web",
        "card_network": "visa",
        "card_type": "credit",
        "device_type": "desktop",
        "device_id": "dev_0",
        "ip_address": "10.0.0.1",
        "email_domain": "example.com",
        "region": "US",
        "is_fraud": 0,
        "label_reported_at": pd.NaT,
    }
    df_extended = pd.concat([df_base, pd.DataFrame([future_test_row])], ignore_index=True)

    cutoff2 = training_cutoff(df_extended["ts"], params["train_weeks"], params["embargo_hours"])
    prep2 = train_fraud.prepare(params, df=df_extended)

    # 1. Training cutoff timestamp is strictly invariant
    assert cutoff1 == cutoff2

    # 2. Training row membership and count are strictly invariant
    assert len(prep1["train"]) == len(prep2["train"])
    assert list(prep1["train"]["transaction_id"]) == list(prep2["train"]["transaction_id"])


# ==============================================================================
# Suite E: Exact Design B Dataset Reconciliation
# ==============================================================================

def test_canonical_design_b_dataset_reconciliation() -> None:
    """Suite E: Full canonical dataset reconciles exactly to 590,540 rows and approved partitions."""
    canonical_csv = pathlib.Path("data/raw/ieee/train_transaction.csv")
    if not canonical_csv.exists():
        pytest.skip("Canonical dataset train_transaction.csv not present locally")

    df = pd.read_csv(canonical_csv, usecols=["TransactionID", "isFraud", "TransactionDT"])
    base_date = pd.Timestamp("2017-12-01", tz="UTC")
    ts = base_date + pd.to_timedelta(df["TransactionDT"], unit="s")
    df["ts"] = ts
    df["is_fraud"] = df["isFraud"].astype(int)

    train_weeks = 12
    test_weeks = 2
    embargo_hours = 24
    horizon_days = 30

    # Temporal boundaries
    cut = ts.min() + pd.Timedelta(weeks=train_weeks)
    training_as_of = cut - pd.Timedelta(hours=embargo_hours)
    maturity_boundary = training_as_of - pd.Timedelta(days=horizon_days)

    assert training_as_of == pd.Timestamp("2018-02-23 00:00:00", tz="UTC")
    assert maturity_boundary == pd.Timestamp("2018-01-24 00:00:00", tz="UTC")

    diff_sec = (ts - ts.min()).dt.total_seconds()
    w = (diff_sec // (7 * 86400)).astype(int)

    df["split"] = "drop"
    df.loc[(w < train_weeks) & (ts < training_as_of), "split"] = "train"
    df.loc[(w >= train_weeks) & (w < train_weeks + test_weeks), "split"] = "test"

    mature_train = (df["split"] == "train") & (ts <= maturity_boundary)
    immature_train = (df["split"] == "train") & (ts > maturity_boundary)
    embargo = (w < train_weeks) & (ts >= training_as_of) & (ts < cut)
    test_part = (df["split"] == "test")
    m4_reserve = (w >= train_weeks + test_weeks)

    legit = (df["is_fraud"] == 0)
    fraud = (df["is_fraud"] == 1)

    # 1. Total dataset verification
    assert len(df) == 590540
    assert legit.sum() == 569877
    assert fraud.sum() == 20663

    # 2. Partition counts verification
    assert mature_train.sum() == 204091
    assert (mature_train & legit).sum() == 197946
    assert (mature_train & fraud).sum() == 6145

    assert immature_train.sum() == 90102
    assert (immature_train & legit).sum() == 86393
    assert (immature_train & fraud).sum() == 3709

    assert embargo.sum() == 3350
    assert (embargo & legit).sum() == 3234
    assert (embargo & fraud).sum() == 116

    assert test_part.sum() == 49820
    assert (test_part & legit).sum() == 48024
    assert (test_part & fraud).sum() == 1796

    assert m4_reserve.sum() == 243177
    assert (m4_reserve & legit).sum() == 234280
    assert (m4_reserve & fraud).sum() == 8897

    # 3. Sum of partitions is exactly 590,540
    assert mature_train.sum() + immature_train.sum() + embargo.sum() + test_part.sum() + m4_reserve.sum() == 590540

    # 4. Fit / Val / Cal partition counts
    train_df = df[mature_train].sort_values("ts").reset_index(drop=True)
    n_train = len(train_df)
    cut_val = int(n_train * 0.75)
    cut_cal = int(n_train * 0.85)

    fit = train_df.iloc[:cut_val]
    val = train_df.iloc[cut_val:cut_cal]
    cal = train_df.iloc[cut_cal:]

    assert len(fit) == 153068
    assert (fit["is_fraud"] == 0).sum() == 148981
    assert (fit["is_fraud"] == 1).sum() == 4087

    assert len(val) == 20409
    assert (val["is_fraud"] == 0).sum() == 19613
    assert (val["is_fraud"] == 1).sum() == 796

    assert len(cal) == 30614
    assert (cal["is_fraud"] == 0).sum() == 29352
    assert (cal["is_fraud"] == 1).sum() == 1262


# ==============================================================================
# Suite F: Feature and Index Integrity
# ==============================================================================

def test_feature_and_index_integrity_policy_a() -> None:
    """Suite F: M2 canonical features and index alignment are preserved under Policy A."""
    # Verify the 19 canonical M2 features are defined and untouched
    assert len(FEATURE_LIST) == 19

    # Verify prepare() returns consistent indices
    start = pd.Timestamp("2026-01-05 00:00:00", tz="UTC")
    rows = []
    for d in range(95):
        rows.append({
            "transaction_id": f"tx_{d}",
            "ts": start + pd.Timedelta(days=d),
            "account_id": f"acc_{d % 5}",
            "amount": 25.0 + float(d),
            "channel": "web",
            "card_network": "visa",
            "card_type": "debit",
            "device_type": "desktop",
            "device_id": f"dev_{d % 3}",
            "ip_address": "10.0.0.1",
            "email_domain": "example.com",
            "region": "US",
            "is_fraud": 1 if d % 8 == 0 else 0,
            "label_reported_at": start + pd.Timedelta(days=d + 4) if d % 8 == 0 else pd.NaT,
        })
    df = pd.DataFrame(rows)

    params = {
        "train_weeks": 12,
        "test_weeks": 2,
        "embargo_hours": 24,
        "label_horizon_days": 30,
        "val_fraction": 0.10,
        "calibration_fraction": 0.15,
    }

    res = train_fraud.prepare(params, df=df)

    # 1. Row counts between feature matrices and label vectors match exactly
    assert len(res["X_fit"]) == len(res["y_fit"])
    assert len(res["X_val"]) == len(res["y_val"])
    assert len(res["X_cal"]) == len(res["y_cal"])
    assert len(res["X_test"]) == len(res["y_test"])

    # 2. Check no time leak between train and test
    problems = check_no_time_leak(res["train"]["ts"], res["test"]["ts"])
    assert problems == []
