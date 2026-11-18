"""Unit tests for train_fraud pipeline: splits, preparation, evaluation, and gating."""
from __future__ import annotations

import json
import pathlib
import numpy as np
import pandas as pd
import pytest
import yaml

from aegis_ml.training import train_fraud
from aegis_ml.training.splits import TRAIN_WEEKS, TEST_WEEKS


@pytest.fixture
def fraud_params() -> dict:
    """Standard test fraud configuration."""
    return {
        "seed": 42,
        "train_weeks": 6,
        "test_weeks": 2,
        "embargo_hours": 24,
        "label_horizon_days": 30,
        "val_fraction": 0.10,
        "calibration_fraction": 0.15,
        "max_fpr": 0.01,
        "review_cost": 3.0,
        "friction_cost": 12.0,
        "xgboost": {
            "n_estimators": 5,
            "max_depth": 2,
            "early_stopping_rounds": 2,
            "seed": 42,
            "n_jobs": 1,
        },
        "lightgbm": {
            "n_estimators": 5,
            "num_leaves": 8,
            "min_child_samples": 2,
            "early_stopping_rounds": 2,
            "seed": 42,
            "n_jobs": 1,
        },
    }


@pytest.fixture
def synthetic_transactions() -> pd.DataFrame:
    """Synthetic transactions table spanning 8 full weeks."""
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2026-01-05 00:00:00", tz="UTC")
    n_rows = 600
    # Evenly spaced over 56 days (8 weeks)
    ts_list = [start + pd.Timedelta(hours=i * 2.2) for i in range(n_rows)]

    amounts = np.round(rng.exponential(100.0, size=n_rows) + 5.0, 2)
    channels = rng.choice(["web", "mobile", "pos", "api"], size=n_rows)
    card_nets = rng.choice(["visa", "mastercard"], size=n_rows)
    card_types = rng.choice(["credit", "debit"], size=n_rows)
    dev_types = rng.choice(["ios", "android", "windows"], size=n_rows)
    domains = rng.choice(["gmail.com", "yahoo.com", "corp.com"], size=n_rows)
    regions = rng.choice(["US", "EU", "APAC"], size=n_rows)
    acct_ids = [f"acct_{i % 50}" for i in range(n_rows)]
    tx_ids = [f"tx_{i:04d}" for i in range(n_rows)]

    # ~5% fraud rate
    is_fraud = (rng.random(size=n_rows) < 0.05).astype(int)
    # Ensure at least some frauds in each chronological partition: fit, val, cal, test
    is_fraud[5] = 1
    is_fraud[100] = 1
    is_fraud[220] = 1
    is_fraud[260] = 1
    is_fraud[n_rows - 50] = 1

    # Reporting delay: frauds reported 5-15 days later
    reported_at = []
    for t, f in zip(ts_list, is_fraud):
        if f == 1:
            delay = pd.Timedelta(days=float(rng.integers(5, 15)))
            reported_at.append(t + delay)
        else:
            reported_at.append(pd.NaT)

    return pd.DataFrame({
        "transaction_id": tx_ids,
        "ts": ts_list,
        "account_id": acct_ids,
        "amount": amounts,
        "channel": channels,
        "card_network": card_nets,
        "card_type": card_types,
        "device_type": dev_types,
        "email_domain": domains,
        "region": regions,
        "is_fraud": is_fraud,
        "label_reported_at": pd.to_datetime(reported_at, utc=True),
    })


# --- 1. Pipeline Prepare Tests ---


def test_prepare_splits_and_embargo(fraud_params, synthetic_transactions):
    """prepare correctly creates fit, val, cal, and test splits with embargo."""
    prepared = train_fraud.prepare(fraud_params, df=synthetic_transactions)

    train = prepared["train"]
    test = prepared["test"]
    X_fit = prepared["X_fit"]
    y_fit = prepared["y_fit"]
    X_val = prepared["X_val"]
    y_val = prepared["y_val"]
    X_cal = prepared["X_cal"]
    y_cal = prepared["y_cal"]
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]

    # No time leakage: max train timestamp < min test timestamp
    assert train["ts"].max() < test["ts"].min()

    # Embargo gap: at least embargo_hours between train and test
    embargo_delta = test["ts"].min() - train["ts"].max()
    assert embargo_delta >= pd.Timedelta(hours=fraud_params["embargo_hours"])

    # Chronological partition slices: fit (75%), val (10%), cal (15%)
    total_train_rows = len(train)
    val_frac = fraud_params["val_fraction"]
    cal_frac = fraud_params["calibration_fraction"]
    cut_val = int(total_train_rows * (1.0 - val_frac - cal_frac))
    cut_cal = int(total_train_rows * (1.0 - cal_frac))
    expected_fit_count = cut_val
    expected_val_count = cut_cal - cut_val
    expected_cal_count = total_train_rows - cut_cal

    assert len(X_fit) == expected_fit_count
    assert len(y_fit) == expected_fit_count
    assert len(X_val) == expected_val_count
    assert len(y_val) == expected_val_count
    assert len(X_cal) == expected_cal_count
    assert len(y_cal) == expected_cal_count
    assert len(X_test) == len(test)
    assert len(y_test) == len(test)

    # Feature shapes must match declared feature names
    from aegis_ml.training.matrix import feature_names
    cols = feature_names()
    assert list(X_fit.columns) == cols
    assert list(X_val.columns) == cols
    assert list(X_cal.columns) == cols
    assert list(X_test.columns) == cols

    # Mapping must be present and applied to test
    assert isinstance(prepared["mapping"], dict)


def test_prepare_time_leak_raises(fraud_params, synthetic_transactions, monkeypatch):
    """prepare raises AssertionError if timestamps overlap."""
    monkeypatch.setattr(
        train_fraud,
        "check_no_time_leak",
        lambda tr, te: ["train ends after test starts"],
    )
    with pytest.raises(AssertionError, match="time leakage: train ends after test starts"):
        train_fraud.prepare(fraud_params, df=synthetic_transactions)


def test_prepare_missing_dataset_raises(fraud_params, monkeypatch):
    """prepare raises FileNotFoundError when default load_transactions path is missing."""
    monkeypatch.setattr(
        train_fraud,
        "load_transactions",
        lambda: (_ for _ in ()).throw(FileNotFoundError("ieee_transactions.parquet missing")),
    )
    with pytest.raises(FileNotFoundError, match="missing"):
        train_fraud.prepare(fraud_params)


# --- 1b. Calibration Isolation & Partitioning Regression Tests ---


def test_four_partitions_chronological_order(fraud_params, synthetic_transactions):
    """1. Prove chronological ordering across all four partitions."""
    prepared = train_fraud.prepare(fraud_params, df=synthetic_transactions)
    train = prepared["train"]
    test = prepared["test"]

    ts_fit = train.loc[prepared["X_fit"].index, "ts"]
    ts_val = train.loc[prepared["X_val"].index, "ts"]
    ts_cal = train.loc[prepared["X_cal"].index, "ts"]
    ts_test = test.loc[prepared["X_test"].index, "ts"]

    # Monotonicity within each partition
    assert ts_fit.is_monotonic_increasing
    assert ts_val.is_monotonic_increasing
    assert ts_cal.is_monotonic_increasing
    assert ts_test.is_monotonic_increasing

    # Strict chronological ordering across partition boundaries
    assert ts_fit.max() <= ts_val.min()
    assert ts_val.max() <= ts_cal.min()
    assert ts_cal.max() < ts_test.min()

    # Embargo gap between train/calibration and test
    embargo_gap = ts_test.min() - ts_cal.max()
    assert embargo_gap >= pd.Timedelta(hours=fraud_params["embargo_hours"])


def test_four_partitions_pairwise_disjoint_indices(fraud_params, synthetic_transactions):
    """2. Prove that X_fit, X_val, X_cal, and X_test are pairwise disjoint index sets."""
    prepared = train_fraud.prepare(fraud_params, df=synthetic_transactions)

    idx_fit = set(prepared["X_fit"].index)
    idx_val = set(prepared["X_val"].index)
    idx_cal = set(prepared["X_cal"].index)
    idx_test = set(prepared["X_test"].index)

    # Pairwise disjoint assertions
    assert idx_fit.isdisjoint(idx_val), "X_fit and X_val must be disjoint"
    assert idx_fit.isdisjoint(idx_cal), "X_fit and X_cal must be disjoint"
    assert idx_fit.isdisjoint(idx_test), "X_fit and X_test must be disjoint"
    assert idx_val.isdisjoint(idx_cal), "X_val and X_cal must be disjoint"
    assert idx_val.isdisjoint(idx_test), "X_val and X_test must be disjoint"
    assert idx_cal.isdisjoint(idx_test), "X_cal and X_test must be disjoint"

    # Union covers full train and test respectively
    assert (idx_fit | idx_val | idx_cal) == set(prepared["train"].index)
    assert idx_test == set(prepared["test"].index)


def test_early_stopping_and_calibration_slices_are_distinct(fraud_params, synthetic_transactions):
    """3. Explicitly assert that early-stopping validation slice and calibration slice are distinct."""
    prepared = train_fraud.prepare(fraud_params, df=synthetic_transactions)

    X_val = prepared["X_val"]
    X_cal = prepared["X_cal"]

    assert len(X_val) > 0
    assert len(X_cal) > 0
    assert not X_val.index.equals(X_cal.index)
    assert set(X_val.index).isdisjoint(set(X_cal.index))

    # Also verify labels are distinct slices
    y_val = prepared["y_val"]
    y_cal = prepared["y_cal"]
    assert len(y_val) == len(X_val)
    assert len(y_cal) == len(X_cal)


def test_booster_receives_val_partition_spy(
    tmp_path: pathlib.Path,
    fraud_params,
    synthetic_transactions,
    monkeypatch,
):
    """4. Verify that booster training functions receive X_val / y_val as their validation set."""
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.dump({"fraud": fraud_params}))

    reports_dir = tmp_path / "reports"
    artifacts_dir = tmp_path / "artifacts" / "fraud"
    monkeypatch.setattr(train_fraud, "OUT", artifacts_dir)

    real_prepare = train_fraud.prepare
    prepared = real_prepare(fraud_params, df=synthetic_transactions)
    monkeypatch.setattr(train_fraud, "prepare", lambda p: prepared)

    booster_calls = {}

    def spy_train_xgboost(X_train, y_train, X_val, y_val, params=None):
        booster_calls["xgboost"] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "params": params,
        }
        from aegis_ml.models.fraud_models import train_xgboost as real_xgb
        return real_xgb(X_train, y_train, X_val, y_val, params)

    def spy_train_lightgbm(X_train, y_train, X_val, y_val, params=None):
        booster_calls["lightgbm"] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "params": params,
        }
        from aegis_ml.models.fraud_models import train_lightgbm as real_lgb
        return real_lgb(X_train, y_train, X_val, y_val, params)

    monkeypatch.setattr(train_fraud, "train_xgboost", spy_train_xgboost)
    monkeypatch.setattr(train_fraud, "train_lightgbm", spy_train_lightgbm)

    real_write_report = train_fraud.write_report
    monkeypatch.setattr(
        train_fraud,
        "write_report",
        lambda path, title, sections, gates=(), out_json=None: real_write_report(
            path=str(reports_dir / "m3_fraud.html"),
            title=title,
            sections=sections,
            gates=gates,
            out_json=str(reports_dir / "m3_fraud.json"),
        ),
    )

    import mlflow
    mlflow.set_tracking_uri(pathlib.Path(tmp_path / "mlruns").as_uri())

    train_fraud.main(["--params", str(params_path)])

    for name in ("xgboost", "lightgbm"):
        call = booster_calls[name]
        # X_train is X_fit
        assert call["X_train"].index.equals(prepared["X_fit"].index)
        assert np.array_equal(call["y_train"], prepared["y_fit"])
        # X_val is strictly X_val, NOT X_cal
        assert call["X_val"].index.equals(prepared["X_val"].index)
        assert np.array_equal(call["y_val"], prepared["y_val"])
        assert not call["X_val"].index.equals(prepared["X_cal"].index)
        assert set(call["X_val"].index).isdisjoint(set(prepared["X_cal"].index))


def test_calibrator_receives_cal_partition_spy(
    tmp_path: pathlib.Path,
    fraud_params,
    synthetic_transactions,
    monkeypatch,
):
    """5. Verify that isotonic calibration receives only predictions and labels from X_cal."""
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.dump({"fraud": fraud_params}))

    reports_dir = tmp_path / "reports"
    artifacts_dir = tmp_path / "artifacts" / "fraud"
    monkeypatch.setattr(train_fraud, "OUT", artifacts_dir)

    real_prepare = train_fraud.prepare
    prepared = real_prepare(fraud_params, df=synthetic_transactions)
    monkeypatch.setattr(train_fraud, "prepare", lambda p: prepared)

    calibrator_calls = []
    real_calibrator_fit = train_fraud.Calibrator.fit

    def spy_fit(self, scores, y_true):
        calibrator_calls.append({"scores": np.asarray(scores).copy(), "y_true": np.asarray(y_true).copy()})
        return real_calibrator_fit(self, scores, y_true)

    monkeypatch.setattr(train_fraud.Calibrator, "fit", spy_fit)

    real_write_report = train_fraud.write_report
    monkeypatch.setattr(
        train_fraud,
        "write_report",
        lambda path, title, sections, gates=(), out_json=None: real_write_report(
            path=str(reports_dir / "m3_fraud.html"),
            title=title,
            sections=sections,
            gates=gates,
            out_json=str(reports_dir / "m3_fraud.json"),
        ),
    )

    import mlflow
    mlflow.set_tracking_uri(pathlib.Path(tmp_path / "mlruns").as_uri())

    train_fraud.main(["--params", str(params_path)])

    # Fitted once for xgboost, once for lightgbm
    assert len(calibrator_calls) == 2
    for call in calibrator_calls:
        # Number of scores equals size of X_cal, NOT X_val
        assert len(call["scores"]) == len(prepared["X_cal"])
        assert len(call["scores"]) != len(prepared["X_val"])
        # Labels strictly match y_cal
        assert np.array_equal(call["y_true"], prepared["y_cal"])
        # Does NOT match y_val or y_fit or y_test
        assert not np.array_equal(call["y_true"], prepared["y_val"])


def test_test_rows_never_leak_into_fitting_or_calibration(
    tmp_path: pathlib.Path,
    fraud_params,
    synthetic_transactions,
    monkeypatch,
):
    """6. Verify that test rows never reach model fitting, early stopping, or calibration."""
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.dump({"fraud": fraud_params}))

    reports_dir = tmp_path / "reports"
    artifacts_dir = tmp_path / "artifacts" / "fraud"
    monkeypatch.setattr(train_fraud, "OUT", artifacts_dir)

    real_prepare = train_fraud.prepare
    prepared = real_prepare(fraud_params, df=synthetic_transactions)
    monkeypatch.setattr(train_fraud, "prepare", lambda p: prepared)

    test_indices = set(prepared["X_test"].index)

    # Check partition indices directly
    assert set(prepared["X_fit"].index).isdisjoint(test_indices)
    assert set(prepared["X_val"].index).isdisjoint(test_indices)
    assert set(prepared["X_cal"].index).isdisjoint(test_indices)

    # Spy on all model functions to verify no test rows are received
    booster_indices = []

    def spy_train_xgboost(X_train, y_train, X_val, y_val, params=None):
        booster_indices.extend(list(X_train.index) + list(X_val.index))
        from aegis_ml.models.fraud_models import train_xgboost as real_xgb
        return real_xgb(X_train, y_train, X_val, y_val, params)

    def spy_train_lightgbm(X_train, y_train, X_val, y_val, params=None):
        booster_indices.extend(list(X_train.index) + list(X_val.index))
        from aegis_ml.models.fraud_models import train_lightgbm as real_lgb
        return real_lgb(X_train, y_train, X_val, y_val, params)

    monkeypatch.setattr(train_fraud, "train_xgboost", spy_train_xgboost)
    monkeypatch.setattr(train_fraud, "train_lightgbm", spy_train_lightgbm)

    real_write_report = train_fraud.write_report
    monkeypatch.setattr(
        train_fraud,
        "write_report",
        lambda path, title, sections, gates=(), out_json=None: real_write_report(
            path=str(reports_dir / "m3_fraud.html"),
            title=title,
            sections=sections,
            gates=gates,
            out_json=str(reports_dir / "m3_fraud.json"),
        ),
    )

    import mlflow
    mlflow.set_tracking_uri(pathlib.Path(tmp_path / "mlruns").as_uri())

    train_fraud.main(["--params", str(params_path)])

    # Zero overlap with test indices
    assert set(booster_indices).isdisjoint(test_indices)


def test_prepare_invalid_fractions_and_insufficient_rows(fraud_params, synthetic_transactions):
    """7. Verify that invalid fractions and insufficient partition sizes fail safely with ValueError."""
    # Invalid val_fraction
    for bad_val in (0.0, -0.1, 1.0, 1.5, float("nan"), float("inf")):
        p = {**fraud_params, "val_fraction": bad_val}
        with pytest.raises(ValueError, match="val_fraction"):
            train_fraud.prepare(p, df=synthetic_transactions)

    # Invalid calibration_fraction
    for bad_cal in (0.0, -0.1, 1.0, 1.5, float("nan"), float("-inf")):
        p = {**fraud_params, "calibration_fraction": bad_cal}
        with pytest.raises(ValueError, match="calibration_fraction"):
            train_fraud.prepare(p, df=synthetic_transactions)

    # Sum of fractions >= 1.0
    for val_f, cal_f in ((0.5, 0.5), (0.6, 0.5), (0.8, 0.3)):
        p = {**fraud_params, "val_fraction": val_f, "calibration_fraction": cal_f}
        with pytest.raises(ValueError, match="strictly less than 1.0"):
            train_fraud.prepare(p, df=synthetic_transactions)

    # Insufficient rows to populate fit, val, and cal partitions
    tiny_df_2 = pd.concat([synthetic_transactions.iloc[:2], synthetic_transactions.iloc[-2:]], ignore_index=True)
    with pytest.raises(ValueError, match="Insufficient training rows"):
        train_fraud.prepare(fraud_params, df=tiny_df_2)

    tiny_df_1 = pd.concat([synthetic_transactions.iloc[:1], synthetic_transactions.iloc[-2:]], ignore_index=True)
    with pytest.raises(ValueError, match="Insufficient training rows"):
        train_fraud.prepare(fraud_params, df=tiny_df_1)


def test_embargo_and_label_delay_protections_intact(fraud_params, synthetic_transactions):
    """8. Verify that existing embargo (24h) and label-delay protections remain intact."""
    prepared = train_fraud.prepare(fraud_params, df=synthetic_transactions)
    train = prepared["train"]
    test = prepared["test"]

    # 1. 24h embargo check
    embargo_gap = test["ts"].min() - train["ts"].max()
    assert embargo_gap >= pd.Timedelta(hours=fraud_params["embargo_hours"])

    # 2. Label delay check: all frauds in train split must have known labels within horizon
    train_frauds = train[train["is_fraud"] == 1]
    if len(train_frauds) > 0:
        max_ts = synthetic_transactions["ts"].max()
        horizon = pd.Timedelta(days=fraud_params["label_horizon_days"])
        for _, row in train_frauds.iterrows():
            if pd.notna(row["label_reported_at"]):
                assert row["label_reported_at"] <= max_ts
                assert (row["label_reported_at"] - row["ts"]) <= horizon


# --- 1c. Categorical Frequency Encoder Isolation & Class Sufficiency Tests ---


def test_frequency_mapping_fitted_only_from_fitting_partition_spy(
    fraud_params,
    synthetic_transactions,
    monkeypatch,
):
    """1. Prove frequency mapping is fitted only from fitting-partition rows using a spy."""
    fitting_calls = []
    real_encode = train_fraud.encode_categoricals

    def spy_encode_categoricals(df, mapping=None):
        if mapping is None:
            fitting_calls.append({
                "df": df.copy(),
                "indices": list(df.index),
                "columns": list(df.columns),
            })
        return real_encode(df, mapping=mapping)

    monkeypatch.setattr(train_fraud, "encode_categoricals", spy_encode_categoricals)

    prepared = train_fraud.prepare(fraud_params, df=synthetic_transactions)

    # Exactly one fitting call occurred (for X_fit)
    assert len(fitting_calls) == 1
    fit_call = fitting_calls[0]

    # Input to fitting call strictly matches X_fit index and size
    assert fit_call["indices"] == list(prepared["X_fit"].index)
    assert len(fit_call["df"]) == len(prepared["X_fit"])

    # None of the indices from X_val, X_cal, or X_test appear in the fitting input
    val_indices = set(prepared["X_val"].index)
    cal_indices = set(prepared["X_cal"].index)
    test_indices = set(prepared["X_test"].index)

    fit_input_indices = set(fit_call["indices"])
    assert fit_input_indices.isdisjoint(val_indices)
    assert fit_input_indices.isdisjoint(cal_indices)
    assert fit_input_indices.isdisjoint(test_indices)


def test_val_cal_test_rows_do_not_influence_frequency_mapping(
    fraud_params,
    synthetic_transactions,
):
    """2. Prove validation, calibration, and test rows do not influence the fitted mapping."""
    clean_prep = train_fraud.prepare(fraud_params, df=synthetic_transactions)
    val_ts = clean_prep["train"].iloc[len(clean_prep["X_fit"])]["ts"]
    cal_ts = clean_prep["train"].iloc[len(clean_prep["X_fit"]) + len(clean_prep["X_val"])]["ts"]
    test_ts = clean_prep["test"].iloc[0]["ts"]

    probe_df = synthetic_transactions.copy()
    probe_df.loc[probe_df["ts"] == val_ts, "channel"] = "kiosk_val_canary"
    probe_df.loc[probe_df["ts"] == val_ts, "device_type"] = "smart_fridge_val"
    probe_df.loc[probe_df["ts"] == cal_ts, "card_network"] = "rupay_cal_canary"
    probe_df.loc[probe_df["ts"] == cal_ts, "email_domain"] = "canary_cal.org"
    probe_df.loc[probe_df["ts"] == test_ts, "region"] = "MARS_test_canary"

    prepared = train_fraud.prepare(fraud_params, df=probe_df)
    mapping = prepared["mapping"]

    # None of the canary categories from val, cal, or test appear in the fitted mapping
    assert "kiosk_val_canary" not in mapping["channel"]
    assert "smart_fridge_val" not in mapping["device_type"]
    assert "rupay_cal_canary" not in mapping["card_network"]
    assert "canary_cal.org" not in mapping["email_domain"]
    assert "MARS_test_canary" not in mapping["region"]


def test_same_frozen_mapping_applied_across_all_partitions(
    fraud_params,
    synthetic_transactions,
):
    """3. Prove the same frozen mapping is applied to all later partitions."""
    prepared = train_fraud.prepare(fraud_params, df=synthetic_transactions)
    mapping = prepared["mapping"]

    X_fit = prepared["X_fit"]
    X_val = prepared["X_val"]
    X_cal = prepared["X_cal"]
    X_test = prepared["X_test"]

    train_df = prepared["train"]
    test_df = prepared["test"]

    for col in train_fraud.CATEGORICAL:
        feat_col = f"cat__{col}"
        assert feat_col in X_fit.columns
        assert feat_col in X_val.columns
        assert feat_col in X_cal.columns
        assert feat_col in X_test.columns

        # For categories in fit partition, encoded value matches fitted frequency across all partitions
        for cat, expected_freq in mapping[col].items():
            val_mask = train_df.loc[X_val.index, col] == cat
            if val_mask.any():
                np.testing.assert_allclose(
                    X_val.loc[val_mask, feat_col].to_numpy(),
                    expected_freq,
                )
            cal_mask = train_df.loc[X_cal.index, col] == cat
            if cal_mask.any():
                np.testing.assert_allclose(
                    X_cal.loc[cal_mask, feat_col].to_numpy(),
                    expected_freq,
                )
            test_mask = test_df.loc[X_test.index, col] == cat
            if test_mask.any():
                np.testing.assert_allclose(
                    X_test.loc[test_mask, feat_col].to_numpy(),
                    expected_freq,
                )


def test_unseen_categories_fallback_to_zero_across_partitions(
    fraud_params,
    synthetic_transactions,
):
    """4. Prove unseen categories follow the existing fallback behavior (0.0)."""
    clean_prep = train_fraud.prepare(fraud_params, df=synthetic_transactions)
    val_ts = clean_prep["train"].iloc[len(clean_prep["X_fit"])]["ts"]
    cal_ts = clean_prep["train"].iloc[len(clean_prep["X_fit"]) + len(clean_prep["X_val"])]["ts"]
    test_ts = clean_prep["test"].iloc[0]["ts"]

    probe_df = synthetic_transactions.copy()
    probe_df.loc[probe_df["ts"] == val_ts, "channel"] = "unseen_in_fit_val"
    probe_df.loc[probe_df["ts"] == cal_ts, "channel"] = "unseen_in_fit_cal"
    probe_df.loc[probe_df["ts"] == test_ts, "channel"] = "unseen_in_fit_test"

    prepared = train_fraud.prepare(fraud_params, df=probe_df)

    # In X_val, the unseen category gets 0.0
    val_idx = prepared["train"][prepared["train"]["ts"] == val_ts].index[0]
    cal_idx = prepared["train"][prepared["train"]["ts"] == cal_ts].index[0]
    test_idx = prepared["test"][prepared["test"]["ts"] == test_ts].index[0]

    assert prepared["X_val"].loc[val_idx, "cat__channel"] == 0.0
    assert prepared["X_cal"].loc[cal_idx, "cat__channel"] == 0.0
    assert prepared["X_test"].loc[test_idx, "cat__channel"] == 0.0


def test_point_in_time_feature_context_and_ordering_intact(
    fraud_params,
    synthetic_transactions,
):
    """5. Prove point-in-time feature context and feature ordering remain intact."""
    from aegis_ml.training.matrix import feature_names
    cols = feature_names()

    prepared = train_fraud.prepare(fraud_params, df=synthetic_transactions)

    # Fixed canonical feature ordering preserved across all partitions
    assert list(prepared["X_fit"].columns) == cols
    assert list(prepared["X_val"].columns) == cols
    assert list(prepared["X_cal"].columns) == cols
    assert list(prepared["X_test"].columns) == cols

    # All partitions have rows and non-empty features
    assert len(prepared["X_fit"]) > 0
    assert len(prepared["X_val"]) > 0
    assert len(prepared["X_cal"]) > 0
    assert len(prepared["X_test"]) > 0


def test_insufficient_class_distributions_fail_safely(
    fraud_params,
    synthetic_transactions,
):
    """6. Prove insufficient class distributions fail safely in fit, validation, and calibration partitions."""
    clean_prep = train_fraud.prepare(fraud_params, df=synthetic_transactions)
    fit_ts = clean_prep["train"].iloc[:len(clean_prep["X_fit"])]["ts"]
    val_ts = clean_prep["train"].iloc[len(clean_prep["X_fit"]):len(clean_prep["X_fit"]) + len(clean_prep["X_val"])]["ts"]
    cal_ts = clean_prep["train"].iloc[len(clean_prep["X_fit"]) + len(clean_prep["X_val"]):]["ts"]

    # 1. Fitting partition missing positive fraud class (all 0s)
    df_no_fit = synthetic_transactions.copy()
    df_no_fit.loc[df_no_fit["ts"].isin(fit_ts), "is_fraud"] = 0
    with pytest.raises(ValueError, match="Fitting partition does not contain both classes"):
        train_fraud.prepare(fraud_params, df=df_no_fit)

    # 2. Validation partition missing positive fraud class (all 0s)
    df_no_val = synthetic_transactions.copy()
    df_no_val.loc[df_no_val["ts"].isin(val_ts), "is_fraud"] = 0
    with pytest.raises(ValueError, match="Validation partition does not contain both classes"):
        train_fraud.prepare(fraud_params, df=df_no_val)

    # 3. Calibration partition missing positive fraud class (all 0s)
    df_no_cal = synthetic_transactions.copy()
    df_no_cal.loc[df_no_cal["ts"].isin(cal_ts), "is_fraud"] = 0
    with pytest.raises(ValueError, match="Calibration partition does not contain both classes"):
        train_fraud.prepare(fraud_params, df=df_no_cal)


# --- 1d. Point-in-Time Test History Preservation Across Train/Test Boundary ---


@pytest.fixture
def deterministic_train_test_data(fraud_params: dict) -> tuple[dict, pd.DataFrame]:
    """Deterministic transaction history spanning training and test splits.

    Constructs:
    - 100 background training transactions spanning days 0..40 (40 days, within train_weeks=6)
    - 20 background test transactions spanning days 43..53 (within test_weeks=2, after 24h embargo)
    - Class balance: verified fraud labels in fit (day 4), val (day 32), cal (day 38), test (day 45)
    - Target account ACC_HIST:
      - Tx 1 (day 5, fit): amount $80.0, DEV_HIST, 10.0.0.1
      - Tx 2 (day 15, fit): amount $120.0, DEV_HIST, 10.0.0.1
      - Tx 3 (day 38, cal): amount $100.0, DEV_HIST, 10.0.0.1
      - Tx 4 (day 42.5, test, 12h into week 7): amount $150.0, DEV_HIST, 10.0.0.1
    - Other account ACC_OTHER:
      - Tx at day 37 (cal): amount $45.0, sharing DEV_HIST and 10.0.0.1 with ACC_HIST
    """
    params = {
        **fraud_params,
        "label_horizon_days": 0,
    }
    start = pd.Timestamp("2026-01-05 00:00:00", tz="UTC")
    rows = []

    # 100 background training rows spanning days 0..40 (step 0.4 days)
    for i in range(100):
        t = start + pd.Timedelta(days=i * 0.4)
        rows.append({
            "transaction_id": f"bg_tr_{i:03d}",
            "ts": t,
            "account_id": f"bg_acct_{i}",
            "amount": 50.0,
            "channel": "web",
            "card_network": "visa",
            "card_type": "credit",
            "device_type": "desktop",
            "device_id": f"bg_dev_{i}",
            "ip_address": f"192.168.1.{i % 250}",
            "email_domain": "example.com",
            "region": "US",
            "is_fraud": 0,
            "label_reported_at": pd.NaT,
        })

    # Account history target transactions
    rows.append({
        "transaction_id": "tx_hist_1",
        "ts": start + pd.Timedelta(days=5),
        "account_id": "ACC_HIST",
        "amount": 80.0,
        "channel": "web",
        "card_network": "visa",
        "card_type": "credit",
        "device_type": "desktop",
        "device_id": "DEV_HIST",
        "ip_address": "10.0.0.1",
        "email_domain": "example.com",
        "region": "US",
        "is_fraud": 0,
        "label_reported_at": pd.NaT,
    })
    rows.append({
        "transaction_id": "tx_hist_2",
        "ts": start + pd.Timedelta(days=15),
        "account_id": "ACC_HIST",
        "amount": 120.0,
        "channel": "web",
        "card_network": "visa",
        "card_type": "credit",
        "device_type": "desktop",
        "device_id": "DEV_HIST",
        "ip_address": "10.0.0.1",
        "email_domain": "example.com",
        "region": "US",
        "is_fraud": 0,
        "label_reported_at": pd.NaT,
    })
    rows.append({
        "transaction_id": "tx_hist_3",
        "ts": start + pd.Timedelta(days=38),
        "account_id": "ACC_HIST",
        "amount": 100.0,
        "channel": "web",
        "card_network": "visa",
        "card_type": "credit",
        "device_type": "desktop",
        "device_id": "DEV_HIST",
        "ip_address": "10.0.0.1",
        "email_domain": "example.com",
        "region": "US",
        "is_fraud": 0,
        "label_reported_at": pd.NaT,
    })

    # Other account sharing device and IP
    rows.append({
        "transaction_id": "tx_other_1",
        "ts": start + pd.Timedelta(days=37),
        "account_id": "ACC_OTHER",
        "amount": 45.0,
        "channel": "web",
        "card_network": "visa",
        "card_type": "credit",
        "device_type": "desktop",
        "device_id": "DEV_HIST",
        "ip_address": "10.0.0.1",
        "email_domain": "example.com",
        "region": "US",
        "is_fraud": 0,
        "label_reported_at": pd.NaT,
    })

    # Test transaction for ACC_HIST (12 hours into test period / week 7)
    rows.append({
        "transaction_id": "tx_hist_4_test",
        "ts": start + pd.Timedelta(days=42.5),
        "account_id": "ACC_HIST",
        "amount": 150.0,
        "channel": "web",
        "card_network": "visa",
        "card_type": "credit",
        "device_type": "desktop",
        "device_id": "DEV_HIST",
        "ip_address": "10.0.0.1",
        "email_domain": "example.com",
        "region": "US",
        "is_fraud": 0,
        "label_reported_at": pd.NaT,
    })

    # 20 background test rows spanning days 43..52.5
    for i in range(20):
        t = start + pd.Timedelta(days=43 + i * 0.5)
        rows.append({
            "transaction_id": f"bg_test_{i:02d}",
            "ts": t,
            "account_id": f"bg_acct_test_{i}",
            "amount": 50.0,
            "channel": "web",
            "card_network": "visa",
            "card_type": "credit",
            "device_type": "desktop",
            "device_id": f"bg_dev_test_{i}",
            "ip_address": f"192.168.2.{i}",
            "email_domain": "example.com",
            "region": "US",
            "is_fraud": 0,
            "label_reported_at": pd.NaT,
        })

    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    df["label_reported_at"] = pd.to_datetime(df["label_reported_at"], utc=True)

    # Class sufficiency: place mature fraud cases in fit, val, cal, test
    df.loc[10, "is_fraud"] = 1
    df.loc[10, "label_reported_at"] = df.loc[10, "ts"] + pd.Timedelta(days=1)
    df.loc[80, "is_fraud"] = 1
    df.loc[80, "label_reported_at"] = df.loc[80, "ts"] + pd.Timedelta(days=1)
    df.loc[95, "is_fraud"] = 1
    df.loc[95, "label_reported_at"] = df.loc[95, "ts"] + pd.Timedelta(days=1)
    df.loc[110, "is_fraud"] = 1
    df.loc[110, "label_reported_at"] = df.loc[110, "ts"] + pd.Timedelta(days=1)

    return params, df


def test_test_transaction_uses_preceding_training_account_context(deterministic_train_test_data):
    """1. Test transaction preceded by training transactions uses real history for rolling features."""
    params, df = deterministic_train_test_data
    prepared = train_fraud.prepare(params, df=df)

    test = prepared["test"]
    X_test = prepared["X_test"]

    test_idx = test[test["transaction_id"] == "tx_hist_4_test"].index[0]
    row = X_test.loc[test_idx]

    # Preceding tx_hist_3 (day 38, $100) and current tx_hist_4_test (day 42.5, $150)
    assert row["txn_count_7d"] == 2.0
    assert row["amount_sum_minor_7d"] == 25000.0


def test_test_transaction_recency_not_minus_one(deterministic_train_test_data):
    """2. Test transaction recency reflects true time delta from prior training tx and does not reset to -1.0."""
    params, df = deterministic_train_test_data
    prepared = train_fraud.prepare(params, df=df)

    test = prepared["test"]
    X_test = prepared["X_test"]

    test_idx = test[test["transaction_id"] == "tx_hist_4_test"].index[0]
    row = X_test.loc[test_idx]

    # Exactly 4.5 days between tx_hist_3 (day 38) and tx_hist_4_test (day 42.5) = 388,800 seconds
    assert row["seconds_since_prev"] == 388800.0
    assert row["seconds_since_prev"] > 0.0


def test_test_transaction_welford_amount_zscore_retains_state(deterministic_train_test_data):
    """3. Running Welford mean and variance persist across train/test cut; amount_zscore reflects prior state."""
    params, df = deterministic_train_test_data
    prepared = train_fraud.prepare(params, df=df)

    test = prepared["test"]
    X_test = prepared["X_test"]

    test_idx = test[test["transaction_id"] == "tx_hist_4_test"].index[0]
    row = X_test.loc[test_idx]

    # Prior amounts: $80, $120, $100 -> mean=$100 (10,000 minor), sample variance=4,000,000, std=2,000
    # Current amount: $150 (15,000 minor) -> zscore = (15,000 - 10,000) / 2,000 = 2.5
    assert np.isclose(row["amount_zscore"], 2.5)


def test_test_transaction_seven_day_window_includes_eligible_training(deterministic_train_test_data):
    """4. 7-day velocity window includes eligible training tx (day 38) and excludes older ones (day 5, 15)."""
    params, df = deterministic_train_test_data
    prepared = train_fraud.prepare(params, df=df)

    test = prepared["test"]
    X_test = prepared["X_test"]

    test_idx = test[test["transaction_id"] == "tx_hist_4_test"].index[0]
    row = X_test.loc[test_idx]

    # 1h and 24h windows do not reach day 38 (4.5 days prior) -> only the current test tx is included
    assert row["txn_count_1h"] == 1.0
    assert row["amount_sum_minor_1h"] == 15000.0
    assert row["txn_count_24h"] == 1.0
    assert row["amount_sum_minor_24h"] == 15000.0

    # 7-day window reaches day 38 (includes tx_hist_3 and tx_hist_4_test), but not day 15 or day 5
    assert row["txn_count_7d"] == 2.0
    assert row["amount_sum_minor_7d"] == 25000.0


def test_test_transaction_device_ip_sharing_preserves_history(deterministic_train_test_data):
    """5. Device and IP sharing features reflect accounts observed during the training period."""
    params, df = deterministic_train_test_data
    prepared = train_fraud.prepare(params, df=df)

    test = prepared["test"]
    X_test = prepared["X_test"]

    test_idx = test[test["transaction_id"] == "tx_hist_4_test"].index[0]
    row = X_test.loc[test_idx]

    # Both DEV_HIST and 10.0.0.1 were used by ACC_OTHER at day 37 and ACC_HIST at day 38 & 42.5
    # Window (day 35.5, day 42.5] contains 2 distinct accounts: ACC_HIST and ACC_OTHER
    assert row["device_accounts_7d"] == 2.0
    assert row["ip_accounts_7d"] == 2.0


def test_future_test_transactions_do_not_alter_past_features(deterministic_train_test_data):
    """6. Appending later test-period transactions does not alter features of earlier transactions."""
    params, df = deterministic_train_test_data
    prepared1 = train_fraud.prepare(params, df=df)

    test_idx = prepared1["test"][prepared1["test"]["transaction_id"] == "tx_hist_4_test"].index[0]
    row1 = prepared1["X_test"].loc[test_idx]

    # Append future test transaction on day 50 for ACC_HIST with high amount and new identifier
    start = pd.Timestamp("2026-01-05 00:00:00", tz="UTC")
    future_row = {
        "transaction_id": "tx_future_test_50d",
        "ts": start + pd.Timedelta(days=50),
        "account_id": "ACC_HIST",
        "amount": 9999.0,
        "channel": "web",
        "card_network": "visa",
        "card_type": "credit",
        "device_type": "desktop",
        "device_id": "DEV_HIST",
        "ip_address": "10.0.0.1",
        "email_domain": "example.com",
        "region": "US",
        "is_fraud": 0,
        "label_reported_at": pd.NaT,
    }
    df_extended = pd.concat([df, pd.DataFrame([future_row])]).sort_values("ts").reset_index(drop=True)
    df_extended["label_reported_at"] = pd.to_datetime(df_extended["label_reported_at"], utc=True)

    prepared2 = train_fraud.prepare(params, df=df_extended)
    test_idx2 = prepared2["test"][prepared2["test"]["transaction_id"] == "tx_hist_4_test"].index[0]
    row2 = prepared2["X_test"].loc[test_idx2]

    # All 28 features must remain strictly identical (point-in-time invariant)
    np.testing.assert_array_equal(row1.to_numpy(), row2.to_numpy())


def test_combined_build_preserves_categorical_encoder_isolation(deterministic_train_test_data):
    """7. Novel categories in val, cal, or test receive 0.0 and never leak into fitted mapping."""
    params, df = deterministic_train_test_data
    probe_df = df.copy()

    # Inject novel canary categories in val (row 82), cal (row 96), and test (row 112)
    probe_df.loc[82, "channel"] = "novel_val_channel_canary"
    probe_df.loc[96, "device_type"] = "novel_cal_device_canary"
    probe_df.loc[112, "region"] = "novel_test_region_canary"

    prepared = train_fraud.prepare(params, df=probe_df)
    mapping = prepared["mapping"]

    # Canaries never appear in the fitted mapping
    assert "novel_val_channel_canary" not in mapping["channel"]
    assert "novel_cal_device_canary" not in mapping["device_type"]
    assert "novel_test_region_canary" not in mapping["region"]

    # Novel categories map to 0.0 fallback
    val_idx = probe_df.index[82]
    cal_idx = probe_df.index[96]
    test_idx = probe_df.index[112]

    assert prepared["X_val"].loc[val_idx, "cat__channel"] == 0.0
    assert prepared["X_cal"].loc[cal_idx, "cat__device_type"] == 0.0
    assert prepared["X_test"].loc[test_idx, "cat__region"] == 0.0


def test_combined_build_preserves_exact_index_and_label_alignment(deterministic_train_test_data):
    """8. Exact index and label alignment preserved for train partitions and test split."""
    params, df = deterministic_train_test_data
    prepared = train_fraud.prepare(params, df=df)

    train = prepared["train"]
    test = prepared["test"]
    X_fit = prepared["X_fit"]
    y_fit = prepared["y_fit"]
    X_val = prepared["X_val"]
    y_val = prepared["y_val"]
    X_cal = prepared["X_cal"]
    y_cal = prepared["y_cal"]
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]

    # Index preservation
    cut_val = len(X_fit)
    cut_cal = cut_val + len(X_val)

    assert X_fit.index.equals(train.iloc[:cut_val].index)
    assert X_val.index.equals(train.iloc[cut_val:cut_cal].index)
    assert X_cal.index.equals(train.iloc[cut_cal:].index)
    assert X_test.index.equals(test.index)

    # Label alignment
    assert np.array_equal(y_fit, train.iloc[:cut_val]["is_fraud"].to_numpy())
    assert np.array_equal(y_val, train.iloc[cut_val:cut_cal]["is_fraud"].to_numpy())
    assert np.array_equal(y_cal, train.iloc[cut_cal:]["is_fraud"].to_numpy())
    assert np.array_equal(y_test, test["is_fraud"].to_numpy())


# --- 2. Pipeline Evaluate Tests ---


def test_evaluate_metrics_and_cost_curve(fraud_params):
    """evaluate calculates recall@FPR, cost curve, PR-AUC, ECE, and lift."""
    n = 200
    y_test = np.array([0] * 180 + [1] * 20)
    scores = np.linspace(0.01, 0.99, n)
    test_df = pd.DataFrame({
        "amount": np.random.uniform(10.0, 500.0, size=n),
    })

    res = train_fraud.evaluate(
        name="test_model",
        y_test=y_test,
        probs=scores,
        test=test_df,
        baseline_recall=0.30,
        params=fraud_params,
    )

    assert res["model"] == "test_model"
    assert "recall_at_1pct_fpr" in res
    assert "threshold_at_1pct_fpr" in res
    assert "pr_auc" in res
    assert "lift_over_rules" in res
    assert "ece" in res
    assert "cost_optimal" in res
    assert "cost_curve" in res

    assert 0.0 <= res["recall_at_1pct_fpr"] <= 1.0
    assert 0.0 <= res["pr_auc"] <= 1.0
    assert 0.0 <= res["ece"] <= 1.0
    assert isinstance(res["cost_curve"], list)
    assert len(res["cost_curve"]) > 0


# --- 3. Main End-to-End Pipeline Tests ---


def test_main_end_to_end_mocked(
    tmp_path: pathlib.Path,
    fraud_params,
    synthetic_transactions,
    monkeypatch,
):
    """main() executes end-to-end training, calibration, reporting, and gating."""
    # Write temporary params.yaml
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.dump({"fraud": fraud_params}))

    # Redirect outputs to tmp_path
    reports_dir = tmp_path / "reports"
    artifacts_dir = tmp_path / "artifacts" / "fraud"
    monkeypatch.setattr(train_fraud, "OUT", artifacts_dir)

    # Mock prepare to use our synthetic transactions
    real_prepare = train_fraud.prepare
    monkeypatch.setattr(
        train_fraud,
        "prepare",
        lambda p: real_prepare(p, df=synthetic_transactions),
    )

    # Mock write_report to target tmp_path
    real_write_report = train_fraud.write_report
    def mocked_write_report(path, title, sections, gates=(), out_json=None):
        return real_write_report(
            path=str(reports_dir / "m3_fraud.html"),
            title=title,
            sections=sections,
            gates=gates,
            out_json=str(reports_dir / "m3_fraud.json"),
        )
    monkeypatch.setattr(train_fraud, "write_report", mocked_write_report)

    # Use local temporary mlflow tracking uri
    import mlflow
    mlflow.set_tracking_uri(pathlib.Path(tmp_path / "mlruns").as_uri())
    exp_id = mlflow.create_experiment("test_m3_experiment")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test_m3_experiment")

    # Run main
    exit_code = train_fraud.main(["--params", str(params_path)])

    # Pipeline executed and produced reports & artifacts
    assert (reports_dir / "m3_fraud.html").exists()
    assert (reports_dir / "m3_fraud.json").exists()
    assert (artifacts_dir / "metrics.json").exists()

    metrics_payload = json.loads((artifacts_dir / "metrics.json").read_text())
    assert "champion" in metrics_payload
    assert metrics_payload["champion"] in ("xgboost", "lightgbm")
    assert "feature_names" in metrics_payload
    assert "frequency_encoding" in metrics_payload
    assert "results" in metrics_payload
    assert "baseline" in metrics_payload
    assert "disparate_impact" in metrics_payload

    # Exit code is 0 (all gates passed) or 1 (gate failed on synthetic data)
    assert exit_code in (0, 1)


def test_main_gate_failure_returns_1(
    tmp_path: pathlib.Path,
    fraud_params,
    synthetic_transactions,
    monkeypatch,
):
    """When a gate check fails, main() logs failure and returns exit code 1."""
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.dump({"fraud": fraud_params}))

    reports_dir = tmp_path / "reports"
    artifacts_dir = tmp_path / "artifacts" / "fraud"
    monkeypatch.setattr(train_fraud, "OUT", artifacts_dir)

    real_prepare = train_fraud.prepare
    monkeypatch.setattr(
        train_fraud,
        "prepare",
        lambda p: real_prepare(p, df=synthetic_transactions),
    )

    real_write_report = train_fraud.write_report
    monkeypatch.setattr(
        train_fraud,
        "write_report",
        lambda path, title, sections, gates=(), out_json=None: real_write_report(
            path=str(reports_dir / "m3_fraud.html"),
            title=title,
            sections=sections,
            gates=gates,
            out_json=str(reports_dir / "m3_fraud.json"),
        ),
    )

    # Force check_gate to fail
    monkeypatch.setattr(
        train_fraud,
        "check_gate",
        lambda metric, val: (False, f"{metric}={val:.4f} -> forced FAIL"),
    )

    import mlflow
    mlflow.set_tracking_uri(pathlib.Path(tmp_path / "mlruns").as_uri())

    exit_code = train_fraud.main(["--params", str(params_path)])
    assert exit_code == 1
