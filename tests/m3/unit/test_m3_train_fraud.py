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
    # Ensure at least some frauds in each period
    is_fraud[5] = 1
    is_fraud[100] = 1
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
    """prepare correctly creates train, calibration, and test splits with embargo."""
    prepared = train_fraud.prepare(fraud_params, df=synthetic_transactions)

    train = prepared["train"]
    test = prepared["test"]
    X_fit = prepared["X_fit"]
    y_fit = prepared["y_fit"]
    X_cal = prepared["X_cal"]
    y_cal = prepared["y_cal"]
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]

    # No time leakage: max train timestamp < min test timestamp
    assert train["ts"].max() < test["ts"].min()

    # Embargo gap: at least embargo_hours between train and test
    embargo_delta = test["ts"].min() - train["ts"].max()
    assert embargo_delta >= pd.Timedelta(hours=fraud_params["embargo_hours"])

    # Calibration slice is the LAST fraction of training time
    total_train_rows = len(train)
    cut = int(total_train_rows * (1 - fraud_params["calibration_fraction"]))
    expected_fit_count = cut
    expected_cal_count = total_train_rows - cut

    assert len(X_fit) == expected_fit_count
    assert len(y_fit) == expected_fit_count
    assert len(X_cal) == expected_cal_count
    assert len(y_cal) == expected_cal_count
    assert len(X_test) == len(test)
    assert len(y_test) == len(test)

    # Feature shapes must match declared feature names
    from aegis_ml.training.matrix import feature_names
    cols = feature_names()
    assert list(X_fit.columns) == cols
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
