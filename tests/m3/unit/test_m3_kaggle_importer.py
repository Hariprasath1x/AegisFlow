"""Unit tests for AegisFlow M3 Kaggle training notebook and candidate importer."""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import tarfile
import tempfile
import pytest

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from aegis_ml.training.matrix import feature_names
from scripts.import_kaggle_candidate import (
    APPROVED_COMMIT,
    ValidationError,
    main as importer_main,
    validate_bundle,
    validate_environment,
    validate_model,
    register_candidate_run,
    restore_artifacts,
)


@pytest.fixture
def mock_kaggle_bundle(tmp_path):
    """Synthesizes a minimal valid mock Kaggle export bundle for testing importer stages."""
    bundle_dir = tmp_path / "mock_staging"
    bundle_dir.mkdir()

    # 1. dvc.lock
    lock_content = (
        "schema: '2.0'\n"
        "stages:\n"
        "  prepare_ieee:\n"
        "    cmd: PYTHONPATH=ml/src python -m aegis_ml.training.prepare_ieee --raw data/raw/ieee --out data/processed\n"
        "    outs:\n"
        "    - path: data/processed/ieee_transactions.parquet\n"
        "  train_fraud:\n"
        "    cmd: PYTHONPATH=ml/src python -m aegis_ml.training.train_fraud\n"
        "    outs:\n"
        "    - path: artifacts/fraud\n"
    )
    (bundle_dir / "dvc.lock").write_text(lock_content, encoding="utf-8")

    # 2. execution_manifest.json
    manifest = {
        "git_commit": APPROVED_COMMIT,
        "git_branch": "feature/m3-fraud-models",
        "kaggle_run_id": "test-kaggle-run-12345",
        "dvc_version": "3.56.0",
        "timestamp_utc": "2026-11-19T12:00:00Z",
        "platform": "kaggle_cpu",
        "champion_algorithm": "xgboost",
    }
    (bundle_dir / "execution_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # 3. data/processed/
    proc_dir = bundle_dir / "data/processed"
    proc_dir.mkdir(parents=True)
    # create small dummy parquet
    dummy_df = pd.DataFrame({"transaction_id": ["1", "2"], "amount": [10.0, 20.0], "is_fraud": [0, 1]})
    dummy_df.to_parquet(proc_dir / "ieee_transactions.parquet")
    (proc_dir / "ieee_meta.json").write_text(json.dumps({"rows": 2, "fraud_rate": 0.5}), encoding="utf-8")

    # 4. reports/
    rep_dir = bundle_dir / "reports"
    rep_dir.mkdir(parents=True)
    (rep_dir / "m3_fraud.json").write_text(json.dumps({"gates": [{"detail": "test", "pass": True}]}), encoding="utf-8")
    (rep_dir / "m3_fraud.html").write_text("<html>Report</html>", encoding="utf-8")

    # 5. artifacts/fraud/
    art_dir = bundle_dir / "artifacts/fraud"
    art_dir.mkdir(parents=True)
    metrics_payload = {
        "champion": "xgboost",
        "feature_names": feature_names(),
        "categorical": ["channel"],
        "frequency_encoding": {"channel": {"W": 1.0}},
        "results": {
            "xgboost": {
                "recall_at_1pct_fpr": 0.55,
                "lift_over_rules": 0.35,
                "pr_auc": 0.65,
                "ece": 0.03,
                "threshold_at_1pct_fpr": 0.45,
            }
        },
    }
    (art_dir / "metrics.json").write_text(json.dumps(metrics_payload), encoding="utf-8")

    # 6. exported_mlflow/
    exp_mlflow = bundle_dir / "exported_mlflow"
    exp_mlflow.mkdir(parents=True)

    # Train dummy logistic regression on dummy feature vector matching feature_names
    f_names = feature_names()
    X_dummy = np.random.RandomState(42).randn(10, len(f_names))
    y_dummy = np.array([0, 1] * 5)
    model = LogisticRegression().fit(X_dummy, y_dummy)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit([0.1, 0.9], [0, 1])

    mlflow.sklearn.save_model(model, path=str(exp_mlflow / "model"))
    mlflow.sklearn.save_model(calibrator, path=str(exp_mlflow / "calibrator"))

    (exp_mlflow / "model/columns.json").write_text(json.dumps({"feature_names": f_names}), encoding="utf-8")
    (exp_mlflow / "model/frequency_encoding.json").write_text(json.dumps({"channel": {"W": 1.0}}), encoding="utf-8")
    (exp_mlflow / "model_meta.json").write_text(json.dumps(metrics_payload), encoding="utf-8")

    # Package into tar.gz
    tar_path = tmp_path / "test_bundle.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for item in bundle_dir.iterdir():
            tar.add(item, arcname=item.name)

    return tar_path, manifest, metrics_payload


def test_kaggle_notebook_structure_and_stages():
    """Verify kaggle_train_m3.ipynb structure, valid JSON, and required stages."""
    nb_path = pathlib.Path("kaggle_train_m3.ipynb")
    assert nb_path.exists(), "kaggle_train_m3.ipynb missing from repository root"

    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
    assert "cells" in nb_data and len(nb_data["cells"]) >= 7

    full_code = " ".join("".join(c.get("source", [])) for c in nb_data["cells"] if c.get("cell_type") == "code")

    # 1. Approved commit
    assert APPROVED_COMMIT in full_code
    # 2. Dependency install
    assert "requirements-train.txt" in full_code
    assert "dvc" in full_code
    # 3. Dynamic dataset discovery
    assert "train_transaction.csv" in full_code
    assert "train_identity.csv" in full_code
    assert "data/raw/ieee" in full_code
    # 4. MLflow tracking & SQLite
    assert "MLFLOW_TRACKING_URI" in full_code
    assert "sqlite:///" in full_code
    # 5. Pipeline execution
    assert "dvc" in full_code and "repro" in full_code
    # 6. Model verification
    assert "load_model" in full_code
    # 7. Packaging
    assert "aegisflow_m3_kaggle_bundle.tar.gz" in full_code
    assert "execution_manifest.json" in full_code


def test_importer_cli_help():
    """Verify importer CLI argument handling and help message."""
    with pytest.raises(SystemExit) as exc_info:
        importer_main(["--help"])
    assert exc_info.value.code == 0


def test_bundle_validation_success(mock_kaggle_bundle, tmp_path):
    """Verify Stage A passes on valid bundle."""
    bundle_path, _, _ = mock_kaggle_bundle
    extract_dir = tmp_path / "extract_test"
    extract_dir.mkdir()

    manifest = validate_bundle(bundle_path, extract_dir)
    assert manifest["git_commit"] == APPROVED_COMMIT
    assert manifest["kaggle_run_id"] == "test-kaggle-run-12345"


def test_bundle_validation_rejects_invalid_commit(mock_kaggle_bundle, tmp_path):
    """Verify Stage A rejects bundle built from unapproved Git commit."""
    bundle_path, _, _ = mock_kaggle_bundle
    extract_dir = tmp_path / "extract_test"
    extract_dir.mkdir()

    with pytest.raises(ValidationError, match="Git commit provenance mismatch"):
        validate_bundle(bundle_path, extract_dir, approved_commit="0000000000000000000000000000000000000000")


def test_bundle_validation_rejects_missing_lockfile(tmp_path):
    """Verify Stage A rejects archive missing required files like dvc.lock."""
    bad_dir = tmp_path / "bad_bundle"
    bad_dir.mkdir()
    (bad_dir / "execution_manifest.json").write_text("{}", encoding="utf-8")

    bad_tar = tmp_path / "bad.tar.gz"
    with tarfile.open(bad_tar, "w:gz") as tar:
        tar.add(bad_dir / "execution_manifest.json", arcname="execution_manifest.json")

    extract_dir = tmp_path / "extract_bad"
    extract_dir.mkdir()

    with pytest.raises(ValidationError, match="missing mandatory required file: dvc.lock"):
        validate_bundle(bad_tar, extract_dir)


def test_model_validation_and_inference(mock_kaggle_bundle, tmp_path):
    """Verify Stage B model loading, feature ordering, and inference test."""
    bundle_path, _, _ = mock_kaggle_bundle
    extract_dir = tmp_path / "extract_model_test"
    extract_dir.mkdir()
    validate_bundle(bundle_path, extract_dir)

    model, cal, f_list, freq = validate_model(extract_dir)
    assert f_list == feature_names()
    assert "channel" in freq
    assert hasattr(model, "predict_proba")
    assert hasattr(cal, "predict")


def test_model_validation_rejects_shuffled_features(mock_kaggle_bundle, tmp_path):
    """Verify Stage B rejects model with scrambled feature ordering."""
    bundle_path, _, _ = mock_kaggle_bundle
    extract_dir = tmp_path / "extract_shuffle_test"
    extract_dir.mkdir()
    validate_bundle(bundle_path, extract_dir)

    # Scramble feature ordering
    cols_path = extract_dir / "exported_mlflow/model/columns.json"
    cols_data = json.loads(cols_path.read_text())
    cols_data["feature_names"] = list(reversed(cols_data["feature_names"]))
    cols_path.write_text(json.dumps(cols_data))

    with pytest.raises(ValidationError, match="Feature ordering mismatch"):
        validate_model(extract_dir)


def test_dry_run_full_importer(mock_kaggle_bundle, tmp_path):
    """Verify dry-run mode validates Stages A-D without mutating registry."""
    bundle_path, _, _ = mock_kaggle_bundle
    db_path = tmp_path / "mlflow_test.db"
    test_tracking_uri = f"sqlite:///{db_path}"

    rc = importer_main([
        "--bundle", str(bundle_path),
        "--target-dir", str(tmp_path),
        "--tracking-uri", test_tracking_uri,
        "--dry-run",
    ])
    assert rc == 0


def test_registration_bootstrap_contract(mock_kaggle_bundle, tmp_path):
    """Verify Stage E follows documented bootstrap contract: v1 becomes both champion & challenger."""
    bundle_path, manifest, _ = mock_kaggle_bundle
    db_path = tmp_path / "mlflow_reg_test.db"
    test_tracking_uri = f"sqlite:///{db_path}"

    mlflow.set_tracking_uri(test_tracking_uri)
    client = MlflowClient(tracking_uri=test_tracking_uri)

    extract_dir = tmp_path / "extract_reg_test"
    extract_dir.mkdir()
    validate_bundle(bundle_path, extract_dir)
    model, cal, f_names, freq = validate_model(extract_dir)

    # First registration on empty registry
    env_info = validate_environment(client, model_name="aegisflow-fraud")
    assert env_info["is_bootstrap"] is True

    decision = register_candidate_run(
        model=model,
        calibrator=cal,
        feature_names_list=f_names,
        freq_data=freq,
        manifest=manifest,
        extract_dir=extract_dir,
        client=client,
        model_name="aegisflow-fraud",
    )
    assert decision.changed is True
    assert "champion and challenger" in decision.reason
    assert decision.version == "1"

    # Verify both aliases set on version 1
    champ_v = client.get_model_version_by_alias("aegisflow-fraud", "champion")
    chal_v = client.get_model_version_by_alias("aegisflow-fraud", "challenger")
    assert str(champ_v.version) == "1"
    assert str(chal_v.version) == "1"

    # Second registration on non-empty registry (candidate becomes challenger ONLY)
    decision2 = register_candidate_run(
        model=model,
        calibrator=cal,
        feature_names_list=f_names,
        freq_data=freq,
        manifest=manifest,
        extract_dir=extract_dir,
        client=client,
        model_name="aegisflow-fraud",
    )
    assert decision2.changed is True
    assert "registered as challenger (champion stays v1)" in decision2.reason
    assert str(decision2.version) == "2"

    # Verify champion remained v1 while challenger became v2
    champ_v_after = client.get_model_version_by_alias("aegisflow-fraud", "champion")
    chal_v_after = client.get_model_version_by_alias("aegisflow-fraud", "challenger")
    assert str(champ_v_after.version) == "1"
    assert str(chal_v_after.version) == "2"
