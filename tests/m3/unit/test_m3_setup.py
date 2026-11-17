"""Acceptance tests for AegisFlow Milestone 3 - Step 3.1 Setup & Repository Readiness.

Verifies:
1. Package directories and __init__.py scaffolding.
2. params.yaml fraud configuration and parameter validation.
3. ml/requirements-train.txt pinned dependencies and installed environment consistency.
4. Makefile M3 targets and .PHONY declarations.
5. ADR-0003 dependency deviation documentation.
"""
from __future__ import annotations

import importlib.metadata
import pathlib
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_package_scaffolding() -> None:
    """Verify all M3 directories and package markers exist."""
    required_dirs = [
        ROOT / "ml/src/aegis_ml/training",
        ROOT / "ml/src/aegis_ml/models",
        ROOT / "ml/src/aegis_ml/evaluation",
        ROOT / "docs/adr",
        ROOT / "tests/m3/unit",
        ROOT / "tests/m3/integration",
        ROOT / "reports",
    ]
    for d in required_dirs:
        assert d.is_dir(), f"Required directory missing: {d}"

    package_inits = [
        ROOT / "ml/src/aegis_ml/training/__init__.py",
        ROOT / "ml/src/aegis_ml/models/__init__.py",
        ROOT / "ml/src/aegis_ml/evaluation/__init__.py",
        ROOT / "tests/m3/__init__.py",
        ROOT / "tests/m3/unit/__init__.py",
        ROOT / "tests/m3/integration/__init__.py",
    ]
    for init_file in package_inits:
        assert init_file.is_file(), f"Package init file missing: {init_file}"


def test_params_yaml_fraud_configuration() -> None:
    """Verify params.yaml contains the exact required fraud configuration block."""
    params_path = ROOT / "params.yaml"
    assert params_path.is_file(), "params.yaml does not exist"

    with open(params_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "fraud" in config, "'fraud' root key missing from params.yaml"
    fraud = config["fraud"]

    # Global fraud parameters
    assert fraud["seed"] == 42
    assert fraud["train_weeks"] == 6
    assert fraud["test_weeks"] == 2
    assert fraud["embargo_hours"] == 24
    assert fraud["label_horizon_days"] == 30, "label_horizon_days must be 30"
    assert fraud["calibration_fraction"] == 0.15
    assert fraud["max_fpr"] == 0.01
    assert fraud["review_cost"] == 3.0
    assert fraud["friction_cost"] == 12.0

    # XGBoost hyperparameter block
    assert "xgboost" in fraud
    xgb = fraud["xgboost"]
    assert xgb["n_estimators"] == 800
    assert xgb["max_depth"] == 6
    assert xgb["learning_rate"] == 0.05
    assert xgb["subsample"] == 0.8
    assert xgb["colsample_bytree"] == 0.8
    assert xgb["min_child_weight"] == 5
    assert xgb["reg_lambda"] == 1.0
    assert xgb["early_stopping_rounds"] == 60
    assert xgb["seed"] == 42
    assert xgb["n_jobs"] == 4

    # LightGBM hyperparameter block
    assert "lightgbm" in fraud
    lgb = fraud["lightgbm"]
    assert lgb["n_estimators"] == 800
    assert lgb["learning_rate"] == 0.05
    assert lgb["num_leaves"] == 63
    assert lgb["min_child_samples"] == 50
    assert lgb["subsample"] == 0.8
    assert lgb["colsample_bytree"] == 0.8
    assert lgb["early_stopping_rounds"] == 60
    assert lgb["seed"] == 42
    assert lgb["n_jobs"] == 4


def test_training_requirements_pinned() -> None:
    """Verify ml/requirements-train.txt pins the authorized dependencies."""
    req_path = ROOT / "ml/requirements-train.txt"
    assert req_path.is_file(), "ml/requirements-train.txt missing"

    content = req_path.read_text(encoding="utf-8")

    expected_pins = {
        "xgboost": "2.1.3",
        "lightgbm": "4.5.0",
        "scikit-learn": "1.5.2",
        "shap": "0.46.0",
        "pandas": "2.2.3",
        "pyarrow": "17.0.0",  # Authorized deviation per ADR-0003
        "numpy": "1.26.4",
        "networkx": "3.4.2",
        "mlflow": "2.17.2",
        "dvc": "3.56.0",
        "pyyaml": "6.0.2",
        "sqlalchemy": "2.0.36",
        "psycopg[binary]": "3.2.3",
        "pytest": "8.3.4",
    }

    for pkg_spec, ver in expected_pins.items():
        expected_line = f"{pkg_spec}=={ver}"
        assert expected_line in content, f"Missing pinned requirement: {expected_line}"

    # Verify active interpreter installation
    installed_checks = {
        "xgboost": "2.1.3",
        "lightgbm": "4.5.0",
        "scikit-learn": "1.5.2",
        "shap": "0.46.0",
        "pandas": "2.2.3",
        "pyarrow": "17.0.0",
        "numpy": "1.26.4",
        "networkx": "3.4.2",
        "mlflow": "2.17.2",
        "dvc": "3.56.0",
        "pyyaml": "6.0.2",
        "sqlalchemy": "2.0.36",
        "psycopg": "3.2.3",
        "pytest": "8.3.4",
    }
    for pkg, ver in installed_checks.items():
        installed_ver = importlib.metadata.version(pkg)
        assert installed_ver == ver, f"Package {pkg} version mismatch: expected {ver}, got {installed_ver}"


def test_makefile_m3_targets_present() -> None:
    """Verify Makefile includes required M3 targets and PHONY declarations."""
    makefile_path = ROOT / "Makefile"
    assert makefile_path.is_file(), "Makefile missing"

    content = makefile_path.read_text(encoding="utf-8")

    m3_targets = [
        "prepare-ieee:",
        "train-fraud:",
        "compare:",
        "promote:",
        "rollback:",
        "typologies:",
        "test-m3:",
        "test-m3-int:",
        "verify-m3:",
        "m3:",
    ]
    for target in m3_targets:
        assert target in content, f"Makefile missing target: {target}"

    # Verify .PHONY declaration contains M3 targets
    assert ".PHONY: prepare-ieee train-fraud compare promote rollback typologies" in content
    assert "test-m3 test-m3-int verify-m3 m3" in content


def test_adr_reconciliation_document_exists() -> None:
    """Verify ADR-0003 documenting the PyArrow deviation exists and covers key criteria."""
    adr_path = ROOT / "docs/adr/0003-m3-dependency-reconciliation-pyarrow.md"
    assert adr_path.is_file(), "ADR-0003 document missing"

    content = adr_path.read_text(encoding="utf-8")
    assert "mlflow==2.17.2" in content
    assert "pyarrow==18.1.0" in content
    assert "17.0.0" in content
    assert "pyarrow<18" in content
