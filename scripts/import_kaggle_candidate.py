"""AegisFlow M3: Local Candidate Importer for Kaggle Training Artifacts.

Performs staged, fail-safe validation, restoration, and registration:
- Stage A: Bundle validation (archive integrity, Git commit, DVC lockfile, stage outputs)
- Stage B: Model validation (MLflow flavor loading, feature ordering, inference test)
- Stage C: Local environment validation (MLflow server and registry connectivity)
- Stage D: Restoration (DVC outputs, lockfile, and DVC verification)
- Stage E: MLflow import and candidate registration (calling register_candidate)

Usage:
  python scripts/import_kaggle_candidate.py --bundle /path/to/aegisflow_m3_kaggle_bundle.tar.gz
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Mapping

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

from aegis_ml.training.matrix import feature_names
from aegis_ml.training.registry import MODEL, register_candidate

APPROVED_COMMIT = "2533ccc7325c32255a707e123f49418e833aa6d3"

REQUIRED_BUNDLE_MEMBERS = [
    "dvc.lock",
    "execution_manifest.json",
    "data/processed/ieee_transactions.parquet",
    "data/processed/ieee_meta.json",
    "reports/m3_fraud.json",
    "reports/m3_fraud.html",
    "artifacts/fraud/metrics.json",
    "exported_mlflow/model/MLmodel",
    "exported_mlflow/model/model.pkl",
    "exported_mlflow/model/columns.json",
    "exported_mlflow/model/frequency_encoding.json",
    "exported_mlflow/calibrator/MLmodel",
    "exported_mlflow/calibrator/model.pkl",
]


class ValidationError(Exception):
    """Raised when bundle, model, or environment validation fails."""


def validate_bundle(
    bundle_path: pathlib.Path,
    extract_dir: pathlib.Path,
    approved_commit: str = APPROVED_COMMIT,
    allow_commit_override: bool = False,
) -> dict[str, Any]:
    """Stage A: Validate archive structure, execution provenance, and DVC lockfile."""
    print("=== Stage A: Bundle Validation ===")
    if not bundle_path.exists():
        raise ValidationError(f"Bundle file not found: {bundle_path}")

    if not tarfile.is_tarfile(bundle_path):
        raise ValidationError(f"Bundle is not a valid tar/gzip archive: {bundle_path}")

    with tarfile.open(bundle_path, "r:gz") as tar:
        names = set(tar.getnames())
        # Check required files
        for req in REQUIRED_BUNDLE_MEMBERS:
            if req not in names:
                raise ValidationError(f"Archive missing mandatory required file: {req}")

        # Check for path traversal safety
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in member.name:
                raise ValidationError(f"Insecure path detected in archive member: {member.name}")

        tar.extractall(path=extract_dir)

    print(f"Archive extracted to temporary staging: {extract_dir}")

    # Validate execution_manifest.json
    manifest_path = extract_dir / "execution_manifest.json"
    if not manifest_path.exists():
        raise ValidationError("execution_manifest.json missing from bundle root")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"execution_manifest.json is corrupted or not valid JSON: {exc}") from exc

    commit = manifest.get("git_commit")
    if not commit:
        raise ValidationError("execution_manifest.json missing 'git_commit' attribute")

    if commit != approved_commit and not allow_commit_override:
        raise ValidationError(
            f"Git commit provenance mismatch! Bundle commit '{commit}' != approved baseline '{approved_commit}'"
        )

    kaggle_run_id = manifest.get("kaggle_run_id")
    if not kaggle_run_id:
        raise ValidationError("execution_manifest.json missing 'kaggle_run_id' attribute")

    # Validate dvc.lock content
    lock_path = extract_dir / "dvc.lock"
    lock_text = lock_path.read_text(encoding="utf-8")
    if "prepare_ieee" not in lock_text or "train_fraud" not in lock_text:
        raise ValidationError("dvc.lock in bundle is missing required M3 stages ('prepare_ieee', 'train_fraud')")

    # Validate data and reports non-empty
    parquet_path = extract_dir / "data/processed/ieee_transactions.parquet"
    if parquet_path.stat().st_size < 1000:
        raise ValidationError(f"ieee_transactions.parquet is suspiciously small ({parquet_path.stat().st_size} bytes)")

    metrics_path = extract_dir / "artifacts/fraud/metrics.json"
    try:
        metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "champion" not in metrics_data or "results" not in metrics_data:
            raise ValidationError("artifacts/fraud/metrics.json missing 'champion' or 'results' fields")
    except Exception as exc:
        raise ValidationError(f"Error reading artifacts/fraud/metrics.json: {exc}") from exc

    print(f"[PASS] Stage A: Bundle structure, manifest ({commit[:8]}), and DVC lockfile validated.")
    return manifest


def validate_model(extract_dir: pathlib.Path) -> tuple[Any, Any, list[str], dict[str, Any]]:
    """Stage B: Validate model, calibrator, feature ordering, and inference compatibility."""
    print("=== Stage B: Model Validation ===")
    model_dir = extract_dir / "exported_mlflow/model"
    calibrator_dir = extract_dir / "exported_mlflow/calibrator"

    # 1. Load estimator and calibrator using actual MLflow flavor
    try:
        model = mlflow.sklearn.load_model(str(model_dir))
    except Exception as exc:
        raise ValidationError(f"Failed to load exported MLflow model from {model_dir}: {exc}") from exc

    try:
        calibrator = mlflow.sklearn.load_model(str(calibrator_dir))
    except Exception as exc:
        raise ValidationError(f"Failed to load exported MLflow calibrator from {calibrator_dir}: {exc}") from exc

    # 2. Validate feature names and exact ordering
    cols_path = model_dir / "columns.json"
    if not cols_path.exists():
        raise ValidationError("model/columns.json missing from exported model directory")

    cols_data = json.loads(cols_path.read_text(encoding="utf-8"))
    bundle_feature_names = cols_data.get("feature_names", [])
    expected_names = feature_names()

    if bundle_feature_names != expected_names:
        raise ValidationError(
            f"Feature ordering mismatch!\nBundle ({len(bundle_feature_names)}): {bundle_feature_names}\n"
            f"Expected ({len(expected_names)}): {expected_names}"
        )

    # 3. Validate frequency encoding metadata
    freq_path = model_dir / "frequency_encoding.json"
    if not freq_path.exists():
        raise ValidationError("model/frequency_encoding.json missing from exported model directory")

    freq_data = json.loads(freq_path.read_text(encoding="utf-8"))
    if not isinstance(freq_data, dict) or len(freq_data) == 0:
        raise ValidationError("Frequency encoding metadata is empty or not a valid dictionary")

    # 4. Inference test with dummy row
    dummy_row = pd.DataFrame([np.zeros(len(expected_names), dtype=np.float64)], columns=expected_names)
    try:
        raw_probs = model.predict_proba(dummy_row)[:, 1]
    except Exception as exc:
        raise ValidationError(f"Model inference failed on dummy input: {exc}") from exc

    if len(raw_probs) != 1 or not np.isfinite(raw_probs[0]) or not (0.0 <= raw_probs[0] <= 1.0):
        raise ValidationError(f"Model produced invalid raw probability: {raw_probs}")

    try:
        cal_probs = calibrator.predict(raw_probs)
    except Exception as exc:
        raise ValidationError(f"Calibrator prediction failed: {exc}") from exc

    if len(cal_probs) != 1 or not np.isfinite(cal_probs[0]) or not (0.0 <= cal_probs[0] <= 1.0):
        raise ValidationError(f"Calibrator produced invalid calibrated probability: {cal_probs}")

    print(f"[PASS] Stage B: Model ({type(model).__name__}) and Calibrator ({type(calibrator).__name__}) validated.")
    return model, calibrator, expected_names, freq_data


def validate_environment(client: MlflowClient, model_name: str = MODEL) -> dict[str, Any]:
    """Stage C: Verify local MLflow tracking server and inspect model registry state."""
    print("=== Stage C: Local Environment Validation ===")
    try:
        experiments = client.search_experiments()
        print(f"Local MLflow tracking server reachable ({len(experiments)} experiments).")
    except Exception as exc:
        raise ValidationError(f"Cannot connect to local MLflow tracking server: {exc}") from exc

    # Inspect model registry state
    champ_version = None
    chal_version = None
    try:
        reg_model = client.get_registered_model(model_name)
        aliases = reg_model.aliases or {}
        champ_version = aliases.get("champion")
        chal_version = aliases.get("challenger")
        print(f"Registered model '{model_name}' exists: @champion=v{champ_version}, @challenger=v{chal_version}")
    except Exception:
        print(f"Registered model '{model_name}' does not exist yet (initial bootstrap state).")

    print("[PASS] Stage C: Local MLflow environment and registry validated.")
    return {
        "champion_version": champ_version,
        "challenger_version": chal_version,
        "is_bootstrap": champ_version is None,
    }


def restore_artifacts(
    extract_dir: pathlib.Path,
    target_repo: pathlib.Path,
    dry_run: bool = False,
) -> None:
    """Stage D: Restore stage outputs and lockfile into target repo and run DVC verification."""
    print("=== Stage D: Artifact Restoration & DVC Verification ===")
    files_to_restore = [
        "data/processed/ieee_transactions.parquet",
        "data/processed/ieee_meta.json",
        "reports/m3_fraud.json",
        "reports/m3_fraud.html",
        "dvc.lock",
    ]

    if dry_run:
        print("[DRY-RUN] Simulating file restoration into repository...")
        for rel in files_to_restore:
            src = extract_dir / rel
            dest = target_repo / rel
            print(f"  [DRY-RUN] Would copy {src} -> {dest}")
        print("  [DRY-RUN] Would restore artifacts/fraud/ directory")
        return

    # Restore individual files
    for rel in files_to_restore:
        src = extract_dir / rel
        dest = target_repo / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"Restored: {dest}")

    # Restore artifacts/fraud/ directory
    src_fraud = extract_dir / "artifacts/fraud"
    dest_fraud = target_repo / "artifacts/fraud"
    dest_fraud.mkdir(parents=True, exist_ok=True)
    for item in src_fraud.iterdir():
        dest_item = dest_fraud / item.name
        if item.is_dir():
            shutil.copytree(item, dest_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest_item)
        print(f"Restored: {dest_item}")

    # Verify restored outputs against dvc.lock
    dvc_status = subprocess.run(
        ["dvc", "status"],
        cwd=target_repo,
        capture_output=True,
        text=True,
    )
    print("DVC status after restoration:")
    print(dvc_status.stdout)
    if "data/processed/ieee_transactions.parquet" in dvc_status.stdout and "deleted" in dvc_status.stdout:
        raise ValidationError("DVC reports data/processed outputs are missing or inconsistent with lockfile!")

    print("[PASS] Stage D: Artifacts restored and verified against DVC lockfile.")


def register_candidate_run(
    model: Any,
    calibrator: Any,
    feature_names_list: list[str],
    freq_data: dict[str, Any],
    manifest: dict[str, Any],
    extract_dir: pathlib.Path,
    client: MlflowClient,
    model_name: str = MODEL,
    dry_run: bool = False,
) -> Any:
    """Stage E: Log new local MLflow run with Kaggle provenance and invoke register_candidate."""
    print("=== Stage E: MLflow Import & Candidate Registration ===")
    if dry_run:
        print("[DRY-RUN] Skipping actual MLflow run creation and model registration.")
        return None

    metrics_path = extract_dir / "artifacts/fraud/metrics.json"
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    champ_name = metrics_payload.get("champion", manifest.get("champion_algorithm", "xgboost"))
    champ_metrics = metrics_payload["results"][champ_name]

    # Start new local MLflow run
    with mlflow.start_run(run_name="kaggle-import-fraud") as run:
        local_run_id = run.info.run_id
        print(f"Created local MLflow run: {local_run_id}")

        # Set provenance tags
        tags = {
            "original_kaggle_run_id": manifest.get("kaggle_run_id", "unknown"),
            "git_commit": manifest.get("git_commit", APPROVED_COMMIT),
            "git_branch": manifest.get("git_branch", "feature/m3-fraud-models"),
            "execution_platform": manifest.get("platform", "kaggle_cpu"),
            "execution_timestamp_utc": manifest.get("timestamp_utc", ""),
            "ingestion_source": "aegisflow_m3_kaggle_bundle",
            "champion_algorithm": champ_name,
        }
        for k, v in tags.items():
            mlflow.set_tag(k, v)

        # Log metrics
        for k, v in champ_metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                mlflow.log_metric(k, float(v))

        # Log report and metadata artifacts
        report_html = extract_dir / "reports/m3_fraud.html"
        if report_html.exists():
            mlflow.log_artifact(str(report_html))

        meta_json = extract_dir / "exported_mlflow/model_meta.json"
        if meta_json.exists():
            mlflow.log_artifact(str(meta_json))

        # Invoke the authoritative register_candidate implementation
        print(f"Calling register_candidate() for model '{model_name}'...")
        decision = register_candidate(
            run_id=local_run_id,
            model=model,
            calibrator=calibrator,
            feature_names=feature_names_list,
            metrics=champ_metrics,
            client=client,
            model_name=model_name,
            frequency_encoding=freq_data,
        )

        print(f"Registration Decision: {decision.reason} (version {decision.version})")
        print("[PASS] Stage E: Model candidate successfully imported and registered.")
        return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import Kaggle M3 training bundle into AegisFlow.")
    parser.add_argument("--bundle", required=True, help="Path to aegisflow_m3_kaggle_bundle.tar.gz")
    parser.add_argument("--target-dir", default=".", help="Root of local AegisFlow repository (default: '.')")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5008"),
        help="Local MLflow tracking URI (default: http://127.0.0.1:5008)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate bundle, model, and environment without mutating repo or registry",
    )
    parser.add_argument(
        "--allow-commit-override",
        action="store_true",
        help="Allow import from a commit other than the approved baseline (for testing only)",
    )
    args = parser.parse_args(argv)

    bundle_path = pathlib.Path(args.bundle).resolve()
    target_repo = pathlib.Path(args.target_dir).resolve()
    tracking_uri = args.tracking_uri

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    with tempfile.TemporaryDirectory() as tmp_str:
        extract_dir = pathlib.Path(tmp_str)

        try:
            # Stage A: Bundle validation
            manifest = validate_bundle(
                bundle_path,
                extract_dir,
                allow_commit_override=args.allow_commit_override,
            )

            # Stage B: Model validation
            model, calibrator, feature_names_list, freq_data = validate_model(extract_dir)

            # Stage C: Local environment validation
            validate_environment(client, model_name=MODEL)

            # Stage D: Restoration & DVC check
            restore_artifacts(extract_dir, target_repo, dry_run=args.dry_run)

            # Stage E: MLflow import and candidate registration
            register_candidate_run(
                model=model,
                calibrator=calibrator,
                feature_names_list=feature_names_list,
                freq_data=freq_data,
                manifest=manifest,
                extract_dir=extract_dir,
                client=client,
                model_name=MODEL,
                dry_run=args.dry_run,
            )

        except ValidationError as val_err:
            print(f"\n[FAIL] Validation Error: {val_err}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"\n[ERROR] Unexpected error during candidate import: {exc}", file=sys.stderr)
            return 2

    print("\nCandidate import workflow finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
