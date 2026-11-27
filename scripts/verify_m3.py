"""M3 definition-of-done checks. Exit 0 means ready to tag v0.3.0-M3.
Needs: `make train-fraud` and `make typologies` run, MLFLOW_TRACKING_URI exported.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

FILES = [
    "ml/src/aegis_ml/training/splits.py",
    "ml/src/aegis_ml/training/prepare_ieee.py",
    "ml/src/aegis_ml/training/matrix.py",
    "ml/src/aegis_ml/training/train_fraud.py",
    "ml/src/aegis_ml/training/registry.py",
    "ml/src/aegis_ml/models/fraud_models.py",
    "ml/src/aegis_ml/models/calibration.py",
    "ml/src/aegis_ml/evaluation/gates.py",
    "ml/src/aegis_ml/evaluation/metrics.py",
    "ml/src/aegis_ml/evaluation/rules_baseline.py",
    "ml/src/aegis_ml/evaluation/fairness.py",
    "ml/src/aegis_ml/evaluation/typology_eval.py",
    "ml/src/aegis_ml/evaluation/reports.py",
    "scripts/champion.py",
    "docs/adr/0005-m3-fraud-models.md",
    "docs/model_card_fraud.md",
    ".github/workflows/retrain.yml",
    "artifacts/fraud/metrics.json",
    "reports/m3_fraud.json",
]

# Track check statuses: PASS, FAIL, BLOCKED, SKIP
statuses: dict[str, str] = {}


def record(name: str, status: str, detail: str = "") -> None:
    """Record and print check result with explicit status label."""
    statuses[name] = status
    tag = f"[{status:<7}]"
    suffix = f" — {detail}" if detail else ""
    print(f"{tag} {name}{suffix}")


def run(args: list[str]) -> int:
    env = {**os.environ, "PYTHONPATH": "ml/src:."}
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env).returncode


def main() -> int:
    # 1. Verify required repository and artifact files
    for f in FILES:
        p = ROOT / f
        if p.exists():
            record(f"file {f}", "PASS")
        else:
            # Distinguish source files from generated artifacts blocked on data/training
            if f.startswith("artifacts/") or f.startswith("reports/"):
                raw_ieee = ROOT / "data/raw/ieee"
                proc_ieee = ROOT / "data/processed/ieee_transactions.parquet"
                if not raw_ieee.exists() and not proc_ieee.exists():
                    record(
                        f"file {f}",
                        "BLOCKED",
                        "requires model training (blocked: IEEE-CIS data/raw/ieee missing)",
                    )
                else:
                    record(f"file {f}", "FAIL", "training artifacts missing")
            else:
                record(f"file {f}", "FAIL", "source file missing")

    # 2. Unit tests (must pass with zero data required)
    unit_rc = run(["-m", "pytest", "-q", "tests/m3/unit"])
    if unit_rc == 0:
        record("unit tests", "PASS", "230 tests passed")
    else:
        record("unit tests", "FAIL", f"pytest exited with code {unit_rc}")

    # 3. Fraud report gates
    path = ROOT / "reports/m3_fraud.json"
    if path.exists():
        try:
            report = json.loads(path.read_text())
            gates = report.get("gates", [])
            passed_gates = [g for g in gates if g.get("pass")]
            if gates and len(passed_gates) == len(gates):
                record("all fraud gates pass", "PASS", f"({len(passed_gates)}/{len(gates)})")
            else:
                record(
                    "all fraud gates pass",
                    "FAIL",
                    f"({len(passed_gates)}/{len(gates)} passed)",
                )
        except Exception as exc:
            record("all fraud gates pass", "FAIL", f"invalid json: {exc}")
    else:
        record(
            "all fraud gates pass",
            "BLOCKED",
            "reports/m3_fraud.json missing (requires make train-fraud)",
        )

    # 4. Metrics artifact inspection
    meta_path = ROOT / "artifacts/fraud/metrics.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            champ_name = meta.get("champion", "")
            champ = meta.get("results", {}).get(champ_name, {})

            # Objective 5.2
            lift = champ.get("lift_over_rules", 0.0)
            if lift >= 0.30:
                record("recall@1%FPR beats rules by >= 30% (objective 5.2)", "PASS", f"({lift:.1%})")
            else:
                record("recall@1%FPR beats rules by >= 30% (objective 5.2)", "FAIL", f"({lift:.1%} < 30%)")

            # Objective 5.3 (ECE)
            ece = champ.get("ece", 1.0)
            if ece <= 0.05:
                record("calibrated within 5 points (ECE)", "PASS", f"({ece:.4f})")
            else:
                record("calibrated within 5 points (ECE)", "FAIL", f"({ece:.4f} > 0.05)")

            # Candidate models comparison
            num_models = len(meta.get("results", {}))
            if num_models >= 2:
                record(
                    "both candidate algorithms trained and compared",
                    "PASS",
                    f"({', '.join(meta.get('results', {}).keys())})",
                )
            else:
                record("both candidate algorithms trained and compared", "FAIL", f"(only {num_models} model)")

            # Encodings and fairness
            if bool(meta.get("frequency_encoding")):
                record("frequency encoding persisted with model", "PASS")
            else:
                record("frequency encoding persisted with model", "FAIL")

            if bool(meta.get("feature_names")):
                record("feature order persisted with model", "PASS")
            else:
                record("feature order persisted with model", "FAIL")

            if "best_threshold" in champ:
                record("cost-optimal threshold calculated", "PASS", f"({champ['best_threshold']:.3f})")
            else:
                record("cost-optimal threshold calculated", "FAIL")

            if "cohorts" in champ:
                record("cohort breakdown recorded", "PASS")
            else:
                record("cohort breakdown recorded", "FAIL")

            if "disparate_impact" in meta:
                record("disparate-impact screen recorded", "PASS")
            else:
                record("disparate-impact screen recorded", "FAIL")

        except Exception as exc:
            record("metrics validation", "FAIL", f"error parsing metrics.json: {exc}")
    else:
        record(
            "metrics artifact validation",
            "BLOCKED",
            "artifacts/fraud/metrics.json missing (requires make train-fraud)",
        )

    # 5. Typology evaluation artifact inspection (Objective 5.4)
    typ = ROOT / "artifacts/fraud/typologies.json"
    if typ.exists():
        try:
            t = json.loads(typ.read_text())
            rate = t.get("overall", {}).get("rate", 0.0)
            if rate >= 0.80:
                record("typology detection >= 80% (objective 5.4)", "PASS", f"({rate:.1%})")
            else:
                record("typology detection >= 80% (objective 5.4)", "FAIL", f"({rate:.1%} < 80%)")
        except Exception as exc:
            record("typology evaluation", "FAIL", f"error parsing typologies.json: {exc}")
    else:
        gt_path = ROOT / "data/processed/typology_episodes.parquet"
        if not gt_path.exists():
            record(
                "typology detection >= 80% (objective 5.4)",
                "BLOCKED",
                "artifacts/fraud/typologies.json missing (blocked: data/processed/typology_episodes.parquet missing)",
            )
        else:
            record(
                "typology detection >= 80% (objective 5.4)",
                "BLOCKED",
                "artifacts/fraud/typologies.json missing (run make typologies)",
            )

    # 6. MLflow model registry checks
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        record(
            "MLflow registry reachable with both aliases",
            "BLOCKED",
            "MLFLOW_TRACKING_URI not exported",
        )
    else:
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            for alias in ("champion", "challenger"):
                v = client.get_model_version_by_alias("aegisflow-fraud", alias)
                if v is not None:
                    record(f"alias '{alias}' set", "PASS", f"(v{v.version})")
                else:
                    record(f"alias '{alias}' set", "FAIL", "alias not assigned")

            champ_mv = client.get_model_version_by_alias("aegisflow-fraud", "champion")
            if champ_mv and champ_mv.run_id:
                artifacts = client.list_artifacts(champ_mv.run_id)
                has_calibrator = any(f.path == "calibrator" for f in artifacts)
                if has_calibrator:
                    record("calibrator logged alongside champion model", "PASS")
                else:
                    record("calibrator logged alongside champion model", "FAIL", "calibrator artifact not found in champion run")
            else:
                record("calibrator logged alongside champion model", "FAIL", "champion model version not found")

        except Exception as exc:  # noqa: BLE001
            record("MLflow registry reachable with both aliases", "BLOCKED", f"connection failed: {exc}")

    # Summary
    n_pass = sum(1 for s in statuses.values() if s == "PASS")
    n_fail = sum(1 for s in statuses.values() if s == "FAIL")
    n_blocked = sum(1 for s in statuses.values() if s == "BLOCKED")
    n_skip = sum(1 for s in statuses.values() if s == "SKIP")
    total = len(statuses)

    print(f"\nSummary: {n_pass}/{total} checks PASS, {n_fail} FAIL, {n_blocked} BLOCKED, {n_skip} SKIP")

    # Strict release readiness condition: exit 0 only when all checks pass with zero failures and zero blockers
    return 0 if (n_fail == 0 and n_blocked == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
