"""DVC stage: train champion and challenger, calibrate, evaluate, register.

python -m aegis_ml.training.train_fraud
Order matters and is not negotiable: split by time -> build features point-in-time
-> fit the frequency encoder on TRAIN only -> train -> fit the calibrator on a
held-out slice of TRAIN -> evaluate once on TEST. Any other order leaks.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any
import mlflow
import numpy as np
import pandas as pd
import yaml

from ..evaluation import rules_baseline
from ..evaluation.fairness import disparate_impact, group_rates
from ..evaluation.gates import check_gate
from ..evaluation.metrics import (
    amount_bands,
    best_threshold,
    cohort_slices,
    cost_curve,
    lift_over,
    pr_auc,
    recall_at_fpr,
)
from ..evaluation.reports import line_chart, write_report
from ..models.calibration import Calibrator, expected_calibration_error, reliability
from ..models.fraud_models import train_lightgbm, train_xgboost
from .matrix import CATEGORICAL, build, feature_names, load_transactions
from .splits import check_no_time_leak, label_delay_mask, time_split

try:
    from .registry import register_candidate
except ImportError:
    register_candidate = None

OUT = pathlib.Path("artifacts/fraud")


def prepare(params: dict[str, Any], df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Prepare time-split and leak-checked training, calibration, and test matrices."""
    if df is None:
        df = load_transactions()
    else:
        df = df.copy()

    df["split"] = time_split(
        df["ts"],
        params["train_weeks"],
        params["test_weeks"],
        params["embargo_hours"],
    )
    df = df[df["split"] != "drop"].reset_index(drop=True)
    known = label_delay_mask(
        df["ts"],
        df["label_reported_at"],
        as_of=df["ts"].max(),
        horizon_days=params["label_horizon_days"],
    )
    df = df[known | (df["split"] == "test")].reset_index(drop=True)
    train, test = df[df.split == "train"], df[df.split == "test"]
    problems = check_no_time_leak(train["ts"], test["ts"])
    if problems:
        raise AssertionError("time leakage: " + "; ".join(problems))

    X_train, mapping = build(train)
    X_test, _ = build(test, mapping=mapping)

    # The calibration slice is the LAST 15% of training time, never random rows
    cut = int(len(X_train) * (1 - params["calibration_fraction"]))
    return {
        "df": df,
        "train": train,
        "test": test,
        "mapping": mapping,
        "X_fit": X_train.iloc[:cut],
        "y_fit": train["is_fraud"].to_numpy()[:cut],
        "X_cal": X_train.iloc[cut:],
        "y_cal": train["is_fraud"].to_numpy()[cut:],
        "X_test": X_test,
        "y_test": test["is_fraud"].to_numpy(),
    }


def evaluate(
    name: str,
    y_test: np.ndarray,
    probs: np.ndarray,
    test: pd.DataFrame,
    baseline_recall: float,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate calibrated predictions on test data across metrics and cost curve."""
    at_fpr = recall_at_fpr(y_test, probs, params["max_fpr"])
    curve = cost_curve(
        y_test,
        probs,
        test["amount"].to_numpy(),
        params["review_cost"],
        params["friction_cost"],
    )
    return {
        "model": name,
        "recall_at_1pct_fpr": at_fpr["recall"],
        "threshold_at_1pct_fpr": at_fpr["threshold"],
        "pr_auc": pr_auc(y_test, probs),
        "lift_over_rules": lift_over(baseline_recall, at_fpr["recall"]),
        "ece": expected_calibration_error(y_test, probs),
        "cost_optimal": best_threshold(curve),
        "cost_curve": curve,
    }


def main(argv: list[str] | None = None) -> int:
    """Run end-to-end training, calibration, evaluation, reporting, and gating."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    a = ap.parse_args(argv)
    params = yaml.safe_load(open(a.params))["fraud"]
    OUT.mkdir(parents=True, exist_ok=True)
    d = prepare(params)
    test, y_test = d["test"], d["y_test"]
    base_scores = rules_baseline.score(test)
    base = recall_at_fpr(y_test, base_scores, params["max_fpr"])

    with mlflow.start_run(run_name="fraud-champion-challenger") as run:
        mlflow.log_params({
            **{k: v for k, v in params.items() if not isinstance(v, dict)},
            "features": len(feature_names()),
            "train_rows": len(d["X_fit"]),
            "cal_rows": len(d["X_cal"]),
            "test_rows": len(d["X_test"]),
            "test_fraud_rate": float(y_test.mean()),
        })
        mlflow.log_metric("rules_baseline_recall", base["recall"])
        results, models, calibrated = {}, {}, {}
        for name, fn, key in (
            ("xgboost", train_xgboost, "xgboost"),
            ("lightgbm", train_lightgbm, "lightgbm"),
        ):
            model, _ = fn(d["X_fit"], d["y_fit"], d["X_cal"], d["y_cal"], params[key])
            cal = Calibrator().fit(model.predict_proba(d["X_cal"])[:, 1], d["y_cal"])
            probs = cal.transform(model.predict_proba(d["X_test"])[:, 1])
            models[name], calibrated[name] = model, cal
            results[name] = evaluate(name, y_test, probs, test, base["recall"], params)
            results[name]["_probs"] = probs
            mlflow.log_metrics({
                f"{name}_{k}": v
                for k, v in results[name].items()
                if isinstance(v, float)
            })

        champion = max(results, key=lambda n: results[n]["recall_at_1pct_fpr"])
        r = results[champion]
        probs = r.pop("_probs")
        results[[n for n in results if n != champion][0]].pop("_probs", None)
        mlflow.set_tag("champion", champion)
        mlflow.log_metrics({k: v for k, v in r.items() if isinstance(v, float)})

        flags = probs >= r["threshold_at_1pct_fpr"]
        cohorts = {
            **amount_bands(test["amount"]),
            **{
                f"channel_{c}": (test["channel"] == c).to_numpy()
                for c in test["channel"].unique()[:4]
            },
        }
        slices = cohort_slices(y_test, probs, cohorts, params["max_fpr"])
        rates = group_rates(y_test, flags, cohorts)
        di = disparate_impact(rates)

        gates = [
            check_gate("recall_at_1pct_fpr", r["recall_at_1pct_fpr"]),
            check_gate("lift_over_rules", r["lift_over_rules"]),
            check_gate("pr_auc", r["pr_auc"]),
            check_gate("ece", r["ece"]),
        ]
        gate_rows = [(detail.split("=")[0], ok, detail) for ok, detail in gates]
        report = write_report(
            "reports/m3_fraud.html",
            "AegisFlow M3 - fraud model evaluation",
            [
                {
                    "heading": f"Champion: {champion}",
                    "metrics": {
                        k: v
                        for k, v in r.items()
                        if isinstance(v, (int, float, str))
                    },
                    "note": (
                        "Accuracy is deliberately absent. At a 3.5% fraud rate, "
                        "approving everything scores 96.5%."
                    ),
                },
                {"heading": "Rules baseline (what we must beat)", "metrics": base},
                {
                    "heading": "Calibration",
                    "rows": reliability(y_test, probs),
                    "svg": line_chart(
                        {
                            "predicted = observed": [(0, 0), (1, 1)],
                            "observed": [
                                (x["predicted"], x["observed"])
                                for x in reliability(y_test, probs)
                            ],
                        },
                        (0, 1),
                        (0, 1),
                        "predicted probability",
                        "observed fraud rate",
                    ),
                },
                {
                    "heading": "Cost curve",
                    "metrics": r["cost_optimal"],
                    "svg": line_chart(
                        {
                            "total cost": [
                                (c["flag_rate"], c["total_cost"])
                                for c in r["cost_curve"]
                            ]
                        },
                        (0, 1),
                        (0, max(c["total_cost"] for c in r["cost_curve"])),
                        "flag rate",
                        "expected cost",
                    ),
                    "note": (
                        "The cost-optimal threshold is what M4's threshold API is "
                        "seeded with; it is almost never 0.5."
                    ),
                },
                {
                    "heading": "Cohort slices (recall at 1% FPR)",
                    "rows": [
                        {
                            "cohort": k,
                            **{
                                kk: vv
                                for kk, vv in v.items()
                                if kk in ("n", "recall", "fpr")
                            },
                        }
                        for k, v in slices.items()
                    ],
                },
                {
                    "heading": "Disparate-impact screen",
                    "metrics": di,
                    "note": (
                        "A gap that tracks the cohort fraud rate is expected. An "
                        "unexplained gap goes in the model card, not in a silent fix."
                    ),
                },
            ],
            gates=gate_rows,
            out_json="reports/m3_fraud.json",
        )
        mlflow.log_artifact(report)
        payload = {
            "champion": champion,
            "feature_names": feature_names(),
            "categorical": CATEGORICAL,
            "frequency_encoding": d["mapping"],
            "results": {
                n: {k: v for k, v in res.items() if k != "cost_curve"}
                for n, res in results.items()
            },
            "baseline": base,
            "disparate_impact": di,
        }
        (OUT / "metrics.json").write_text(json.dumps(payload, indent=2, default=float))
        mlflow.log_dict(payload, "model_meta.json")

        failed = [detail for ok, detail in gates if not ok]
        if failed:
            print("GATE FAILED: " + "; ".join(failed))
            return 1

        if register_candidate is not None:
            print(
                register_candidate(
                    run.info.run_id,
                    models[champion],
                    calibrated[champion],
                    feature_names(),
                    r,
                ).reason
            )
        else:
            print("Step 3.11 candidate registration deferred (registry module not yet implemented).")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
