"""Objective 5.4: catch at least 80% of the injected laundering typologies.
The supervised model is trained on IEEE-CIS card fraud; smurfing and layering are
a different problem and live in the synthetic stream M1's generator produces with
ground truth. This module scores the generator's labelled typology episodes with
the champion model PLUS the graph features and reports detection per typology, so
the 80% claim is measured rather than asserted.

python -m aegis_ml.evaluation.typology_eval
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Sequence
import numpy as np
import pandas as pd

from .reports import write_report

# NOTE: training.matrix imports M2's feature definitions, which pull in Redis and
# the graph job. The import lives inside main() so episode_detection stays unit
# testable with nothing but pandas installed.

TYPOLOGIES = ["smurfing", "layering", "round_tripping"]
GROUND_TRUTH = pathlib.Path("data/processed/typology_episodes.parquet")
STREAM_PATH = pathlib.Path("data/processed/synthetic_transactions.parquet")
METRICS_PATH = pathlib.Path("artifacts/fraud/metrics.json")
REPORTS_DIR = pathlib.Path("reports")
ARTIFACTS_DIR = pathlib.Path("artifacts/fraud")


def episode_detection(
    episodes: pd.DataFrame,
    flagged_ids: set[str] | set[Any] | Sequence[Any],
    min_share: float = 0.30,
) -> dict[str, Any]:
    """An episode counts as detected when at least min_share of its member
    transactions were flagged. One lucky hit out of forty smurfing deposits is not
    a detection, and reporting it as one is how these numbers get inflated.
    """
    if not isinstance(episodes, pd.DataFrame):
        raise ValueError("episodes must be a pandas DataFrame")
    if not isinstance(flagged_ids, (set, frozenset, list, tuple)):
        raise ValueError("flagged_ids must be a set, list, or tuple")
    if not isinstance(min_share, (int, float)) or min_share <= 0 or min_share > 1.0:
        raise ValueError("min_share must be a float between 0 and 1.0")

    flagged_set = set(map(str, flagged_ids))
    out: dict[str, Any] = {}
    for typ in TYPOLOGIES:
        if "typology" not in episodes.columns:
            out[typ] = {"episodes": 0, "detected": 0, "rate": None}
            continue
        rows = episodes[episodes["typology"] == typ]
        if rows.empty:
            out[typ] = {"episodes": 0, "detected": 0, "rate": None}
            continue
        detected = 0
        for _, ep in rows.iterrows():
            txns = ep.get("transaction_ids", [])
            if txns is None:
                members = set()
            elif isinstance(txns, (list, tuple, np.ndarray, set)):
                members = set(map(str, txns))
            else:
                members = {str(txns)}
            if members and (len(members & flagged_set) / len(members)) >= min_share:
                detected += 1
        out[typ] = {
            "episodes": int(len(rows)),
            "detected": detected,
            "rate": detected / len(rows),
        }
    rates = [v["rate"] for v in out.values() if v["rate"] is not None]
    out["overall"] = {"rate": float(np.mean(rates)) if rates else 0.0}
    return out


def main() -> int:
    if not GROUND_TRUTH.exists():
        print(
            f"{GROUND_TRUTH} missing - run the M1 generator with typology injection "
            f"(`make generate`) before this evaluation"
        )
        return 2

    if not STREAM_PATH.exists():
        print(
            f"{STREAM_PATH} missing - synthetic transaction stream required for typology evaluation"
        )
        return 2

    if not METRICS_PATH.exists():
        print(f"{METRICS_PATH} missing - run `make train-fraud` before this evaluation")
        return 2

    import mlflow
    from ..training.matrix import build

    episodes = pd.read_parquet(GROUND_TRUTH)
    stream = pd.read_parquet(STREAM_PATH)
    meta = json.loads(METRICS_PATH.read_text())

    model = mlflow.sklearn.load_model("models:/aegisflow-fraud@champion")

    X, _ = build(stream, mapping=meta["frequency_encoding"])
    probs = model.predict_proba(X)[:, 1]
    champ_name = meta["champion"]
    threshold = meta["results"][champ_name]["threshold_at_1pct_fpr"]

    txn_col = "transaction_id" if "transaction_id" in stream.columns else "txn_id"
    flagged = set(stream.loc[probs >= threshold, txn_col].astype(str))

    result = episode_detection(episodes, flagged)
    overall = result["overall"]["rate"]

    gates = [("typology detection >= 80% (objective 5.4)", overall >= 0.80, f"{overall:.1%}")]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    write_report(
        str(REPORTS_DIR / "m3_typologies.html"),
        "AegisFlow M3 - laundering typologies",
        [
            {
                "heading": "Detection by typology",
                "rows": [{"typology": k, **v} for k, v in result.items() if k != "overall"],
                "note": (
                    "An episode counts only when at least 30% of its member "
                    "transactions were flagged."
                ),
            }
        ],
        gates=gates,
        out_json=str(REPORTS_DIR / "m3_typologies.json"),
    )

    (ARTIFACTS_DIR / "typologies.json").write_text(
        json.dumps(result, indent=2, default=float)
    )

    print(f"typology detection {overall:.1%} (gate 80%)")
    return 0 if overall >= 0.80 else 1


if __name__ == "__main__":
    raise SystemExit(main())
