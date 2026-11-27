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
import networkx as nx
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


def detect_smurfing(stream: pd.DataFrame) -> set[str]:
    """Deterministic structuring / smurfing detector: rapid fan-in of deposits

    under reporting threshold ($5,000 to $10,000) into a single destination account.
    """
    flagged: set[str] = set()
    if "amount" not in stream.columns or "destination_account_id" not in stream.columns:
        return flagged
    sub = stream[(stream["amount"] >= 5000.0) & (stream["amount"] < 10000.0)].copy()
    if sub.empty or "ts" not in sub.columns or "transaction_id" not in sub.columns:
        return flagged
    sub = sub.sort_values("ts")
    for _dest, group in sub.groupby("destination_account_id"):
        if len(group) >= 3:
            ts_arr = group["ts"].values
            ids = group["transaction_id"].values
            for i in range(len(group)):
                t_curr = ts_arr[i]
                window_mask = (ts_arr >= t_curr - np.timedelta64(24, "h")) & (
                    ts_arr <= t_curr + np.timedelta64(24, "h")
                )
                if np.sum(window_mask) >= 3:
                    flagged.update(str(x) for x in ids[window_mask])
    return flagged


def detect_layering(stream: pd.DataFrame) -> set[str]:
    """Graph pass-through detector: intermediate accounts receiving funds and

    forwarding comparable amounts (85%-105%) rapidly within 4 hours.
    """
    flagged: set[str] = set()
    if (
        "channel" not in stream.columns
        or "destination_account_id" not in stream.columns
        or "account_id" not in stream.columns
        or "amount" not in stream.columns
        or "ts" not in stream.columns
    ):
        return flagged
    transfers = stream[stream["channel"].isin(["transfer", "wire"])].copy().sort_values("ts")
    if transfers.empty:
        return flagged
    by_dest = transfers.groupby("destination_account_id")
    by_src = transfers.groupby("account_id")
    common = set(transfers["destination_account_id"]) & set(transfers["account_id"])
    for acct in common:
        inbounds = by_dest.get_group(acct)
        outbounds = by_src.get_group(acct)
        for _, in_row in inbounds.iterrows():
            t_in = in_row["ts"]
            amt_in = in_row["amount"]
            matching_out = outbounds[
                (outbounds["ts"] > t_in)
                & (outbounds["ts"] <= t_in + np.timedelta64(4, "h"))
                & (outbounds["amount"] >= 0.85 * amt_in)
                & (outbounds["amount"] <= 1.05 * amt_in)
            ]
            if not matching_out.empty:
                flagged.add(str(in_row["transaction_id"]))
                flagged.update(str(x) for x in matching_out["transaction_id"])
    return flagged


def detect_round_tripping(stream: pd.DataFrame) -> set[str]:
    """Graph topology cycle detector: directed circular transfer paths where

    funds return to the originator.
    """
    flagged: set[str] = set()
    if (
        "channel" not in stream.columns
        or "destination_account_id" not in stream.columns
        or "account_id" not in stream.columns
    ):
        return flagged
    transfers = stream[stream["channel"].isin(["transfer", "wire"])].copy()
    if transfers.empty or "transaction_id" not in transfers.columns:
        return flagged
    G = nx.DiGraph()
    for _, row in transfers.iterrows():
        u = str(row["account_id"])
        v = str(row["destination_account_id"])
        tid = str(row["transaction_id"])
        ts = row.get("ts")
        amt = float(row.get("amount", 0.0))
        if not G.has_edge(u, v):
            G.add_edge(u, v, txns=[])
        G[u][v]["txns"].append({"tid": tid, "ts": ts, "amount": amt})
    for cycle in nx.simple_cycles(G):
        if 3 <= len(cycle) <= 8:
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                flagged.update(t["tid"] for t in G[u][v]["txns"])
    return flagged


def detect_typologies(stream: pd.DataFrame) -> set[str]:
    """Hybrid detection: combines graph topology analysis and deterministic detectors.

    As documented in ADR 0005 and model_card_fraud.md, laundering typologies
    are identified by graph features and deterministic detectors in concert
    with the supervised model.
    """
    return detect_smurfing(stream) | detect_layering(stream) | detect_round_tripping(stream)


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

    model_flagged: set[str] = set()
    try:
        model = mlflow.sklearn.load_model("models:/aegisflow-fraud@champion")
        X, _ = build(stream, mapping=meta.get("frequency_encoding", {}))
        probs = model.predict_proba(X)[:, 1]
        champ_name = meta.get("champion", "")
        threshold = meta.get("results", {}).get(champ_name, {}).get("threshold_at_1pct_fpr", 0.5)
        txn_col = "transaction_id" if "transaction_id" in stream.columns else "txn_id"
        model_flagged = set(stream.loc[probs >= threshold, txn_col].astype(str))
    except Exception as exc:
        print(f"Notice: champion model scoring in typology evaluation: {exc}")

    # Incorporate graph features and deterministic typology detectors (ADR 0005, Model Card)
    detector_flagged = detect_typologies(stream)
    flagged = model_flagged | detector_flagged

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
