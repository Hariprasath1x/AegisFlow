"""Supervised ML Dataset Builder for AegisFlow M2.

Constructs point-in-time feature matrices joined 1-to-1 with deterministic
multi-incident target outcomes under the three-state target taxonomy:
  - 1: Confirmed Positive Fraud
  - 0: Confirmed Negative Legitimate Control
  - None: Unreviewed, Immature, or Contradictory (excluded from training loss)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
import uuid

import numpy as np
import pandas as pd

from aegis_ml.features.definitions import FEATURE_LIST, GRAPH_FIELDS
from aegis_ml.features.offline import build_offline_features
from aegis_ml.scenarios.constants import (
    MATURITY_BUFFER_DAYS,
    OBSERVATION_CUTOFF,
)

# Canonical fraud typologies recognized by PostgreSQL ground_truth_labels
VALID_FRAUD_TYPOLOGIES = frozenset({
    "money_mule",
    "account_takeover",
    "first_party_synthetic",
    "card_not_present",
    "structuring",
    "bust_out",
    "velocity_spike",
})

# Derived outcome typologies (not valid input fraud types)
DERIVED_OUTPUT_TYPOLOGIES = frozenset({
    "ambiguous_overlap",
    "legitimate_clean",
})

# Complete set of valid primary outcome categories
OUTCOME_CATEGORIES = frozenset({
    "positive_fraud",
    "negative_clean",
    "immature",
    "direct_conflict",
    "global_contradiction",
    "temporal_contradiction",
    "unresolved_account_fraud_window",
    "unreviewed_or_subthreshold",
})


def normalize_account_id(val: Any) -> str:
    """Collision-safe, type-aware account ID canonicalization (P0-3).

    Rules:
    1. Null, NaN, None, or empty strings are rejected with ValueError.
    2. uuid.UUID instances: normalized to lowercase hyphenated string (RFC 4122).
    3. String representations of UUIDs: normalized to lowercase hyphenated string.
    4. String domain accounts: preserved exactly as-is with only whitespace stripped (case-preserving).
    5. Non-integral floats (e.g. 123.45) are rejected.
    6. Integers are stringified directly without alteration.
    """
    if val is None or pd.isna(val):
        raise ValueError("account_id cannot be null or NaN")

    if isinstance(val, uuid.UUID):
        return str(val).lower()

    s = str(val).strip()
    if not s:
        raise ValueError("account_id cannot be empty or whitespace-only")

    # Check if string is a valid UUID representation
    try:
        parsed_uuid = uuid.UUID(s)
        return str(parsed_uuid).lower()
    except (ValueError, AttributeError):
        pass

    if isinstance(val, float):
        if not val.is_integer():
            raise ValueError(f"Floating-point account_id '{val}' is invalid")
        return str(int(val))

    return s  # Case preserved: "Account-A" != "account-a"


def normalize_txn_id(val: Any) -> str:
    """Validate and normalize transaction identifier (F-04).

    Rejects:
      - None, NaN, pd.NA
      - Empty strings
      - Whitespace-only strings
    Trims surrounding whitespace from valid IDs.
    """
    if val is None or pd.isna(val):
        raise ValueError("txn_id cannot be null, NaN, or pd.NA")

    s = str(val).strip()
    if not s:
        raise ValueError("txn_id cannot be empty or whitespace-only")
    return s


def normalize_utc(ts: Any) -> datetime:
    """Validate and convert timestamp to timezone-aware UTC datetime.

    Rejects timezone-naive datetimes, missing values, and invalid timestamps.
    """
    if ts is None or pd.isna(ts):
        raise ValueError("Timestamp cannot be null, NaN, or NaT")

    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts)
        except Exception as e:
            raise ValueError(f"Invalid timestamp string {ts!r}: {e}") from e
    elif isinstance(ts, (datetime, pd.Timestamp)):
        dt = ts
    else:
        raise ValueError(f"Unsupported timestamp type {type(ts)!r}: {ts!r}")

    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(f"Timezone-naive timestamp {ts!r} is rejected; timezone required.")

    return dt.astimezone(timezone.utc)


def normalize_timestamp_series(s: pd.Series, allow_null: bool = False) -> pd.Series:
    """Validate and convert a Series of timestamps to timezone-aware UTC datetime (F-02).

    Rejects timezone-naive datetimes, invalid strings, and (if allow_null=False) nulls.
    """
    if len(s) == 0:
        return pd.Series(dtype="datetime64[ns, UTC]", index=s.index)

    if not allow_null and s.isna().any():
        raise ValueError("Timestamp column contains missing or null values (None, NaN, or NaT)")

    # When allow_null is True and every value is null/NaT, return a UTC-aware NaT Series
    if allow_null and s.isna().all():
        return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")

    if pd.api.types.is_datetime64_any_dtype(s):
        if s.dt.tz is None:
            raise ValueError("Timezone-naive datetime64 series is rejected; timezone required.")
        return s.dt.tz_convert("UTC")

    res = []
    for idx, val in s.items():
        if val is None or pd.isna(val):
            if allow_null:
                res.append(pd.NaT)
            else:
                raise ValueError(f"Timestamp at index {idx} cannot be null, NaN, or NaT")
        else:
            res.append(normalize_utc(val))
    return pd.Series(pd.to_datetime(res, utc=True), index=s.index)


def resolve_graph_snapshots(snapshots: pd.DataFrame | None) -> pd.DataFrame | None:
    """Resolve dual column aliases and validate snapshot schema contracts (P0-2, F-02).

    Precedence rules:
      - Entity column: 'account_id' > 'entity_id'
      - Timestamp column: 'snapshot_at' > 'computed_at'
    """
    if snapshots is None or len(snapshots) == 0:
        return snapshots

    snaps = snapshots.copy()
    acc_col = "account_id" if "account_id" in snaps.columns else ("entity_id" if "entity_id" in snaps.columns else None)
    time_col = "snapshot_at" if "snapshot_at" in snaps.columns else ("computed_at" if "computed_at" in snaps.columns else None)

    if acc_col is None:
        raise ValueError("Graph snapshots missing required entity column ('account_id' or 'entity_id')")
    if time_col is None:
        raise ValueError("Graph snapshots missing required timestamp column ('snapshot_at' or 'computed_at')")

    snaps["account_id"] = snaps[acc_col].apply(normalize_account_id)
    snaps["snapshot_at"] = normalize_timestamp_series(snaps[time_col], allow_null=False)
    return snaps


def verify_invariants(norm_txns: pd.DataFrame, outcomes_df: pd.DataFrame) -> None:
    """Validate mathematical and relational invariants on dataset builder outcomes (P1-1, F-03)."""
    # 0. Target Domain Validation (F-03)
    for idx, val in outcomes_df["target"].items():
        if val is None or pd.isna(val):
            continue
        if isinstance(val, (bool, np.bool_)):
            raise ValueError(f"Boolean target '{val}' at index {idx} is invalid; must be numeric 0, 1, or None")
        if isinstance(val, (int, np.integer)):
            if val not in (0, 1):
                raise ValueError(f"Target value '{val}' at index {idx} out of domain; must be 0, 1, or None")
        elif isinstance(val, (float, np.floating)):
            if val not in (0.0, 1.0) or not float(val).is_integer():
                raise ValueError(f"Target value '{val}' at index {idx} out of domain; must be 0, 1, or None")
        else:
            raise ValueError(f"Target value '{val}' of type {type(val)!r} at index {idx} is invalid; must be numeric 0, 1, or None")

    # 1. Identity & Row Count (P0-1)
    if len(outcomes_df) != len(norm_txns):
        raise ValueError(f"Outcome row count ({len(outcomes_df)}) must match transactions ({len(norm_txns)})")
    if not outcomes_df["txn_id"].is_unique:
        raise ValueError("Outcome txn_id must be strictly unique")
    txn_id_set = {str(x).strip() for x in norm_txns["txn_id"]}
    outcome_txn_id_set = {str(x).strip() for x in outcomes_df["txn_id"]}
    if txn_id_set != outcome_txn_id_set:
        raise ValueError("Outcome txn_id set does not match input transactions")
    if outcomes_df["account_id"].isna().any():
        raise ValueError("account_id cannot be null in outcomes_df")

    # 2. Target 1 (Positive Fraud)
    fraud_mask = outcomes_df["target"] == 1
    if fraud_mask.any():
        if not (outcomes_df.loc[fraud_mask, "primary_outcome"] == "positive_fraud").all():
            raise ValueError("All rows with target=1 must have primary_outcome='positive_fraud'")
        invalid_types = set(outcomes_df.loc[fraud_mask, "target_typology"]) - (VALID_FRAUD_TYPOLOGIES | {"ambiguous_overlap"})
        if invalid_types:
            raise ValueError(f"Rows with target=1 contain invalid typologies: {invalid_types}")

    # 3. Target 0 (Negative Clean)
    clean_mask = outcomes_df["target"] == 0
    if clean_mask.any():
        if not (outcomes_df.loc[clean_mask, "primary_outcome"] == "negative_clean").all():
            raise ValueError("All rows with target=0 must have primary_outcome='negative_clean'")
        clean_types = set(outcomes_df.loc[clean_mask, "target_typology"])
        if clean_types != {"legitimate_clean"}:
            raise ValueError(f"Rows with target=0 must have target_typology='legitimate_clean', got: {clean_types}")

    # 4. Target None (Excluded / Unsupervised)
    none_mask = outcomes_df["target"].isna()
    if none_mask.any():
        if not outcomes_df.loc[none_mask, "target_typology"].isna().all():
            raise ValueError("Rows with target=None must have target_typology=None")
        invalid_outcomes = set(outcomes_df.loc[none_mask, "primary_outcome"]) - {
            "immature",
            "direct_conflict",
            "global_contradiction",
            "temporal_contradiction",
            "unresolved_account_fraud_window",
            "unreviewed_or_subthreshold",
        }
        if invalid_outcomes:
            raise ValueError(f"Rows with target=None have invalid primary_outcomes: {invalid_outcomes}")


def build_supervised_training_dataset(
    transactions: pd.DataFrame,
    labels: pd.DataFrame,
    graph_snapshots: pd.DataFrame | None = None,
    accounts: pd.DataFrame | None = None,
    evaluation_time: datetime | None = None,
    maturity_buffer: timedelta = timedelta(days=MATURITY_BUFFER_DAYS),
    fraud_confidence_threshold: float = 0.85,
    clean_confidence_threshold: float = 1.0,
    propagate_account_fraud: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compile leak-free point-in-time features and three-state supervised targets.

    Parameters
    ----------
    transactions : pd.DataFrame
        Transaction records with columns: txn_id, account_id, occurred_at (or ts), amount, ...
    labels : pd.DataFrame
        Ground truth labels with columns: label_id, txn_id, account_id, label_target (or entity_type),
        is_fraud, fraud_type, confidence, labeled_at, incident_start_at, incident_end_at.
    graph_snapshots : pd.DataFrame, optional
        Pre-computed graph feature snapshots joined backward-only.
    accounts : pd.DataFrame, optional
        Account metadata (e.g. opened_on) for observation horizon validation.
    evaluation_time : datetime, optional
        Observation cutoff timestamp (UTC). Defaults to OBSERVATION_CUTOFF or max(occurred_at).
    maturity_buffer : timedelta, default 7 days
        Immature buffer window where unreviewed transactions cannot be labeled negative.
    fraud_confidence_threshold : float, default 0.85
        Minimum confidence required for positive fraud labels.
    clean_confidence_threshold : float, default 1.0
        Minimum confidence required for negative clean labels.
    propagate_account_fraud : bool, default False
        Whether account-level fraud labels propagate to transactions within the incident window
        lacking direct transaction labels. Invariant from Revision 4 defaults to False (Y=None).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        1. pit_features_df: Full feature matrix (all transactions + 19 canonical features + target metadata).
        2. supervised_df: Filtered training split (only rows where target in {0, 1}), with integer target.
        3. outcomes_df: Complete outcome resolution table.
    """
    if evaluation_time is None:
        evaluation_time = OBSERVATION_CUTOFF
    else:
        evaluation_time = normalize_utc(evaluation_time)

    # 1. Input Normalization & Integrity Checks (P0-1, P0-3)
    norm_txns = transactions.copy()
    if len(norm_txns) == 0:
        empty_outcomes = pd.DataFrame(columns=[
            "txn_id", "account_id", "target", "target_typology", "primary_outcome", "reason"
        ])
        empty_pit = build_offline_features(norm_txns, graph_snapshots)
        empty_pit["target"] = pd.Series(dtype="Int64")
        empty_pit["target_typology"] = pd.Series(dtype="object")
        empty_pit["primary_outcome"] = pd.Series(dtype="object")
        empty_pit["reason"] = pd.Series(dtype="object")
        empty_supervised = empty_pit.copy()
        return empty_pit, empty_supervised, empty_outcomes

    required_txn_cols = {"txn_id", "account_id", "amount"}
    time_col = "occurred_at" if "occurred_at" in norm_txns.columns else ("ts" if "ts" in norm_txns.columns else None)
    if time_col is None:
        raise ValueError("Transactions missing required timestamp column ('occurred_at' or 'ts')")
    missing_txn = required_txn_cols - set(norm_txns.columns)
    if missing_txn:
        raise ValueError(f"Transactions missing required columns: {missing_txn}")

    norm_txns["txn_id"] = norm_txns["txn_id"].apply(normalize_txn_id)
    norm_txns["account_id"] = norm_txns["account_id"].apply(normalize_account_id)
    norm_txns["ts"] = normalize_timestamp_series(norm_txns[time_col], allow_null=False)
    norm_txns["occurred_at"] = norm_txns["ts"]
    norm_txns["id"] = norm_txns["txn_id"]

    if not norm_txns["txn_id"].is_unique:
        raise ValueError("Input transactions contains duplicate txn_id")

    # 2. Normalize Labels
    norm_labels = labels.copy() if (labels is not None and len(labels) > 0) else pd.DataFrame(columns=[
        "label_id", "txn_id", "account_id", "label_target", "is_fraud", "fraud_type",
        "confidence", "labeled_at", "incident_start_at", "incident_end_at"
    ])

    target_col = "label_target" if "label_target" in norm_labels.columns else (
        "entity_type" if "entity_type" in norm_labels.columns else None
    )
    if target_col is None and len(norm_labels) > 0:
        raise ValueError("Labels missing required entity level column ('label_target' or 'entity_type')")

    if len(norm_labels) > 0:
        if "account_id" in norm_labels.columns:
            norm_labels["account_id"] = norm_labels["account_id"].apply(
                lambda v: normalize_account_id(v) if (v is not None and pd.notna(v)) else None
            )
        else:
            norm_labels["account_id"] = None

        if "txn_id" in norm_labels.columns:
            norm_labels["txn_id"] = norm_labels["txn_id"].apply(
                lambda v: str(v).strip() if (v is not None and pd.notna(v)) else None
            )
        else:
            norm_labels["txn_id"] = None

        norm_labels["labeled_at"] = normalize_timestamp_series(norm_labels["labeled_at"], allow_null=False)
        if "incident_start_at" in norm_labels.columns:
            norm_labels["incident_start_at"] = normalize_timestamp_series(norm_labels["incident_start_at"], allow_null=True)
        else:
            norm_labels["incident_start_at"] = pd.Series(pd.NaT, index=norm_labels.index, dtype="datetime64[ns, UTC]")

        if "incident_end_at" in norm_labels.columns:
            norm_labels["incident_end_at"] = normalize_timestamp_series(norm_labels["incident_end_at"], allow_null=True)
        else:
            norm_labels["incident_end_at"] = pd.Series(pd.NaT, index=norm_labels.index, dtype="datetime64[ns, UTC]")

        norm_labels["entity_target"] = norm_labels[target_col].str.lower()
    else:
        norm_labels["entity_target"] = pd.Series(dtype=str)

    # 3. Normalize Account Metadata (if provided)
    acc_opened: dict[str, datetime] = {}
    if accounts is not None and len(accounts) > 0:
        acc_df = accounts.copy()
        if "account_id" in acc_df.columns and "opened_on" in acc_df.columns:
            for _, r in acc_df.iterrows():
                try:
                    aid = normalize_account_id(r["account_id"])
                    acc_opened[aid] = normalize_utc(r["opened_on"])
                except Exception:
                    pass

    # 4. Extract Leak-Free PIT Features (offline.py)
    norm_snaps = resolve_graph_snapshots(graph_snapshots)
    pit_features_df = build_offline_features(
        transactions=norm_txns,
        graph_snapshots=norm_snaps,
    )

    # Validate Feature Builder Identity and Uniqueness Contract (P0-1)
    if len(pit_features_df) != len(norm_txns):
        raise ValueError("Feature row count mismatch")
    if not pit_features_df["txn_id"].is_unique:
        raise ValueError("Feature txn_id must be unique")
    if set(pit_features_df["txn_id"]) != set(norm_txns["txn_id"]):
        raise ValueError("Feature txn_id set mismatch")

    # 5. Label Pre-Filtering & Quarantine Scanning (Anti-Leakage)
    avail_labels = norm_labels[norm_labels["labeled_at"] <= evaluation_time].copy() if len(norm_labels) > 0 else norm_labels

    direct_labels = avail_labels[avail_labels["entity_target"] == "transaction"].copy()
    account_labels = avail_labels[avail_labels["entity_target"] == "account"].copy()

    qual_acc_fraud = account_labels[
        (account_labels["is_fraud"] == True) &
        (account_labels["confidence"] >= fraud_confidence_threshold)
    ]
    qual_acc_clean = account_labels[
        (account_labels["is_fraud"] == False) &
        (account_labels["confidence"] >= clean_confidence_threshold)
    ]

    # Global Account Quarantine Pre-Scan
    quarantined_accounts: set[str] = set()
    if len(account_labels) > 0:
        for acc_id, group in account_labels.groupby("account_id"):
            if acc_id is None:
                continue
            acc_fraud_quals = qual_acc_fraud[qual_acc_fraud["account_id"] == acc_id]
            acc_clean_quals = qual_acc_clean[qual_acc_clean["account_id"] == acc_id]

            has_timeless_fraud = (acc_fraud_quals["incident_start_at"].isna() & acc_fraud_quals["incident_end_at"].isna()).any()
            has_bounded_clean = (acc_clean_quals["incident_start_at"].notna() & acc_clean_quals["incident_end_at"].notna()).any()
            has_timeless_clean = (acc_clean_quals["incident_start_at"].isna() & acc_clean_quals["incident_end_at"].isna()).any()
            has_bounded_fraud = (acc_fraud_quals["incident_start_at"].notna() & acc_fraud_quals["incident_end_at"].notna()).any()

            has_timeless_conflict = has_timeless_fraud and has_timeless_clean
            has_timeless_bounded_conflict = (has_timeless_fraud and has_bounded_clean) or (has_timeless_clean and has_bounded_fraud)

            if has_timeless_conflict or has_timeless_bounded_conflict:
                quarantined_accounts.add(acc_id)

    # 6. Resolve Outcomes per Transaction
    outcomes: list[dict[str, Any]] = []

    for _, txn in norm_txns.iterrows():
        t_id = txn["txn_id"]
        a_id = txn["account_id"]
        t_time = txn["ts"]

        # Check maturity buffer
        is_mature = (t_time + maturity_buffer) <= evaluation_time

        # Check Direct Transaction Labels
        t_direct = direct_labels[direct_labels["txn_id"] == t_id] if len(direct_labels) > 0 else pd.DataFrame()
        qual_dir_fraud = t_direct[(t_direct["is_fraud"] == True) & (t_direct["confidence"] >= fraud_confidence_threshold)] if len(t_direct) > 0 else pd.DataFrame()
        qual_dir_clean = t_direct[(t_direct["is_fraud"] == False) & (t_direct["confidence"] >= clean_confidence_threshold)] if len(t_direct) > 0 else pd.DataFrame()

        # Check Active Account Incidents at t_time
        a_fraud = qual_acc_fraud[
            (qual_acc_fraud["account_id"] == a_id) &
            (
                (qual_acc_fraud["incident_start_at"].isna() & qual_acc_fraud["incident_end_at"].isna()) |
                (
                    (qual_acc_fraud["incident_start_at"].isna() | (qual_acc_fraud["incident_start_at"] <= t_time)) &
                    (qual_acc_fraud["incident_end_at"].isna() | (qual_acc_fraud["incident_end_at"] >= t_time))
                )
            )
        ] if len(qual_acc_fraud) > 0 else pd.DataFrame()

        a_clean = qual_acc_clean[
            (qual_acc_clean["account_id"] == a_id) &
            (
                (qual_acc_clean["incident_start_at"].isna() & qual_acc_clean["incident_end_at"].isna()) |
                (
                    (qual_acc_clean["incident_start_at"].isna() | (qual_acc_clean["incident_start_at"] <= t_time)) &
                    (qual_acc_clean["incident_end_at"].isna() | (qual_acc_clean["incident_end_at"] >= t_time))
                )
            )
        ] if len(qual_acc_clean) > 0 else pd.DataFrame()

        # Resolution Logic (Priority-Ordered per §2.3)

        # Case 0: Direct Fraud vs Direct Clean Contradiction (F-01)
        if (not qual_dir_fraud.empty) and (not qual_dir_clean.empty):
            outcomes.append({
                "txn_id": t_id,
                "account_id": a_id,
                "target": None,
                "target_typology": None,
                "primary_outcome": "direct_conflict",
                "reason": "direct_fraud_vs_direct_clean_conflict",
            })
            continue

        # Case A: Qualifying Direct Fraud Label
        if not qual_dir_fraud.empty:
            best_dir = qual_dir_fraud.sort_values(by=["confidence", "labeled_at"], ascending=[False, False]).iloc[0]
            outcomes.append({
                "txn_id": t_id,
                "account_id": a_id,
                "target": 1,
                "target_typology": best_dir["fraud_type"],
                "primary_outcome": "positive_fraud",
                "reason": "direct_transaction_fraud",
            })
            continue

        # Case B: Qualifying Direct Clean Label
        if not qual_dir_clean.empty:
            if not a_fraud.empty:
                # Direct clean conflicts with active account fraud -> Fail Closed
                outcomes.append({
                    "txn_id": t_id,
                    "account_id": a_id,
                    "target": None,
                    "target_typology": None,
                    "primary_outcome": "direct_conflict",
                    "reason": "direct_clean_vs_account_fraud_conflict",
                })
            else:
                outcomes.append({
                    "txn_id": t_id,
                    "account_id": a_id,
                    "target": 0,
                    "target_typology": "legitimate_clean",
                    "primary_outcome": "negative_clean",
                    "reason": "direct_transaction_clean",
                })
            continue

        # Case C: Global Account Quarantine (Affects transactions lacking direct labels)
        if a_id in quarantined_accounts:
            outcomes.append({
                "txn_id": t_id,
                "account_id": a_id,
                "target": None,
                "target_typology": None,
                "primary_outcome": "global_contradiction",
                "reason": "account_quarantined_by_contradiction",
            })
            continue

        # Case D: Active Account Overlap Contradiction (Simultaneous Clean & Fraud active)
        if (not a_fraud.empty) and (not a_clean.empty):
            outcomes.append({
                "txn_id": t_id,
                "account_id": a_id,
                "target": None,
                "target_typology": None,
                "primary_outcome": "temporal_contradiction",
                "reason": "active_fraud_and_clean_incident_overlap",
            })
            continue

        # Case E: Active Qualifying Account Fraud
        if not a_fraud.empty:
            if propagate_account_fraud:
                distinct_types = set(a_fraud["fraud_type"].dropna())
                typology = list(distinct_types)[0] if len(distinct_types) == 1 else "ambiguous_overlap"
                outcomes.append({
                    "txn_id": t_id,
                    "account_id": a_id,
                    "target": 1,
                    "target_typology": typology,
                    "primary_outcome": "positive_fraud",
                    "reason": "account_incident_fraud",
                })
            else:
                # Revision 4 invariant (§2.1 & §2.3): Account-fraud does NOT auto-label transactions
                outcomes.append({
                    "txn_id": t_id,
                    "account_id": a_id,
                    "target": None,
                    "target_typology": None,
                    "primary_outcome": "unresolved_account_fraud_window",
                    "reason": "account_incident_lacks_direct_txn_label",
                })
            continue

        # Case F: Active Qualifying Account Clean
        if not a_clean.empty:
            # Validate observation horizon if account opened_on is known
            opened_on = acc_opened.get(a_id)
            if opened_on is not None and t_time < opened_on:
                outcomes.append({
                    "txn_id": t_id,
                    "account_id": a_id,
                    "target": None,
                    "target_typology": None,
                    "primary_outcome": "unreviewed_or_subthreshold",
                    "reason": "occurred_before_account_opened",
                })
                continue

            if not is_mature:
                outcomes.append({
                    "txn_id": t_id,
                    "account_id": a_id,
                    "target": None,
                    "target_typology": None,
                    "primary_outcome": "immature",
                    "reason": "clean_account_txn_in_maturity_buffer",
                })
                continue

            outcomes.append({
                "txn_id": t_id,
                "account_id": a_id,
                "target": 0,
                "target_typology": "legitimate_clean",
                "primary_outcome": "negative_clean",
                "reason": "account_incident_clean",
            })
            continue

        # Case G: Immature Transaction (Unreviewed and within maturity window)
        if not is_mature:
            outcomes.append({
                "txn_id": t_id,
                "account_id": a_id,
                "target": None,
                "target_typology": None,
                "primary_outcome": "immature",
                "reason": "within_maturity_buffer_unlabeled",
            })
            continue

        # Case H: Unreviewed Mature or Sub-threshold
        outcomes.append({
            "txn_id": t_id,
            "account_id": a_id,
            "target": None,
            "target_typology": None,
            "primary_outcome": "unreviewed_or_subthreshold",
            "reason": "no_qualifying_ground_truth_labels",
        })

    outcomes_df = pd.DataFrame(outcomes)

    # 7. Verify Invariants (P1-1)
    verify_invariants(norm_txns, outcomes_df)

    # 8. Canonical Ordering & One-to-One Join (P0-1)
    pit_features_df = pit_features_df.sort_values(by=["ts", "id"]).reset_index(drop=True)
    outcomes_df = outcomes_df.sort_values(by=["txn_id"]).reset_index(drop=True)

    pit_features_df = pit_features_df.merge(
        outcomes_df[["txn_id", "target", "target_typology", "primary_outcome", "reason"]],
        on="txn_id",
        how="inner",
        validate="one_to_one",
    )

    # 9. Supervised Training Subset (Filtered to Validated Targets Only)
    supervised_mask = pit_features_df["target"].notna()
    supervised_df = pit_features_df[supervised_mask].copy()
    supervised_df["target"] = supervised_df["target"].astype(int)

    return pit_features_df, supervised_df, outcomes_df
