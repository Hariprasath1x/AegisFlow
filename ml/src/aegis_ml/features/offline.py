"""Offline point-in-time feature computation for AegisFlow M2.

Computes all 19 canonical features on historical transaction DataFrames
using event-time semantics, two-pointer sliding windows, Welford's online
algorithm, and backward-only graph snapshot joins.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from aegis_ml.features import definitions as D

# 64-bit signed integer bounds
_INT64_MIN = -9_223_372_036_854_775_808
_INT64_MAX = 9_223_372_036_854_775_807


def _validate_and_normalize_timestamps(s: pd.Series) -> pd.Series:
    """Validate that every timestamp is present, valid, and timezone-aware.

    Converts to UTC and preserves microsecond precision.
    Rejects timezone-naive datetimes, missing values, and unparseable values.
    """
    if s.isna().any():
        raise ValueError("Timestamp column contains missing or null values (None, NaN, or NaT).")

    # If already a timezone-aware datetime64 series
    if pd.api.types.is_datetime64_any_dtype(s):
        if s.dt.tz is None:
            raise ValueError("Timezone-naive datetime64 series is rejected; timezone required.")
        return s.dt.tz_convert("UTC")

    # For object/string series, inspect each element explicitly
    normalized = []
    for idx, val in s.items():
        if isinstance(val, str):
            try:
                dt = datetime.fromisoformat(val)
            except Exception as e:
                raise ValueError(f"Invalid timestamp string {val!r} at index {idx}: {e}")
        elif isinstance(val, (datetime, pd.Timestamp)):
            dt = val
        else:
            raise ValueError(f"Invalid timestamp value of type {type(val)!r} at index {idx}: {val!r}")

        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            raise ValueError(f"Timezone-naive timestamp {val!r} at index {idx} is rejected; timezone required.")

        dt_utc = dt.astimezone(timezone.utc)
        normalized.append(dt_utc)

    return pd.to_datetime(normalized, utc=True)


def _validate_existing_amount_minor(s: pd.Series) -> pd.Series:
    """Validate that pre-existing amount_minor values are valid 64-bit integer minor units.

    Rejects NaN, +/-Infinity, fractional minor units, non-numeric strings,
    and values outside the 64-bit signed integer range.
    """
    if s.isna().any():
        raise ValueError("Existing amount_minor column contains missing or null values (NaN/None).")

    validated: list[int] = []
    for idx, val in s.items():
        if isinstance(val, (int, np.integer)):
            int_val = int(val)
        elif isinstance(val, (float, np.floating)):
            if not math.isfinite(val):
                raise ValueError(f"Non-finite amount_minor value {val!r} at index {idx}")
            if val != math.floor(val):
                raise ValueError(f"Fractional amount_minor value {val!r} at index {idx} is rejected")
            int_val = int(val)
        elif isinstance(val, Decimal):
            if not val.is_finite():
                raise ValueError(f"Non-finite Decimal amount_minor value {val!r} at index {idx}")
            if val != val.to_integral_value():
                raise ValueError(f"Fractional Decimal amount_minor value {val!r} at index {idx} is rejected")
            int_val = int(val.to_integral_value())
        elif isinstance(val, str):
            try:
                d = Decimal(val)
            except Exception as e:
                raise ValueError(f"Invalid numeric string for amount_minor at index {idx}: {val!r} ({e})")
            if not d.is_finite():
                raise ValueError(f"Non-finite amount_minor string {val!r} at index {idx}")
            if d != d.to_integral_value():
                raise ValueError(f"Fractional amount_minor string {val!r} at index {idx} is rejected")
            int_val = int(d.to_integral_value())
        else:
            raise TypeError(f"Unsupported amount_minor type {type(val)!r} at index {idx}: {val!r}")

        if int_val < _INT64_MIN or int_val > _INT64_MAX:
            raise ValueError(f"amount_minor value {int_val} out of 64-bit signed integer range at index {idx}")

        validated.append(int_val)

    return pd.Series(validated, index=s.index, dtype="int64")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare raw transaction DataFrame for feature building.

    Validates and normalizes timestamps and amounts.
    Orders transactions by (ts, id).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)!r}")

    f = df.copy()

    # 1. Resolve timestamp column
    if "ts" not in f.columns:
        if "occurred_at" in f.columns:
            f["ts"] = f["occurred_at"]
        else:
            raise ValueError("DataFrame must contain timestamp column 'ts' or 'occurred_at'")

    f["ts"] = _validate_and_normalize_timestamps(f["ts"])

    # 2. Resolve amount_minor
    if "amount_minor" in f.columns:
        f["amount_minor"] = _validate_existing_amount_minor(f["amount_minor"])
    elif "amount" in f.columns:
        f["amount_minor"] = f["amount"].apply(D.amount_minor).astype("int64")
    else:
        raise ValueError("DataFrame must contain either 'amount_minor' or 'amount'")

    # 3. Resolve identifier
    if "id" not in f.columns:
        if "txn_id" in f.columns:
            f["id"] = f["txn_id"].astype(str)
        else:
            f["id"] = f.index.astype(str)
    else:
        f["id"] = f["id"].astype(str)

    # 4. Resolve account_id
    if "account_id" not in f.columns:
        raise ValueError("DataFrame missing required 'account_id' column")
    f["account_id"] = f["account_id"].astype(str)

    # 5. Normalize device_id and ip_address if present
    if "device_id" in f.columns:
        f["device_id"] = f["device_id"].where(f["device_id"].notna(), None)
    else:
        f["device_id"] = None

    if "ip_address" in f.columns:
        f["ip_address"] = f["ip_address"].where(f["ip_address"].notna(), None)
    elif "ip" in f.columns:
        f["ip_address"] = f["ip"].where(f["ip"].notna(), None)
    else:
        f["ip_address"] = None

    # 6. Strict event-time sort by (ts, id)
    f = f.sort_values(by=["ts", "id"]).reset_index(drop=True)
    return f


def _windowed_distinct(timestamps: np.ndarray, keys: Sequence[Any], window_s: float) -> np.ndarray:
    """Sliding two-pointer window distinct-count algorithm.

    Window: (ts - window_s, ts]
    Includes current transaction if non-null.
    """
    n = len(timestamps)
    out = np.zeros(n, dtype=np.float64)
    live: dict[Any, int] = defaultdict(int)
    left = 0

    for i in range(n):
        threshold = timestamps[i] - window_s
        while left < i and timestamps[left] <= threshold:
            old_k = keys[left]
            if old_k is not None and not (isinstance(old_k, float) and math.isnan(old_k)) and old_k != "":
                live[old_k] -= 1
                if live[old_k] == 0:
                    del live[old_k]
            left += 1

        k = keys[i]
        if k is not None and not (isinstance(k, float) and math.isnan(k)) and k != "":
            live[k] += 1
        out[i] = float(len(live))

    return out


def _compute_velocity_features(group_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute 8 velocity features for an account group using two pointers."""
    n = len(group_df)
    timestamps = np.array([D.epoch_seconds(t) for t in group_df["ts"]], dtype=np.float64)
    amounts = group_df["amount_minor"].to_numpy(dtype=np.int64)

    results: dict[str, np.ndarray] = {}

    for w_name, w_s in D.WINDOWS.items():
        txn_count = np.zeros(n, dtype=np.float64)
        amount_sum = np.zeros(n, dtype=np.float64)

        left = 0
        cur_count = 0
        cur_sum = 0

        for i in range(n):
            threshold = timestamps[i] - w_s
            while left < i and timestamps[left] <= threshold:
                cur_count -= 1
                cur_sum -= amounts[left]
                left += 1

            cur_count += 1
            cur_sum += amounts[i]

            txn_count[i] = float(cur_count)
            amount_sum[i] = float(cur_sum)

        results[f"txn_count_{w_name}"] = txn_count
        results[f"amount_sum_minor_{w_name}"] = amount_sum

    return results


def _compute_account_stats(group_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute amount_zscore (via Welford's online algorithm) and seconds_since_prev."""
    n = len(group_df)
    timestamps = np.array([D.epoch_seconds(t) for t in group_df["ts"]], dtype=np.float64)
    amounts = group_df["amount_minor"].to_numpy(dtype=np.float64)

    zscores = np.zeros(n, dtype=np.float64)
    recency = np.zeros(n, dtype=np.float64)

    w_count = 0
    w_mean = 0.0
    w_M2 = 0.0

    last_ts: float | None = None

    for i in range(n):
        x = amounts[i]
        t = timestamps[i]

        # 1. Z-score calculated before adding current transaction amount
        if w_count < 2:
            zscores[i] = 0.0
        else:
            variance = w_M2 / (w_count - 1)
            if variance > 0.0:
                std = math.sqrt(variance)
                zscores[i] = (x - w_mean) / std
            else:
                zscores[i] = 0.0

        # Welford update
        w_count += 1
        delta = x - w_mean
        w_mean += delta / w_count
        delta2 = x - w_mean
        w_M2 += delta * delta2

        # 2. Recency (seconds since previous transaction on this account)
        if last_ts is None:
            recency[i] = -1.0
        else:
            recency[i] = max(0.0, t - last_ts)
        last_ts = t

    return {
        "amount_zscore": zscores,
        "seconds_since_prev": recency,
    }


def _join_graph_snapshots(
    txns: pd.DataFrame,
    snapshots: pd.DataFrame | None,
) -> dict[str, np.ndarray]:
    """Join graph features point-in-time: latest snapshot where computed_at <= transaction.ts.

    Guarantees backward-only lookup. Defaults applied if no eligible snapshot exists.
    """
    n = len(txns)
    out_comp_size = np.full(n, 1.0, dtype=np.float64)
    out_dist_fraud = np.full(n, -1.0, dtype=np.float64)
    out_risk_rank = np.full(n, 0.0, dtype=np.float64)

    if snapshots is None or len(snapshots) == 0:
        return {
            "graph_component_size": out_comp_size,
            "graph_dist_to_fraud": out_dist_fraud,
            "graph_risk_rank": out_risk_rank,
        }

    snaps = snapshots.copy()

    # Normalize account ID column
    acc_col = "account_id" if "account_id" in snaps.columns else "entity_id"
    if acc_col not in snaps.columns:
        raise ValueError("Graph snapshots must contain 'account_id' or 'entity_id'")
    snaps["_acc"] = snaps[acc_col].astype(str)

    # Normalize snapshot timestamp column
    time_col = "snapshot_at" if "snapshot_at" in snaps.columns else "computed_at"
    if time_col not in snaps.columns:
        raise ValueError("Graph snapshots must contain 'snapshot_at' or 'computed_at'")
    snaps["_ts"] = _validate_and_normalize_timestamps(snaps[time_col])

    # Convert timestamps to float epoch seconds for numerical point-in-time comparison
    snaps["_epoch_s"] = [D.epoch_seconds(t) for t in snaps["_ts"]]

    # Deterministic tie-breaker for identical (account, timestamp):
    # In PostgreSQL, uq_graph_snapshots_account_time enforces uniqueness on (account_id, snapshot_at).
    # If unconstrained input DataFrames contain duplicate timestamps, use stable tie-breaker:
    sort_cols = ["_epoch_s"]
    for tie_col in ("snapshot_id", "created_at", "id"):
        if tie_col in snaps.columns:
            sort_cols.append(tie_col)
            break

    # Group snapshots by account
    acc_snaps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for acc, group in snaps.groupby("_acc"):
        sorted_g = group.sort_values(by=sort_cols, ascending=True)
        records = sorted_g.to_dict(orient="records")
        acc_snaps[acc] = records

    # Match each transaction with the latest eligible snapshot (computed_at <= txn.ts)
    txn_timestamps = [D.epoch_seconds(t) for t in txns["ts"]]
    txn_accounts = txns["account_id"].to_numpy()

    for i in range(n):
        acc = txn_accounts[i]
        t = txn_timestamps[i]
        eligible = acc_snaps.get(acc, [])

        # Backward search for latest snapshot with snap._epoch_s <= t
        best_snap: dict[str, Any] | None = None
        for s in eligible:
            if s["_epoch_s"] <= t:
                best_snap = s
            else:
                # Snapshots are sorted ascending by time; once we exceed t, later ones are in the future
                break

        if best_snap is not None:
            # Extract features from columns or JSONB 'features' map
            props = best_snap.get("features", {})
            if isinstance(props, str):
                import json
                try:
                    props = json.loads(props)
                except Exception:
                    props = {}

            c_size = best_snap.get(
                "graph_component_size",
                best_snap.get("component_size", props.get("component_size", 1.0)),
            )
            d_fraud = best_snap.get(
                "graph_dist_to_fraud",
                best_snap.get("dist_to_fraud", props.get("dist_to_fraud", -1.0)),
            )
            r_rank = best_snap.get(
                "graph_risk_rank",
                best_snap.get("risk_rank", best_snap.get("pagerank", props.get("risk_rank", 0.0))),
            )

            out_comp_size[i] = float(c_size) if c_size is not None else 1.0
            out_dist_fraud[i] = float(d_fraud) if d_fraud is not None else -1.0
            out_risk_rank[i] = float(r_rank) if r_rank is not None else 0.0

    return {
        "graph_component_size": out_comp_size,
        "graph_dist_to_fraud": out_dist_fraud,
        "graph_risk_rank": out_risk_rank,
    }


def build_offline_features(
    transactions: pd.DataFrame,
    graph_snapshots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build all 19 canonical features for transactions with point-in-time correctness.

    Returns DataFrame containing original transaction records and all 19 features
    aligned to D.FEATURE_LIST.
    """
    if len(transactions) == 0:
        res = transactions.copy()
        for f_name in D.FEATURE_LIST:
            res[f_name] = pd.Series(dtype=np.float64)
        return res

    # 1. Prepare transactions (validates timestamps, amounts, sorts by (ts, id))
    txns = prepare(transactions)
    n = len(txns)

    # Initialize feature storage dictionary
    feat_data: dict[str, np.ndarray] = {
        name: np.full(n, D.BY_NAME[name].default, dtype=np.float64)
        for name in D.FEATURE_LIST
    }

    # 2. Account-grouped features (Velocity, Distinct Devices/IPs, Stats)
    for _, group_indices in txns.groupby("account_id").groups.items():
        sub_df = txns.loc[group_indices]
        idx_arr = group_indices.to_numpy()

        # A. Velocity (8)
        vel = _compute_velocity_features(sub_df)
        for k, v in vel.items():
            feat_data[k][idx_arr] = v

        # B. Distinct devices & IPs (2)
        sub_ts = np.array([D.epoch_seconds(t) for t in sub_df["ts"]], dtype=np.float64)
        feat_data["distinct_devices_24h"][idx_arr] = _windowed_distinct(
            sub_ts, sub_df["device_id"].tolist(), D.DISTINCT_WINDOW_S
        )
        feat_data["distinct_ips_24h"][idx_arr] = _windowed_distinct(
            sub_ts, sub_df["ip_address"].tolist(), D.DISTINCT_WINDOW_S
        )

        # C. Account statistical features (2)
        stats = _compute_account_stats(sub_df)
        for k, v in stats.items():
            feat_data[k][idx_arr] = v

    # 3. Sharing features: device_accounts_7d (entity: device)
    dev_mask = txns["device_id"].notna()
    if dev_mask.any():
        for _, dev_indices in txns[dev_mask].groupby("device_id").groups.items():
            sub_df = txns.loc[dev_indices]
            idx_arr = dev_indices.to_numpy()
            sub_ts = np.array([D.epoch_seconds(t) for t in sub_df["ts"]], dtype=np.float64)
            feat_data["device_accounts_7d"][idx_arr] = _windowed_distinct(
                sub_ts, sub_df["account_id"].tolist(), D.SHARE_WINDOW_S
            )

    # 4. Sharing features: ip_accounts_7d (entity: ip)
    ip_mask = txns["ip_address"].notna()
    if ip_mask.any():
        for _, ip_indices in txns[ip_mask].groupby("ip_address").groups.items():
            sub_df = txns.loc[ip_indices]
            idx_arr = ip_indices.to_numpy()
            sub_ts = np.array([D.epoch_seconds(t) for t in sub_df["ts"]], dtype=np.float64)
            feat_data["ip_accounts_7d"][idx_arr] = _windowed_distinct(
                sub_ts, sub_df["account_id"].tolist(), D.SHARE_WINDOW_S
            )

    # 5. Calendar features (2): hour_of_day, day_of_week
    for i, ts_val in enumerate(txns["ts"]):
        cal = D.calendar(ts_val)
        feat_data["hour_of_day"][i] = cal["hour_of_day"]
        feat_data["day_of_week"][i] = cal["day_of_week"]

    # 6. Graph features (3)
    graph_feats = _join_graph_snapshots(txns, graph_snapshots)
    for k, v in graph_feats.items():
        feat_data[k] = v

    # 7. Assemble result with canonical column ordering
    result = txns.copy()
    for f_name in D.FEATURE_LIST:
        result[f_name] = feat_data[f_name]

    return result


# Standard aliases for compatibility
compute_offline_features = build_offline_features
build_features = build_offline_features
