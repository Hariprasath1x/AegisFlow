"""Offline point-in-time feature computation for AegisFlow M2.

Computes all 19 canonical features on historical transaction DataFrames
using event-time semantics, two-pointer sliding windows, Welford's online
algorithm, and backward-only graph snapshot joins.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import Engine, create_engine, inspect, text

from aegis_ml.features import definitions as D

# 64-bit signed integer bounds
_INT64_MIN = -9_223_372_036_854_775_808
_INT64_MAX = 9_223_372_036_854_775_807

# Canonical column layout for exported PIT feature Parquet files
PIT_TRACE_COLUMNS: tuple[str, ...] = (
    "txn_id",
    "ts",
    "account_id",
    "destination_account_id",
    "merchant_id",
    "device_id",
    "ip_id",
    "feature_snapshot_at",
    "computed_as_of",
    "feature_set_version",
)
FORBIDDEN_PIT_TARGET_COLUMNS: frozenset[str] = frozenset(
    {
        "target",
        "is_fraud",
        "fraud_type",
        "labeled_at",
        "target_typology",
        "primary_outcome",
        "reason",
    }
)
_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_PIT_ARROW_SCHEMA = pa.schema(
    [
        pa.field("txn_id", pa.string(), nullable=False),
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("account_id", pa.string(), nullable=False),
        pa.field("destination_account_id", pa.string(), nullable=True),
        pa.field("merchant_id", pa.string(), nullable=True),
        pa.field("device_id", pa.string(), nullable=True),
        pa.field("ip_id", pa.string(), nullable=True),
        pa.field("feature_snapshot_at", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("computed_as_of", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("feature_set_version", pa.string(), nullable=False),
        *[pa.field(name, pa.float64(), nullable=False) for name in D.FEATURE_LIST],
    ]
)


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


def _normalize_optional_entity(val: Any) -> str | None:
    """Normalize optional entity identifier (device, IP, IP UUID, merchant) to str or None."""
    if val is None:
        return None
    if isinstance(val, (float, np.floating)):
        if math.isnan(float(val)):
            return None
        return str(val)
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    # Handle psycopg3 INET IPv4Interface/IPv6Interface values cleanly
    if hasattr(val, "ip"):
        s = str(val.ip).strip()
    else:
        s = str(val).strip()
    return s if s else None


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare raw transaction DataFrame for feature building.

    Validates and normalizes timestamps and amounts.
    Coalesces IP identity cleanly from `ip_address`, `ip_id`, or `ip`.
    Orders transactions deterministically by (ts, id).
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

    # 5. Normalize device_id and coalesce ip_address / ip_id / ip
    if "device_id" in f.columns:
        f["device_id"] = [_normalize_optional_entity(v) for v in f["device_id"].tolist()]
    else:
        f["device_id"] = None

    has_ip_addr = "ip_address" in f.columns
    has_ip_id = "ip_id" in f.columns
    has_ip = "ip" in f.columns
    if has_ip_addr or has_ip_id or has_ip:
        ip_addrs = f["ip_address"].tolist() if has_ip_addr else [None] * len(f)
        ip_ids = f["ip_id"].tolist() if has_ip_id else [None] * len(f)
        ips = f["ip"].tolist() if has_ip else [None] * len(f)
        resolved_ips: list[str | None] = []
        for raw_addr, raw_id, raw_ip in zip(ip_addrs, ip_ids, ips):
            val = _normalize_optional_entity(raw_addr)
            if val is None:
                val = _normalize_optional_entity(raw_id)
            if val is None:
                val = _normalize_optional_entity(raw_ip)
            resolved_ips.append(val)
        f["ip_address"] = resolved_ips
    else:
        f["ip_address"] = None

    # 6. Strict event-time sort by (ts, id)
    f = f.sort_values(by=["ts", "id"], kind="mergesort").reset_index(drop=True)
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


def _join_graph_snapshots_with_timestamps(
    txns: pd.DataFrame,
    snapshots: pd.DataFrame | None,
) -> tuple[dict[str, np.ndarray], pd.Series]:
    """Join graph features point-in-time and capture matched snapshot timestamps.

    Guarantees backward-only lookup (`snapshot_at <= transaction.ts`).
    Returns `(feature_arrays, matched_snapshot_at_series)`.
    """
    n = len(txns)
    out_comp_size = np.full(n, 1.0, dtype=np.float64)
    out_dist_fraud = np.full(n, -1.0, dtype=np.float64)
    out_risk_rank = np.full(n, 0.0, dtype=np.float64)
    matched_times: list[pd.Timestamp | Any] = [pd.NaT] * n

    if snapshots is None or len(snapshots) == 0:
        return (
            {
                "graph_component_size": out_comp_size,
                "graph_dist_to_fraud": out_dist_fraud,
                "graph_risk_rank": out_risk_rank,
            },
            pd.Series(matched_times, index=txns.index, dtype="datetime64[ns, UTC]"),
        )

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

    # Deterministic tie-breaker for identical (account, timestamp)
    sort_cols = ["_epoch_s"]
    for tie_col in ("snapshot_id", "created_at", "id"):
        if tie_col in snaps.columns:
            sort_cols.append(tie_col)
            break

    acc_snaps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for acc, group in snaps.groupby("_acc"):
        sorted_g = group.sort_values(by=sort_cols, ascending=True)
        acc_snaps[acc] = sorted_g.to_dict(orient="records")

    txn_timestamps = [D.epoch_seconds(t) for t in txns["ts"]]
    txn_accounts = txns["account_id"].to_numpy()

    for i in range(n):
        acc = txn_accounts[i]
        t = txn_timestamps[i]
        eligible = acc_snaps.get(acc, [])

        best_snap: dict[str, Any] | None = None
        for s in eligible:
            if s["_epoch_s"] <= t:
                best_snap = s
            else:
                break

        if best_snap is not None:
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
            matched_times[i] = best_snap["_ts"]

    return (
        {
            "graph_component_size": out_comp_size,
            "graph_dist_to_fraud": out_dist_fraud,
            "graph_risk_rank": out_risk_rank,
        },
        pd.Series(pd.to_datetime(matched_times, utc=True), index=txns.index),
    )


def _join_graph_snapshots(
    txns: pd.DataFrame,
    snapshots: pd.DataFrame | None,
) -> dict[str, np.ndarray]:
    """Join graph features point-in-time: latest snapshot where computed_at <= transaction.ts."""
    feats, _ = _join_graph_snapshots_with_timestamps(txns, snapshots)
    return feats


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
build = build_offline_features


# ============================================================================
# PostgreSQL Extraction & Point-in-Time Parquet CLI (Step 2.10)
# ============================================================================


def _load_dotenv() -> None:
    """Populate unset environment variables from repository-root .env if present."""
    for parent in Path(__file__).resolve().parents:
        env_path = parent / ".env"
        if env_path.is_file():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.removeprefix("export ").strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                    value = value[1:-1]
                if key:
                    os.environ.setdefault(key, value)
            return


def _normalize_sqlalchemy_url(url: str) -> str:
    """Ensure PostgreSQL URL uses psycopg driver when unspecified."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def resolve_database_url(db_url: str | None = None) -> str:
    """Resolve SQLAlchemy database URL from argument, environment, or .env."""
    _load_dotenv()
    if db_url:
        return _normalize_sqlalchemy_url(db_url)

    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return _normalize_sqlalchemy_url(env_url)

    from urllib.parse import quote_plus

    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5433")
    dbname = os.getenv("DB_NAME", "aegisflow")
    if user and password:
        return (
            f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(password)}"
            f"@{host}:{port}/{dbname}"
        )

    raise ValueError(
        "Database connection URL is not configured. Provide --db-url, set DATABASE_URL, "
        "or configure DB_USER/DB_PASS/DB_HOST/DB_PORT/DB_NAME."
    )


def _parse_utc_arg(ts: datetime | str) -> datetime:
    """Parse or coerce CLI/API timestamp argument to UTC, rejecting naive timestamps."""
    if isinstance(ts, str):
        raw = ts.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
    elif isinstance(ts, (datetime, pd.Timestamp)):
        dt = ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts
    else:
        raise TypeError(f"Invalid timestamp type: {type(ts)!r}")

    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"Timezone-naive timestamp is rejected: {ts!r}")
    return dt.astimezone(timezone.utc)


@contextmanager
def _managed_engine(engine: Engine | str | None) -> Iterator[Engine]:
    """Yield SQLAlchemy Engine, disposing only engines constructed by this context."""
    if engine is None:
        eng = create_engine(resolve_database_url(), pool_pre_ping=True)
        try:
            yield eng
        finally:
            eng.dispose()
    elif isinstance(engine, str):
        eng = create_engine(_normalize_sqlalchemy_url(engine), pool_pre_ping=True)
        try:
            yield eng
        finally:
            eng.dispose()
    else:
        yield engine


def load_pit_inputs_from_db(
    engine: Engine | str,
    *,
    as_of: datetime | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load transactions and graph snapshots from PostgreSQL up to optional `as_of`."""
    norm_as_of = _parse_utc_arg(as_of) if as_of is not None else None

    with _managed_engine(engine) as eng:
        with eng.connect() as conn:
            inspector = inspect(conn)
            tables = set(inspector.get_table_names())
            if "transactions" not in tables:
                raise RuntimeError("Table 'transactions' does not exist in target database.")

            txn_cols = {c["name"] for c in inspector.get_columns("transactions")}
            pk_col = "txn_id" if "txn_id" in txn_cols else "id"
            ts_col = "occurred_at" if "occurred_at" in txn_cols else "ts"
            is_pg = conn.dialect.name == "postgresql"

            dest_expr = (
                "CAST(destination_account_id AS text)"
                if "destination_account_id" in txn_cols
                else "NULL"
            )
            merch_expr = "CAST(merchant_id AS text)" if "merchant_id" in txn_cols else "NULL"
            dev_expr = "CAST(device_id AS text)" if "device_id" in txn_cols else "NULL"
            ip_id_expr = "CAST(ip_id AS text)" if "ip_id" in txn_cols else "NULL"
            if "ip_address" in txn_cols:
                ip_addr_expr = "host(ip_address)" if is_pg else "CAST(ip_address AS text)"
            else:
                ip_addr_expr = "NULL"

            where_txn = f"WHERE {ts_col} <= :as_of" if norm_as_of is not None else ""
            txn_sql = text(
                f"""
                SELECT
                    CAST({pk_col} AS text) AS txn_id,
                    {ts_col} AS ts,
                    CAST(account_id AS text) AS account_id,
                    {dest_expr} AS destination_account_id,
                    {merch_expr} AS merchant_id,
                    {dev_expr} AS device_id,
                    {ip_id_expr} AS ip_id,
                    {ip_addr_expr} AS ip_address,
                    amount
                FROM transactions
                {where_txn}
                ORDER BY {ts_col} ASC, CAST({pk_col} AS text) ASC;
                """
            )
            params: dict[str, Any] = {"as_of": norm_as_of} if norm_as_of is not None else {}
            txn_rows = conn.execute(txn_sql, params).mappings().fetchall()
            txns_df = pd.DataFrame([dict(r) for r in txn_rows])

            snaps_df = pd.DataFrame()
            if "graph_feature_snapshots" in tables:
                snap_cols = {c["name"] for c in inspector.get_columns("graph_feature_snapshots")}
                acc_col = "account_id" if "account_id" in snap_cols else "entity_id"
                snap_ts_col = "snapshot_at" if "snapshot_at" in snap_cols else "computed_at"
                select_parts = [f"CAST({acc_col} AS text) AS account_id", f"{snap_ts_col} AS snapshot_at"]
                for col_name in (
                    "snapshot_id",
                    "created_at",
                    "id",
                    "component_size",
                    "dist_to_fraud",
                    "risk_rank",
                    "pagerank",
                    "features",
                ):
                    if col_name in snap_cols:
                        if col_name == "snapshot_id":
                            select_parts.append("CAST(snapshot_id AS text) AS snapshot_id")
                        else:
                            select_parts.append(col_name)

                where_snap = f"WHERE {snap_ts_col} <= :as_of" if norm_as_of is not None else ""
                snap_sql = text(
                    f"""
                    SELECT {', '.join(select_parts)}
                    FROM graph_feature_snapshots
                    {where_snap}
                    ORDER BY {snap_ts_col} ASC, CAST({acc_col} AS text) ASC;
                    """
                )
                snap_rows = conn.execute(snap_sql, params).mappings().fetchall()
                snaps_df = pd.DataFrame([dict(r) for r in snap_rows])

    return txns_df, snaps_df


def build_pit_feature_table(
    transactions: pd.DataFrame,
    graph_snapshots: pd.DataFrame | None = None,
    *,
    as_of: datetime | str | None = None,
) -> pd.DataFrame:
    """Build deterministic, validated point-in-time feature table for Parquet export.

    Excludes any transactions/snapshots strictly after `as_of`, preserves the
    19 canonical features without supervised target columns, and records exact
    per-row transaction and graph-snapshot timestamps for traceability.
    """
    norm_as_of = _parse_utc_arg(as_of) if as_of is not None else None

    if len(transactions) == 0:
        empty_as_of = norm_as_of if norm_as_of is not None else _EPOCH_UTC
        empty_data: dict[str, Any] = {
            "txn_id": pd.Series([], dtype="object"),
            "ts": pd.Series([], dtype="datetime64[ns, UTC]"),
            "account_id": pd.Series([], dtype="object"),
            "destination_account_id": pd.Series([], dtype="object"),
            "merchant_id": pd.Series([], dtype="object"),
            "device_id": pd.Series([], dtype="object"),
            "ip_id": pd.Series([], dtype="object"),
            "feature_snapshot_at": pd.Series([], dtype="datetime64[ns, UTC]"),
            "computed_as_of": pd.Series([], dtype="datetime64[ns, UTC]"),
            "feature_set_version": pd.Series([], dtype="object"),
            **{name: pd.Series([], dtype="float64") for name in D.FEATURE_LIST},
        }
        out_empty = pd.DataFrame(empty_data)
        out_empty.attrs["feature_set_version"] = D.FEATURE_SET_VERSION
        out_empty.attrs["computed_as_of"] = empty_as_of.isoformat()
        return out_empty

    tx_in = transactions.copy()
    if "account_id" not in tx_in.columns or tx_in["account_id"].isna().any():
        raise ValueError("Input transactions contain missing/null account_id")
    if (tx_in["account_id"].astype(str).str.strip() == "").any():
        raise ValueError("Input transactions contain empty account_id")
    for id_col in ("txn_id", "id"):
        if id_col in tx_in.columns:
            if tx_in[id_col].isna().any() or (tx_in[id_col].astype(str).str.strip() == "").any():
                raise ValueError(f"Input transactions contain missing or empty {id_col}")

    prep = prepare(tx_in)
    prep["txn_id"] = prep["id"].astype(str)

    if norm_as_of is not None:
        cutoff_ts = pd.Timestamp(norm_as_of)
        prep = prep[prep["ts"] <= cutoff_ts].reset_index(drop=True)
        if len(prep) == 0:
            return build_pit_feature_table(prep.iloc[0:0], None, as_of=norm_as_of)

    snaps_filtered: pd.DataFrame | None = None
    if graph_snapshots is not None and len(graph_snapshots) > 0:
        snaps_filtered = graph_snapshots.copy()
        time_col = "snapshot_at" if "snapshot_at" in snaps_filtered.columns else "computed_at"
        if time_col not in snaps_filtered.columns:
            raise ValueError("Graph snapshots must contain 'snapshot_at' or 'computed_at'")
        snaps_filtered[time_col] = _validate_and_normalize_timestamps(snaps_filtered[time_col])
        if norm_as_of is not None:
            snaps_filtered = snaps_filtered[
                snaps_filtered[time_col] <= pd.Timestamp(norm_as_of)
            ].reset_index(drop=True)

    # Recompute canonical features from already-validated/sorted frame so index aligns 1:1
    feats_df = build_offline_features(prep, snaps_filtered)
    _, snap_times = _join_graph_snapshots_with_timestamps(prep, snaps_filtered)

    resolved_as_of = (
        norm_as_of
        if norm_as_of is not None
        else prep["ts"].max().to_pydatetime()
    )

    def _opt_col(col_name: str) -> list[str | None]:
        if col_name not in prep.columns:
            return [None] * len(prep)
        return [_normalize_optional_entity(v) for v in prep[col_name].tolist()]

    # Capture effective IP identifier in `ip_id` trace column when `ip_id` was absent/null
    raw_ip_id = _opt_col("ip_id")
    effective_ip = _opt_col("ip_address")
    trace_ip = [
        iid if iid is not None else eff
        for iid, eff in zip(raw_ip_id, effective_ip)
    ]

    result_data: dict[str, Any] = {
        "txn_id": prep["txn_id"].astype(str),
        "ts": prep["ts"],
        "account_id": prep["account_id"].astype(str),
        "destination_account_id": _opt_col("destination_account_id"),
        "merchant_id": _opt_col("merchant_id"),
        "device_id": _opt_col("device_id"),
        "ip_id": trace_ip,
        "feature_snapshot_at": snap_times,
        "computed_as_of": pd.Series(
            pd.Timestamp(resolved_as_of), index=prep.index, dtype="datetime64[ns, UTC]"
        ),
        "feature_set_version": D.FEATURE_SET_VERSION,
    }
    for name in D.FEATURE_LIST:
        result_data[name] = feats_df[name].astype(np.float64)

    out_df = pd.DataFrame(result_data)
    out_df = out_df.sort_values(by=["ts", "txn_id"], kind="mergesort").reset_index(drop=True)
    validate_pit_output(out_df)
    out_df.attrs["feature_set_version"] = D.FEATURE_SET_VERSION
    out_df.attrs["computed_as_of"] = resolved_as_of.isoformat()
    return out_df


def validate_pit_output(df: pd.DataFrame) -> None:
    """Validate PIT feature matrix contract before writing Parquet output."""
    forbidden_present = FORBIDDEN_PIT_TARGET_COLUMNS & set(df.columns)
    if forbidden_present:
        raise ValueError(
            f"Canonical PIT features output must not contain supervised target columns: {sorted(forbidden_present)}"
        )

    expected_cols = list(PIT_TRACE_COLUMNS) + list(D.FEATURE_LIST)
    if list(df.columns) != expected_cols:
        raise ValueError(
            f"PIT output columns mismatch: expected {expected_cols}, got {list(df.columns)}"
        )

    if len(df) == 0:
        return

    if df["txn_id"].isna().any() or (df["txn_id"].astype(str).str.strip() == "").any():
        raise ValueError("PIT output contains null or empty txn_id")
    if not df["txn_id"].is_unique:
        raise ValueError("PIT output contains duplicate txn_id")
    if df["account_id"].isna().any() or (df["account_id"].astype(str).str.strip() == "").any():
        raise ValueError("PIT output contains null or empty account_id")

    for tcol in ("ts", "computed_as_of"):
        if not pd.api.types.is_datetime64_any_dtype(df[tcol]) or df[tcol].dt.tz is None:
            raise ValueError(f"Column {tcol!r} must be timezone-aware UTC datetime")
        if df[tcol].isna().any():
            raise ValueError(f"Column {tcol!r} cannot contain nulls")

    if (df["ts"] > df["computed_as_of"]).any():
        raise ValueError("Temporal leakage detected: row timestamp exceeds computed_as_of")

    snap_col = df["feature_snapshot_at"]
    if not pd.api.types.is_datetime64_any_dtype(snap_col) or snap_col.dt.tz is None:
        raise ValueError("Column 'feature_snapshot_at' must be timezone-aware UTC datetime")
    valid_snaps = snap_col.notna()
    if valid_snaps.any() and (snap_col[valid_snaps] > df.loc[valid_snaps, "ts"]).any():
        raise ValueError(
            "Temporal leakage detected: feature_snapshot_at is strictly after transaction ts"
        )

    for fname in D.FEATURE_LIST:
        col_vals = df[fname].to_numpy(dtype=np.float64)
        if not np.isfinite(col_vals).all():
            raise ValueError(f"Canonical feature {fname!r} contains NaN or Inf values")


def write_pit_parquet(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Atomically write validated PIT feature DataFrame to Parquet with canonical schema."""
    validate_pit_output(df)
    target = Path(out_path)
    out_dir = target.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, schema=_PIT_ARROW_SCHEMA, preserve_index=False)

    temp_fd, temp_path = tempfile.mkstemp(dir=str(out_dir), prefix=".features_pit_tmp_")
    replaced = False
    try:
        current_umask = os.umask(0)
        os.umask(current_umask)
        default_mode = 0o666 & ~current_umask
        target_mode = (
            target.stat().st_mode & 0o7777 if target.exists() else default_mode
        )
        os.fchmod(temp_fd, target_mode)
        os.close(temp_fd)
        temp_fd = -1

        pq.write_table(table, temp_path, compression="snappy")
        os.replace(temp_path, target)
        replaced = True
    finally:
        if temp_fd >= 0:
            try:
                os.close(temp_fd)
            except OSError:
                pass
        if not replaced and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    return target


def export_features_pit(
    out_path: str | Path,
    *,
    engine: Engine | str | None = None,
    as_of: datetime | str | None = None,
) -> dict[str, Any]:
    """Extract transactions + graph snapshots from DB, build PIT features, and write Parquet."""
    with _managed_engine(engine) as eng:
        txns_df, snaps_df = load_pit_inputs_from_db(eng, as_of=as_of)

    pit_df = build_pit_feature_table(txns_df, snaps_df, as_of=as_of)
    written_path = write_pit_parquet(pit_df, out_path)
    return {
        "rows": int(len(pit_df)),
        "num_features": len(D.FEATURE_LIST),
        "feature_set_version": D.FEATURE_SET_VERSION,
        "computed_as_of": pit_df.attrs.get("computed_as_of"),
        "out_path": str(written_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for PostgreSQL-backed PIT feature extraction."""
    parser = argparse.ArgumentParser(
        description="Build canonical 19-feature point-in-time Parquet matrix from PostgreSQL."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output Parquet path (e.g. data/features/features_pit.parquet).",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Optional PostgreSQL SQLAlchemy/libpq connection URL.",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Optional ISO-8601 UTC cutoff timestamp excluding future rows.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for offline point-in-time feature extraction."""
    args = parse_args(argv)
    try:
        summary = export_features_pit(
            out_path=args.out,
            engine=args.db_url,
            as_of=args.as_of,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: PIT feature export failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"{summary['rows']} rows x {summary['num_features']} features "
        f"(feature set {summary['feature_set_version']}) -> {summary['out_path']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
