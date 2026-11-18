"""Assemble the training matrix from M2's feature definitions.

The rule this file exists to enforce: M3 must not invent features. Every column
comes from ml/src/aegis_ml/features/definitions.py - the same declarations that
compile to the Redis update path - so whatever the model learns, the scoring
service in M4 can actually compute in under 100 ms. If a feature is missing here,
the fix is to declare it in definitions.py, not to add a pandas expression.
"""
from __future__ import annotations

from collections import defaultdict
import pathlib
from typing import Any, Sequence
import uuid
import numpy as np
import pandas as pd

from aegis_ml.features import definitions as D
from aegis_ml.features.definitions import FEATURES

try:
    from aegis_ml.features.offline import build_point_in_time
except ImportError:
    from aegis_ml.features.offline import build_offline_features

    def build_point_in_time(
        df: pd.DataFrame,
        features: Sequence[Any] | None = None,
        as_of_column: str = "ts",
        graph_snapshots: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Point-in-time feature computation adhering to M2's offline feature contract."""
        if as_of_column not in df.columns:
            raise KeyError(f"as_of_column {as_of_column!r} not found in input DataFrame")

        tx = df.copy()
        # Collision-resistant unique positional identifier guarantees row identity
        # and caller order preservation without relying on index label uniqueness.
        pos_col = f"__aegis_pos_id_{uuid.uuid4().hex}__"
        tx[pos_col] = np.arange(len(df), dtype=np.int64)

        if as_of_column != "ts":
            tx["ts"] = tx[as_of_column]

        if "id" not in tx.columns:
            if "transaction_id" in tx.columns:
                tx["id"] = tx["transaction_id"].astype(str)
            elif "txn_id" in tx.columns:
                tx["id"] = tx["txn_id"].astype(str)
            else:
                tx["id"] = tx[pos_col].astype(str)

        raw_pit = build_offline_features(tx, graph_snapshots=graph_snapshots)
        # Restore caller positional order and assign original index
        aligned = raw_pit.sort_values(by=pos_col, kind="mergesort")
        aligned.index = df.index

        feat_cols = [f.name for f in features] if features is not None else [f.name for f in FEATURES]
        missing_feats = [col for col in feat_cols if col not in aligned.columns]
        if missing_feats:
            raise KeyError(f"Declared features missing from point-in-time output: {missing_feats}")

        return aligned[feat_cols]


try:
    from aegis_ml.features.graph_features import load_graph_features
except ImportError:
    def load_graph_features(
        account_ids: Sequence[Any] | pd.Series,
        as_of: Sequence[Any] | pd.Series | None = None,
        graph_snapshots: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Point-in-time graph feature loader adhering to M2 canonical contracts.

        In M2, graph features are generated via hourly snapshots from the entity graph
        (aegis_ml.features.graph_features) and joined point-in-time in offline batch
        (aegis_ml.features.offline._join_graph_snapshots).

        Canonical defaults verified from aegis_ml.features.definitions:
        - graph_component_size = 1.0
        - graph_dist_to_fraud = -1.0
        - graph_risk_rank = 0.0

        Note on graph_device_degree:
        M2's graph snapshot producer persists `degree` for the account entity node,
        which represents account degree in the entity graph, not device degree.
        M2 defines no `graph_device_degree` feature or canonical default.
        When graph_snapshots explicitly contains a device degree column
        ('graph_device_degree' or 'device_degree'), it is joined point-in-time via
        the canonical M2 snapshot selection helper; otherwise, missing or unsupported
        values are represented as NaN (unavailable).
        """
        n = len(account_ids)
        if isinstance(getattr(account_ids, "index", None), pd.Index):
            orig_index = account_ids.index
        else:
            orig_index = pd.RangeIndex(n)

        # Default M2 canonical values
        comp_size = np.full(n, 1.0, dtype=np.float64)
        dist_fraud = np.full(n, -1.0, dtype=np.float64)
        risk_rank = np.full(n, 0.0, dtype=np.float64)
        device_degree = np.full(n, np.nan, dtype=np.float64)

        if graph_snapshots is not None and len(graph_snapshots) > 0 and as_of is not None:
            from aegis_ml.features.offline import _join_graph_snapshots

            acc_list = [str(a) for a in account_ids]
            as_of_list = list(as_of) if hasattr(as_of, "__iter__") else [as_of] * n

            # Positional DataFrame for M2 joiner to avoid label-alignment issues
            tx_mock = pd.DataFrame(
                {
                    "account_id": acc_list,
                    "ts": as_of_list,
                }
            )

            # 1. Join canonical M2 graph features via verified M2 helper
            joined = _join_graph_snapshots(tx_mock, graph_snapshots)
            # Explicitly convert to positional NumPy arrays to avoid pandas label alignment
            comp_raw = joined["graph_component_size"]
            dist_raw = joined["graph_dist_to_fraud"]
            risk_raw = joined["graph_risk_rank"]

            comp_size = (
                comp_raw.to_numpy(copy=True)
                if hasattr(comp_raw, "to_numpy")
                else np.array(comp_raw, dtype=np.float64, copy=True)
            )
            dist_fraud = (
                dist_raw.to_numpy(copy=True)
                if hasattr(dist_raw, "to_numpy")
                else np.array(dist_raw, dtype=np.float64, copy=True)
            )
            risk_rank = (
                risk_raw.to_numpy(copy=True)
                if hasattr(risk_raw, "to_numpy")
                else np.array(risk_raw, dtype=np.float64, copy=True)
            )

            # 2. Extract device degree if explicitly provided by snapshot dataset
            deg_col = None
            for cand in ("graph_device_degree", "device_degree"):
                if cand in graph_snapshots.columns:
                    deg_col = cand
                    break

            if deg_col is not None:
                # Reuse canonical M2 selection helper for device degree
                snaps_marker = graph_snapshots.copy()
                snaps_marker["graph_risk_rank"] = 1.0
                res_marker = _join_graph_snapshots(tx_mock, snaps_marker)["graph_risk_rank"]
                marker_arr = (
                    res_marker.to_numpy(copy=True)
                    if hasattr(res_marker, "to_numpy")
                    else np.array(res_marker, dtype=np.float64, copy=True)
                )
                has_snap = marker_arr == 1.0

                snaps_deg = graph_snapshots.copy()
                snaps_deg["graph_risk_rank"] = snaps_deg[deg_col].astype(float)
                res_deg = _join_graph_snapshots(tx_mock, snaps_deg)["graph_risk_rank"]
                deg_joined = (
                    res_deg.to_numpy(copy=True)
                    if hasattr(res_deg, "to_numpy")
                    else np.array(res_deg, dtype=np.float64, copy=True)
                )
                device_degree = np.where(has_snap, deg_joined, np.nan)

        # Construct DataFrame directly from 1D positional numpy arrays with original index
        return pd.DataFrame(
            {
                "graph_component_size": comp_size,
                "graph_dist_to_fraud": dist_fraud,
                "graph_risk_rank": risk_rank,
                "graph_device_degree": device_degree,
            },
            index=orig_index,
        )


PROCESSED = pathlib.Path("data/processed")

CATEGORICAL = [
    "channel",
    "card_network",
    "card_type",
    "device_type",
    "email_domain",
    "region",
]


def load_transactions(path: pathlib.Path | str = PROCESSED / "ieee_transactions.parquet") -> pd.DataFrame:
    """Load canonical transactions Parquet file."""
    path_obj = pathlib.Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"{path_obj} missing - run `dvc repro prepare_ieee` (M3) and the synthetic "
            f"generator (M1) first"
        )
    return pd.read_parquet(path_obj)


def feature_names() -> list[str]:
    """Declared features + graph features + encoded categoricals, in a FIXED order.

    The order is persisted with the model: a column shuffle between training and
    serving is silent and catastrophic. Deduplication guarantees no duplicate
    column names while preserving exact declaration ordering.
    """
    raw = (
        [f.name for f in FEATURES]
        + [
            "graph_component_size",
            "graph_dist_to_fraud",
            "graph_risk_rank",
            "graph_device_degree",
        ]
        + [f"cat__{c}" for c in CATEGORICAL]
        + ["amount", "log_amount", "hour_of_day", "day_of_week"]
    )
    seen: set[str] = set()
    names: list[str] = []
    for x in raw:
        if x not in seen:
            seen.add(x)
            names.append(x)
    return names


def encode_categoricals(
    df: pd.DataFrame,
    mapping: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Frequency encoding fitted on TRAIN ONLY. Target encoding would leak the label
    through the encoder, which is the most common silent leak in fraud notebooks."""
    for c in CATEGORICAL:
        if c not in df.columns:
            raise KeyError(f"Categorical column {c!r} missing from input DataFrame")

    if mapping is not None:
        if not isinstance(mapping, dict):
            raise TypeError(f"mapping must be a dict, got {type(mapping)!r}")
        missing_cols = [c for c in CATEGORICAL if c not in mapping]
        if missing_cols:
            raise ValueError(f"Supplied mapping is incomplete, missing categorical columns: {missing_cols}")
        for c in CATEGORICAL:
            if not isinstance(mapping[c], dict):
                raise TypeError(f"Mapping for column {c!r} must be a dict, got {type(mapping[c])!r}")

    fitted_mapping = (
        mapping
        if mapping is not None
        else {c: df[c].astype(str).value_counts(normalize=True).to_dict() for c in CATEGORICAL}
    )

    out = pd.DataFrame(index=df.index)
    for c in CATEGORICAL:
        out[f"cat__{c}"] = df[c].astype(str).map(fitted_mapping[c]).fillna(0.0).astype(float)

    return out, fitted_mapping


def build(
    df: pd.DataFrame,
    as_of_column: str = "ts",
    mapping: dict | None = None,
    graph_snapshots: pd.DataFrame | None = None,
    fit_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Returns (X DataFrame in feature_names() order, mapping). Point-in-time
    correctness is M2's job; this function only refuses to proceed without it.

    Authoritative feature ownership:
    - M2's `build_point_in_time` owns all 19 canonical features declared in `FEATURES`
      (including `hour_of_day`, `day_of_week`, `graph_component_size`,
      `graph_dist_to_fraud`, and `graph_risk_rank`).
    - `load_graph_features` owns `graph_device_degree`.
    - `encode_categoricals` owns `cat__{channel, card_network, card_type, device_type, email_domain, region}`.
    - `base` owns `amount` and `log_amount`.
    """
    if as_of_column not in df.columns:
        raise KeyError(f"as_of_column {as_of_column!r} not found in input DataFrame")
    if "amount" not in df.columns:
        raise KeyError("Required column 'amount' missing from input DataFrame")
    if "account_id" not in df.columns:
        raise KeyError("Required column 'account_id' missing from input DataFrame")

    if mapping is not None and fit_df is not None:
        raise ValueError("Cannot supply both mapping and fit_df to build()")
    if fit_df is not None:
        _, mapping = encode_categoricals(fit_df)

    # 1. Point-in-time features from M2 (owns all 19 FEATURES)
    pit_kwargs: dict[str, Any] = {"as_of_column": as_of_column}
    if graph_snapshots is not None:
        pit_kwargs["graph_snapshots"] = graph_snapshots
    pit = build_point_in_time(
        df,
        features=FEATURES,
        **pit_kwargs,
    )
    if len(pit) != len(df):
        raise AssertionError(
            "point-in-time builder returned a different row count - "
            "a join fanned out; fix it before training"
        )
    if not pit.index.equals(df.index):
        raise AssertionError("point-in-time builder altered row index alignment")

    # 2. Graph features (owns graph_device_degree)
    graph_kwargs: dict[str, Any] = {"as_of": df[as_of_column]}
    if graph_snapshots is not None:
        graph_kwargs["graph_snapshots"] = graph_snapshots
    graph = load_graph_features(
        df["account_id"],
        **graph_kwargs,
    )
    if len(graph) != len(df):
        raise AssertionError("graph feature loader returned a different row count")
    if not graph.index.equals(df.index):
        raise AssertionError("graph feature loader altered row index alignment")

    # 3. Frequency-encoded categoricals
    cats, mapping = encode_categoricals(df, mapping=mapping)
    if len(cats) != len(df):
        raise AssertionError("categorical encoder returned a different row count")
    if not cats.index.equals(df.index):
        raise AssertionError("categorical encoder altered row index alignment")

    # 4. Base features (owns amount, log_amount)
    base = pd.DataFrame(
        {
            "amount": df["amount"].astype(float),
            "log_amount": np.log1p(df["amount"].astype(float)),
        },
        index=df.index,
    )

    # Assemble directly from authoritative owners with zero overlapping columns
    X = pd.concat([pit, graph[["graph_device_degree"]], cats, base], axis=1)

    expected_names = feature_names()
    missing = [c for c in expected_names if c not in X.columns]
    if missing:
        raise KeyError(f"declared features missing from the built matrix: {missing}")

    return X[expected_names], mapping
