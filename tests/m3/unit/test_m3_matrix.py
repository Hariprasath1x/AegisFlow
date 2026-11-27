"""Unit tests for point-in-time training matrix construction (Milestone 3 - Step 3.4).

Verifies:
1. Parquet loading and missing file error handling.
2. Authoritative feature contract, fixed ordering, uniqueness, and provenance of overlapping columns.
3. Frequency encoding semantics, mapping reuse, unseen category fallback (0.0), and index preservation.
4. Matrix assembly with M2 point-in-time features, graph features, categoricals, and base features.
5. PIT adapter index safety: non-monotonic index, duplicate index labels, collision-resistant columns,
   row-order preservation, and non-mutation of input DataFrame.
6. Strict temporal anti-leakage across all feature families (velocity, amount sums, distinct entities,
   account statistics, calendar, and graph snapshots).
7. Label independence: neither is_fraud nor label_reported_at can influence any feature calculation.
"""
from __future__ import annotations

import pathlib
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from aegis_ml.features.definitions import FEATURES
from aegis_ml.training.matrix import (
    CATEGORICAL,
    build,
    build_point_in_time,
    encode_categoricals,
    feature_names,
    load_graph_features,
    load_transactions,
)


def sample_transactions() -> pd.DataFrame:
    """Generate a minimal valid canonical transaction DataFrame with custom index."""
    return pd.DataFrame(
        {
            "transaction_id": ["TX101", "TX102", "TX103", "TX104"],
            "account_id": ["ACC_A", "ACC_A", "ACC_B", "ACC_A"],
            "ts": pd.to_datetime(
                [
                    "2026-03-01 10:00:00+00:00",
                    "2026-03-01 10:30:00+00:00",
                    "2026-03-01 11:00:00+00:00",
                    "2026-03-01 12:00:00+00:00",
                ],
                utc=True,
            ),
            "amount": [100.0, 50.0, 200.0, 75.0],
            "channel": ["web", "app", "web", "pos"],
            "card_network": ["visa", "mc", "visa", "amex"],
            "card_type": ["debit", "credit", "debit", "credit"],
            "device_type": ["mobile", "desktop", "mobile", "tablet"],
            "email_domain": ["gmail.com", "yahoo.com", "gmail.com", "hotmail.com"],
            "region": ["US", "US", "CA", "GB"],
            "device_id": ["DEV_1", "DEV_2", "DEV_1", "DEV_3"],
            "ip_address": ["10.0.0.1", "10.0.0.2", "10.0.0.1", "10.0.0.3"],
            "is_fraud": [0, 1, 0, 0],
            "label_reported_at": [
                pd.NaT,
                pd.Timestamp("2026-03-05 10:30:00+00:00"),
                pd.NaT,
                pd.NaT,
            ],
        },
        index=[10, 20, 30, 40],
    )


# ============================================================================
# 1. Loading Tests
# ============================================================================


def test_load_transactions_valid(tmp_path: pathlib.Path) -> None:
    """1. Valid Parquet file is loaded cleanly with canonical columns."""
    df = sample_transactions()
    p = tmp_path / "test_txns.parquet"
    df.to_parquet(p, index=True)

    loaded = load_transactions(p)
    assert len(loaded) == len(df)
    assert "transaction_id" in loaded.columns
    assert "label_reported_at" in loaded.columns
    assert loaded["ts"].dt.tz is not None


def test_load_transactions_missing_raises_filenotfound(tmp_path: pathlib.Path) -> None:
    """2. Non-existent path raises FileNotFoundError with actionable instructions."""
    missing_path = tmp_path / "non_existent.parquet"
    with pytest.raises(FileNotFoundError, match="missing - run `dvc repro prepare_ieee`"):
        load_transactions(missing_path)


# ============================================================================
# 2. Feature Contract & Overlapping Column Provenance Tests
# ============================================================================


def test_feature_names_contract() -> None:
    """3. feature_names() contains all M2, graph, categorical, and base features with no duplicates."""
    names = feature_names()

    # All declared features in FEATURES present
    for f in FEATURES:
        assert f.name in names, f"Declared feature {f.name} missing from feature_names()"

    # All 4 graph features present
    for g in [
        "graph_component_size",
        "graph_dist_to_fraud",
        "graph_risk_rank",
        "graph_device_degree",
    ]:
        assert g in names, f"Graph feature {g} missing from feature_names()"

    # All categorical features present
    for c in CATEGORICAL:
        assert f"cat__{c}" in names, f"Categorical feature cat__{c} missing from feature_names()"

    # All base features present
    for b in ["amount", "log_amount", "hour_of_day", "day_of_week"]:
        assert b in names, f"Base feature {b} missing from feature_names()"

    # Strict uniqueness (no duplicate column names)
    assert len(names) == len(set(names)), "feature_names() contains duplicate column names"

    # Deterministic order across calls
    assert names == feature_names()


def test_overlapping_column_ownership_and_provenance() -> None:
    """4. Verify single authoritative owner for overlapping columns:
    - hour_of_day, day_of_week: owned by M2 build_point_in_time (calendar)
    - graph_component_size, graph_dist_to_fraud, graph_risk_rank: owned by M2 build_point_in_time
    - graph_device_degree: owned by load_graph_features
    """
    df = sample_transactions()
    pit = build_point_in_time(df, FEATURES)
    graph = load_graph_features(df["account_id"], as_of=df["ts"])
    X, _ = build(df)

    # Prove exact values of overlapping calendar features come from M2 PIT
    pd.testing.assert_series_equal(X["hour_of_day"], pit["hour_of_day"])
    pd.testing.assert_series_equal(X["day_of_week"], pit["day_of_week"])

    # Prove exact values of overlapping graph features come from M2 PIT
    pd.testing.assert_series_equal(X["graph_component_size"], pit["graph_component_size"])
    pd.testing.assert_series_equal(X["graph_dist_to_fraud"], pit["graph_dist_to_fraud"])
    pd.testing.assert_series_equal(X["graph_risk_rank"], pit["graph_risk_rank"])

    # Prove graph_device_degree comes from load_graph_features
    pd.testing.assert_series_equal(X["graph_device_degree"], graph["graph_device_degree"])

    # Verify canonical M2 defaults when snapshots are absent:
    # 3 declared features follow verified M2 contract; graph_device_degree is NaN (unavailable)
    assert (X["graph_component_size"] == 1.0).all()
    assert (X["graph_dist_to_fraud"] == -1.0).all()
    assert (X["graph_risk_rank"] == 0.0).all()
    assert X["graph_device_degree"].isna().all()


def test_graph_snapshots_point_in_time_integration() -> None:
    """5. Verify that supplying graph_snapshots incorporates point-in-time graph features."""
    df = sample_transactions()

    # Create snapshots for ACC_A:
    # Snapshot 1 at 09:00 (before all transactions)
    # Snapshot 2 at 11:30 (between TX102 at 10:30 and TX104 at 12:00)
    t_snap_1 = pd.Timestamp("2026-03-01 09:00:00+00:00")
    t_snap_2 = pd.Timestamp("2026-03-01 11:30:00+00:00")

    snapshots = pd.DataFrame(
        [
            {
                "account_id": "ACC_A",
                "snapshot_at": t_snap_1,
                "component_size": 4.0,
                "dist_to_fraud": 2.0,
                "risk_rank": 0.25,
                "graph_device_degree": 3.0,
            },
            {
                "account_id": "ACC_A",
                "snapshot_at": t_snap_2,
                "component_size": 10.0,
                "dist_to_fraud": 1.0,
                "risk_rank": 0.85,
                "graph_device_degree": 7.0,
            },
        ]
    )

    X, _ = build(df, graph_snapshots=snapshots)

    # TX101 (ACC_A at 10:00): should match Snapshot 1
    assert X.loc[10, "graph_component_size"] == 4.0
    assert X.loc[10, "graph_dist_to_fraud"] == 2.0
    assert X.loc[10, "graph_risk_rank"] == 0.25
    assert X.loc[10, "graph_device_degree"] == 3.0

    # TX102 (ACC_A at 10:30): should match Snapshot 1
    assert X.loc[20, "graph_component_size"] == 4.0
    assert X.loc[20, "graph_dist_to_fraud"] == 2.0
    assert X.loc[20, "graph_risk_rank"] == 0.25
    assert X.loc[20, "graph_device_degree"] == 3.0

    # TX103 (ACC_B at 11:00): no snapshot exists for ACC_B, should get canonical defaults (NaN for device_degree)
    assert X.loc[30, "graph_component_size"] == 1.0
    assert X.loc[30, "graph_dist_to_fraud"] == -1.0
    assert X.loc[30, "graph_risk_rank"] == 0.0
    assert pd.isna(X.loc[30, "graph_device_degree"])

    # TX104 (ACC_A at 12:00): after 11:30, should match Snapshot 2
    assert X.loc[40, "graph_component_size"] == 10.0
    assert X.loc[40, "graph_dist_to_fraud"] == 1.0
    assert X.loc[40, "graph_risk_rank"] == 0.85
    assert X.loc[40, "graph_device_degree"] == 7.0


# ============================================================================
# 3. Frequency Encoding Tests
# ============================================================================


def test_encode_categoricals_fit_and_transform() -> None:
    """6. Frequency encoding computes normalized frequencies and preserves index."""
    df = sample_transactions()
    encoded, mapping = encode_categoricals(df)

    assert isinstance(mapping, dict)
    assert set(mapping.keys()) == set(CATEGORICAL)

    # Check normalized frequencies for 'channel' (web: 2/4 = 0.5, app: 1/4 = 0.25, pos: 1/4 = 0.25)
    assert mapping["channel"]["web"] == 0.5
    assert mapping["channel"]["app"] == 0.25
    assert mapping["channel"]["pos"] == 0.25

    # Check encoded output
    assert encoded.loc[10, "cat__channel"] == 0.5
    assert encoded.loc[20, "cat__channel"] == 0.25
    assert encoded.index.equals(df.index)


def test_encode_categoricals_mapping_reuse_and_unseen_categories() -> None:
    """7. Reusing train mapping maps unseen test categories to 0.0 without altering mapping."""
    train_df = sample_transactions()
    _, train_mapping = encode_categoricals(train_df)

    # Copy of train mapping to verify it is never mutated
    train_mapping_copy = {c: dict(train_mapping[c]) for c in train_mapping}

    test_df = pd.DataFrame(
        {
            "channel": ["web", "kiosk"],  # 'kiosk' is unseen
            "card_network": ["visa", "jcb"],  # 'jcb' is unseen
            "card_type": ["debit", "gift"],
            "device_type": ["mobile", "embedded"],
            "email_domain": ["gmail.com", "unknown.org"],
            "region": ["US", "JP"],
        },
        index=[100, 200],
    )

    encoded_test, test_mapping = encode_categoricals(test_df, mapping=train_mapping)

    # Test mapping is identical in content and values to train mapping
    assert test_mapping == train_mapping_copy
    assert "kiosk" not in train_mapping["channel"]

    # Seen category gets train frequency
    assert encoded_test.loc[100, "cat__channel"] == 0.5
    # Unseen category gets 0.0
    assert encoded_test.loc[200, "cat__channel"] == 0.0
    assert encoded_test.loc[200, "cat__card_network"] == 0.0


def test_encode_categoricals_malformed_mapping_rejected() -> None:
    """8. Malformed or incomplete mappings raise descriptive errors."""
    df = sample_transactions()

    # Incomplete mapping (missing a categorical column)
    incomplete = {c: {"dummy": 1.0} for c in CATEGORICAL if c != "region"}
    with pytest.raises(ValueError, match="Supplied mapping is incomplete"):
        encode_categoricals(df, mapping=incomplete)

    # Invalid mapping type
    with pytest.raises(TypeError, match="mapping must be a dict"):
        encode_categoricals(df, mapping=["not_a_dict"])  # type: ignore[arg-type]

    # Non-dict column mapping
    bad_col_mapping = {c: "not_a_dict" for c in CATEGORICAL}
    with pytest.raises(TypeError, match="Mapping for column.*must be a dict"):
        encode_categoricals(df, mapping=bad_col_mapping)  # type: ignore[arg-type]


def test_encode_categoricals_missing_column_raises_keyerror() -> None:
    """9. Missing categorical column in DataFrame raises KeyError."""
    df = sample_transactions().drop(columns=["card_network"])
    with pytest.raises(KeyError, match="Categorical column 'card_network' missing"):
        encode_categoricals(df)


# ============================================================================
# 4. PIT Adapter Row Identity and Index Safety Tests
# ============================================================================


def test_pit_adapter_non_monotonic_index() -> None:
    """10. Non-monotonic index preserves exact row-to-transaction identity."""
    df = sample_transactions()
    df.index = [100, 20, 50, 5]

    pit = build_point_in_time(df, FEATURES)
    assert pit.index.equals(df.index)
    assert len(pit) == len(df)
    assert list(pit.index) == [100, 20, 50, 5]

    X, _ = build(df)
    assert X.index.equals(df.index)
    assert len(X) == len(df)
    assert list(X.index) == [100, 20, 50, 5]


def test_pit_adapter_duplicate_index_labels() -> None:
    """11. Duplicate input index labels do NOT cause Cartesian explosion or misalignment."""
    df = sample_transactions()
    # Explicit duplicate index labels [0, 0, 1, 1]
    df.index = [0, 0, 1, 1]

    pit = build_point_in_time(df, FEATURES)
    assert len(pit) == 4, f"Expected 4 rows, got {len(pit)} (Cartesian explosion detected)"
    assert list(pit.index) == [0, 0, 1, 1]

    X, _ = build(df)
    assert len(X) == 4
    assert list(X.index) == [0, 0, 1, 1]
    # Check that individual row values match respective transactions
    assert X["amount"].tolist() == [100.0, 50.0, 200.0, 75.0]


def test_pit_adapter_preexisting_collision_column() -> None:
    """12. Input DataFrame containing pre-existing _orig_idx column is preserved intact."""
    df = sample_transactions()
    df["_orig_idx"] = ["custom_col_1", "custom_col_2", "custom_col_3", "custom_col_4"]

    pit = build_point_in_time(df, FEATURES)
    assert len(pit) == len(df)
    assert pit.index.equals(df.index)

    # Ensure caller df was not mutated
    assert list(df["_orig_idx"]) == ["custom_col_1", "custom_col_2", "custom_col_3", "custom_col_4"]


def test_pit_adapter_preserves_caller_row_order_under_internal_sorting() -> None:
    """13. Internal sorting by M2 does not alter caller-visible row order."""
    df = sample_transactions()
    # Reverse chronological order: 12:00, 11:00, 10:30, 10:00
    df_reversed = df.iloc[::-1].copy()

    pit = build_point_in_time(df_reversed, FEATURES)
    assert list(pit.index) == list(df_reversed.index)
    # Check first row corresponds to TX104 (12:00)
    assert pit.loc[40, "hour_of_day"] == 12.0
    # Check last row corresponds to TX101 (10:00)
    assert pit.loc[10, "hour_of_day"] == 10.0


def test_pit_adapter_does_not_mutate_input_dataframe() -> None:
    """14. Caller DataFrame is completely unmutated after build() execution."""
    df = sample_transactions()
    df_before = df.copy(deep=True)

    _ = build(df)

    pd.testing.assert_frame_equal(df, df_before)


# ============================================================================
# 5. Matrix Assembly & Alignment Tests
# ============================================================================


def test_build_matrix_end_to_end() -> None:
    """15. build() produces exact feature_names() columns with preserved row count and index."""
    df = sample_transactions()
    X, mapping = build(df)

    # Exact columns in exact feature_names() order
    expected = feature_names()
    assert list(X.columns) == expected

    # Exact row count and preserved index
    assert len(X) == len(df)
    assert X.index.equals(df.index)

    # Base features verified
    assert X.loc[10, "amount"] == 100.0
    assert np.isclose(X.loc[10, "log_amount"], np.log1p(100.0))
    assert X.loc[10, "hour_of_day"] == 10.0
    assert X.loc[10, "day_of_week"] == 6.0  # 2026-03-01 is a Sunday (day 6)


def test_build_matrix_custom_as_of_column() -> None:
    """16. build() respects a custom as_of_column."""
    df = sample_transactions()
    df["custom_time"] = df["ts"] + pd.Timedelta(hours=2)

    X, _ = build(df, as_of_column="custom_time")
    assert X.loc[10, "hour_of_day"] == 12.0  # 10:00 + 2h = 12:00


def test_build_matrix_missing_declared_feature_raises_keyerror() -> None:
    """17. Missing declared feature in point-in-time output raises KeyError."""
    df = sample_transactions()

    def mock_pit(d, features=None, as_of_column="ts", **kwargs):
        # Intentionally omit txn_count_1m
        res = build_point_in_time(d, features=features, as_of_column=as_of_column, **kwargs)
        return res.drop(columns=["txn_count_1m"])

    with patch("aegis_ml.training.matrix.build_point_in_time", side_effect=mock_pit):
        with pytest.raises(KeyError, match="declared features missing from the built matrix"):
            build(df)


def test_build_matrix_row_count_and_index_mismatch_raises_assertion_error() -> None:
    """18. Row count or index misalignment in sub-components raises AssertionError."""
    df = sample_transactions()

    # Mismatched row count from pit builder
    with patch(
        "aegis_ml.training.matrix.build_point_in_time",
        return_value=pd.DataFrame(index=[1, 2]),
    ):
        with pytest.raises(AssertionError, match="point-in-time builder returned a different row count"):
            build(df)

    # Mismatched index from graph loader
    def mock_graph_bad_index(accs, as_of=None, **kwargs):
        g = load_graph_features(accs, as_of=as_of, **kwargs)
        g.index = [999, 998, 997, 996]
        return g

    with patch("aegis_ml.training.matrix.load_graph_features", side_effect=mock_graph_bad_index):
        with pytest.raises(AssertionError, match="graph feature loader altered row index alignment"):
            build(df)


# ============================================================================
# 6. Temporal Anti-Leakage Across All Feature Families & Label Independence
# ============================================================================


def test_labels_and_future_timestamps_strictly_excluded() -> None:
    """19. Neither is_fraud nor label_reported_at are ever present in X."""
    df = sample_transactions()
    X, _ = build(df)

    assert "is_fraud" not in X.columns
    assert "label_reported_at" not in X.columns
    assert "isFraud" not in X.columns


def test_labels_have_zero_influence_on_feature_values() -> None:
    """20. Mutating is_fraud and label_reported_at has zero influence on computed features."""
    df1 = sample_transactions()
    # Base build
    X1, m1 = build(df1)

    # Create df2 with completely mutated labels and reporting timestamps
    df2 = df1.copy()
    df2["is_fraud"] = [1, 0, 1, 1]
    df2["label_reported_at"] = [
        pd.Timestamp("2026-03-02 00:00:00+00:00"),
        pd.NaT,
        pd.Timestamp("2026-03-02 12:00:00+00:00"),
        pd.Timestamp("2026-03-03 00:00:00+00:00"),
    ]

    X2, m2 = build(df2, mapping=m1)

    # Every single feature across all rows must be bit-exact identical
    pd.testing.assert_frame_equal(X1, X2)


def test_point_in_time_anti_leakage_guarantee_all_families() -> None:
    """21. Future events cannot alter an earlier transaction's point-in-time features.

    Verifies across all feature families using actual M2 computation:
    - Velocity counts: txn_count_1h, txn_count_24h, txn_count_7d
    - Amount sums: amount_sum_minor_1h, amount_sum_minor_24h
    - Distinct entities: distinct_devices_24h, distinct_ips_24h, device_accounts_7d, ip_accounts_7d
    - Account statistics: amount_zscore, seconds_since_prev
    - Calendar features: hour_of_day, day_of_week
    - Graph features: graph_component_size, graph_dist_to_fraud, graph_risk_rank, graph_device_degree
    - Encoded categoricals: cat__channel, cat__card_network, etc.
    - Base features: amount, log_amount
    """
    df_base = sample_transactions()

    # Base graph snapshots
    snap_past = pd.DataFrame(
        [
            {
                "account_id": "ACC_A",
                "snapshot_at": pd.Timestamp("2026-03-01 09:00:00+00:00"),
                "component_size": 3.0,
                "dist_to_fraud": 2.0,
                "risk_rank": 0.20,
                "graph_device_degree": 2.0,
            }
        ]
    )

    X_base, train_mapping = build(df_base, graph_snapshots=snap_past)

    # Future transaction at t = 15:00 with massive amount, new device, new IP
    future_row = pd.DataFrame(
        {
            "transaction_id": ["TX_FUTURE"],
            "account_id": ["ACC_A"],
            "ts": pd.to_datetime(["2026-03-01 15:00:00+00:00"], utc=True),
            "amount": [999_999.0],  # Massive amount
            "channel": ["web"],
            "card_network": ["visa"],
            "card_type": ["debit"],
            "device_type": ["mobile"],
            "email_domain": ["gmail.com"],
            "region": ["US"],
            "device_id": ["DEV_FUTURE"],
            "ip_address": ["192.168.99.99"],
            "is_fraud": [1],
            "label_reported_at": [pd.Timestamp("2026-03-05 15:00:00+00:00")],
        },
        index=[99],
    )

    df_future = pd.concat([df_base, future_row])

    # Future graph snapshot at t = 14:00 (after historical transactions, before future transaction)
    snap_future = pd.DataFrame(
        [
            {
                "account_id": "ACC_A",
                "snapshot_at": pd.Timestamp("2026-03-01 09:00:00+00:00"),
                "component_size": 3.0,
                "dist_to_fraud": 2.0,
                "risk_rank": 0.20,
                "graph_device_degree": 2.0,
            },
            {
                "account_id": "ACC_A",
                "snapshot_at": pd.Timestamp("2026-03-01 14:00:00+00:00"),
                "component_size": 50.0,
                "dist_to_fraud": 0.0,
                "risk_rank": 0.99,
                "graph_device_degree": 25.0,
            },
        ]
    )

    X_future, _ = build(df_future, mapping=train_mapping, graph_snapshots=snap_future)

    # For all historical transactions (indices 10, 20, 30, 40), ALL 28 features must remain BIT-EXACT
    all_features = feature_names()
    pd.testing.assert_frame_equal(
        X_base.loc[df_base.index, all_features],
        X_future.loc[df_base.index, all_features],
    )


# ============================================================================
# 7. Direct Graph Feature Loader Alignment & Contract Tests
# ============================================================================


def test_load_graph_features_direct_non_monotonic_and_duplicate_index() -> None:
    """22. Direct load_graph_features preserves non-monotonic index, duplicate labels,
    and returns exact values for all 4 graph features with zero alignment NaNs across
    repeated accounts, missing snapshots, and future snapshot boundaries."""
    acc_ids = pd.Series(
        ["ACC_A", "ACC_B", "ACC_A", "ACC_C", "ACC_D", "ACC_D"],
        index=[100, 20, 100, 5, 50, 20],
    )
    as_of = pd.Series(
        pd.to_datetime(
            [
                "2026-03-01 10:00:00+00:00",
                "2026-03-01 10:30:00+00:00",
                "2026-03-01 12:00:00+00:00",
                "2026-03-01 11:00:00+00:00",
                "2026-03-01 08:00:00+00:00",
                "2026-03-01 10:00:00+00:00",
            ]
        ),
        index=[100, 20, 100, 5, 50, 20],
    )

    snaps = pd.DataFrame(
        [
            {
                "account_id": "ACC_A",
                "snapshot_at": pd.Timestamp("2026-03-01 09:00:00+00:00"),
                "component_size": 4.0,
                "dist_to_fraud": 2.0,
                "risk_rank": 0.25,
                "graph_device_degree": 3.0,
            },
            {
                "account_id": "ACC_A",
                "snapshot_at": pd.Timestamp("2026-03-01 11:30:00+00:00"),
                "component_size": 10.0,
                "dist_to_fraud": 1.0,
                "risk_rank": 0.85,
                "graph_device_degree": 7.0,
            },
            {
                "account_id": "ACC_B",
                "snapshot_at": pd.Timestamp("2026-03-01 10:00:00+00:00"),
                "component_size": 2.0,
                "dist_to_fraud": 3.0,
                "risk_rank": 0.10,
                "graph_device_degree": 1.0,
            },
            {
                "account_id": "ACC_D",
                "snapshot_at": pd.Timestamp("2026-03-01 09:00:00+00:00"),
                "component_size": 6.0,
                "dist_to_fraud": 4.0,
                "risk_rank": 0.40,
                "graph_device_degree": 5.0,
            },
        ]
    )

    out = load_graph_features(acc_ids, as_of=as_of, graph_snapshots=snaps)

    # 1. Index contract: exact duplicate non-monotonic index preserved
    assert list(out.index) == [100, 20, 100, 5, 50, 20]
    assert len(out) == 6

    # 2. Row 0 (pos 0, index 100): ACC_A at 10:00 -> Snapshot 1 (09:00)
    assert out.iloc[0]["graph_component_size"] == 4.0
    assert out.iloc[0]["graph_dist_to_fraud"] == 2.0
    assert out.iloc[0]["graph_risk_rank"] == 0.25
    assert out.iloc[0]["graph_device_degree"] == 3.0

    # 3. Row 1 (pos 1, index 20): ACC_B at 10:30 -> ACC_B Snapshot (10:00)
    assert out.iloc[1]["graph_component_size"] == 2.0
    assert out.iloc[1]["graph_dist_to_fraud"] == 3.0
    assert out.iloc[1]["graph_risk_rank"] == 0.10
    assert out.iloc[1]["graph_device_degree"] == 1.0

    # 4. Row 2 (pos 2, index 100): ACC_A at 12:00 -> Snapshot 2 (11:30) (different from Row 0 despite same index label 100)
    assert out.iloc[2]["graph_component_size"] == 10.0
    assert out.iloc[2]["graph_dist_to_fraud"] == 1.0
    assert out.iloc[2]["graph_risk_rank"] == 0.85
    assert out.iloc[2]["graph_device_degree"] == 7.0

    # 5. Row 3 (pos 3, index 5): ACC_C at 11:00 -> No snapshot, canonical M2 defaults (NaN for device degree)
    assert out.iloc[3]["graph_component_size"] == 1.0
    assert out.iloc[3]["graph_dist_to_fraud"] == -1.0
    assert out.iloc[3]["graph_risk_rank"] == 0.0
    assert pd.isna(out.iloc[3]["graph_device_degree"])

    # 6. Row 4 (pos 4, index 50): ACC_D at 08:00 -> Snapshot is at 09:00 (future!), returns M2 defaults
    assert out.iloc[4]["graph_component_size"] == 1.0
    assert out.iloc[4]["graph_dist_to_fraud"] == -1.0
    assert out.iloc[4]["graph_risk_rank"] == 0.0
    assert pd.isna(out.iloc[4]["graph_device_degree"])

    # 7. Row 5 (pos 5, index 20): ACC_D at 10:00 -> Snapshot at 09:00 is valid (different from Row 1 despite same index label 20)
    assert out.iloc[5]["graph_component_size"] == 6.0
    assert out.iloc[5]["graph_dist_to_fraud"] == 4.0
    assert out.iloc[5]["graph_risk_rank"] == 0.40
    assert out.iloc[5]["graph_device_degree"] == 5.0


def test_load_graph_features_future_snapshots_never_affect_earlier_transactions() -> None:
    """23. Future snapshot at 11:30 is never visible to earlier transaction at 10:00."""
    acc_ids = ["ACC_A", "ACC_A"]
    as_of = pd.to_datetime(["2026-03-01 10:00:00+00:00", "2026-03-01 12:00:00+00:00"])

    snaps = pd.DataFrame(
        [
            {
                "account_id": "ACC_A",
                "snapshot_at": pd.Timestamp("2026-03-01 11:30:00+00:00"),
                "component_size": 20.0,
                "dist_to_fraud": 1.0,
                "risk_rank": 0.90,
                "graph_device_degree": 15.0,
            }
        ]
    )

    out = load_graph_features(acc_ids, as_of=as_of, graph_snapshots=snaps)

    # Earlier transaction at 10:00 has no snapshot before 10:00 -> canonical defaults
    assert out.iloc[0]["graph_component_size"] == 1.0
    assert out.iloc[0]["graph_dist_to_fraud"] == -1.0
    assert out.iloc[0]["graph_risk_rank"] == 0.0
    assert pd.isna(out.iloc[0]["graph_device_degree"])

    # Later transaction at 12:00 has snapshot at 11:30 -> snapshot values
    assert out.iloc[1]["graph_component_size"] == 20.0
    assert out.iloc[1]["graph_dist_to_fraud"] == 1.0
    assert out.iloc[1]["graph_risk_rank"] == 0.90
    assert out.iloc[1]["graph_device_degree"] == 15.0


def test_load_graph_features_missing_snapshots_follow_verified_m2_contract() -> None:
    """24. Missing snapshots return verified M2 defaults for 3 declared features, and NaN for device degree."""
    acc_ids = ["ACC_1", "ACC_2"]
    as_of = pd.to_datetime(["2026-03-01 10:00:00+00:00", "2026-03-01 11:00:00+00:00"])

    # 1. graph_snapshots is None
    out_none = load_graph_features(acc_ids, as_of=as_of, graph_snapshots=None)
    assert (out_none["graph_component_size"] == 1.0).all()
    assert (out_none["graph_dist_to_fraud"] == -1.0).all()
    assert (out_none["graph_risk_rank"] == 0.0).all()
    assert out_none["graph_device_degree"].isna().all()

    # 2. graph_snapshots is empty DataFrame
    out_empty = load_graph_features(acc_ids, as_of=as_of, graph_snapshots=pd.DataFrame())
    assert (out_empty["graph_component_size"] == 1.0).all()
    assert (out_empty["graph_dist_to_fraud"] == -1.0).all()
    assert (out_empty["graph_risk_rank"] == 0.0).all()
    assert out_empty["graph_device_degree"].isna().all()


def test_matrix_assembly_no_adapter_alignment_nans() -> None:
    """25. Matrix assembly introduces zero NaNs from adapter index alignment when snapshots are provided."""
    df = sample_transactions()
    # Give all accounts snapshots
    snaps = pd.DataFrame(
        [
            {
                "account_id": "ACC_A",
                "snapshot_at": pd.Timestamp("2026-03-01 09:00:00+00:00"),
                "component_size": 4.0,
                "dist_to_fraud": 2.0,
                "risk_rank": 0.25,
                "graph_device_degree": 3.0,
            },
            {
                "account_id": "ACC_B",
                "snapshot_at": pd.Timestamp("2026-03-01 09:00:00+00:00"),
                "component_size": 2.0,
                "dist_to_fraud": 3.0,
                "risk_rank": 0.10,
                "graph_device_degree": 1.0,
            },
        ]
    )

    X, _ = build(df, graph_snapshots=snaps)

    graph_cols = [
        "graph_component_size",
        "graph_dist_to_fraud",
        "graph_risk_rank",
        "graph_device_degree",
    ]
    # No NaNs anywhere in graph features for snapshotted accounts
    assert not X[graph_cols].isna().any().any()


def test_build_with_fit_df_isolation() -> None:
    """26. build() with fit_df isolates frequency encoding to fit_df while preserving PIT context on df."""
    df = sample_transactions()
    # TX101 (web), TX102 (app), TX103 (web), TX104 (pos)
    # df.iloc[:2] only has web and app; pos is unseen
    fit_df = df.iloc[:2].copy()

    X, mapping = build(df, fit_df=fit_df)

    # Mapping contains only categories from fit_df
    assert "pos" not in mapping["channel"]
    assert mapping["channel"]["web"] == 0.5
    assert mapping["channel"]["app"] == 0.5

    # Rows with seen categories get fit_df frequencies
    assert X.loc[10, "cat__channel"] == 0.5
    assert X.loc[20, "cat__channel"] == 0.5
    assert X.loc[30, "cat__channel"] == 0.5
    # Row with unseen category (pos at index 40) gets fallback 0.0
    assert X.loc[40, "cat__channel"] == 0.0

    # Full matrix has all declared features
    assert list(X.columns) == feature_names()
    assert len(X) == len(df)


def test_build_conflicting_mapping_and_fit_df_raises() -> None:
    """27. Supplying both mapping and fit_df to build() raises ValueError."""
    df = sample_transactions()
    _, dummy_mapping = encode_categoricals(df)
    with pytest.raises(ValueError, match="Cannot supply both mapping and fit_df"):
        build(df, mapping=dummy_mapping, fit_df=df.iloc[:2])


def test_build_matrix_with_sub_cent_prepared_transactions() -> None:
    """28. Regression test: prepared IEEE-CIS transactions with sub-cent source amounts
    and integer amount_minor pass cleanly through build() without raising fractional
    minor units exception, producing exact 28 columns in authoritative order, preserving
    source amounts in base features, and computing canonical amount_zscore from amount_minor.
    """
    df = pd.DataFrame(
        {
            "transaction_id": ["TX201", "TX202", "TX203", "TX204"],
            "account_id": ["ACC_IEEE", "ACC_IEEE", "ACC_IEEE", "ACC_IEEE"],
            "ts": pd.to_datetime(
                [
                    "2026-03-01 10:00:00+00:00",
                    "2026-03-01 10:30:00+00:00",
                    "2026-03-01 11:00:00+00:00",
                    "2026-03-01 12:00:00+00:00",
                ],
                utc=True,
            ),
            # Representative IEEE-CIS sub-cent source amounts
            "amount": [75.887, 16.495, 42.294, 3.595],
            # Corresponding quantized integer cents
            "amount_minor": [7589, 1650, 4229, 360],
            "channel": ["web", "app", "web", "pos"],
            "card_network": ["visa", "mc", "visa", "amex"],
            "card_type": ["debit", "credit", "debit", "credit"],
            "device_type": ["mobile", "desktop", "mobile", "tablet"],
            "email_domain": ["gmail.com", "yahoo.com", "gmail.com", "hotmail.com"],
            "region": ["US", "US", "CA", "GB"],
            "device_id": ["DEV_1", "DEV_2", "DEV_1", "DEV_3"],
            "ip_address": ["10.0.0.1", "10.0.0.2", "10.0.0.1", "10.0.0.3"],
            "is_fraud": [0, 0, 0, 0],
            "label_reported_at": [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        }
    )

    X, _ = build(df)

    # 1. Confirms no fractional minor units exception occurs and matrix has exact 28 columns
    assert list(X.columns) == feature_names()
    assert len(X) == 4

    # 2. Confirms base features use the original unrounded source amount
    np.testing.assert_array_equal(X["amount"].to_numpy(), np.array([75.887, 16.495, 42.294, 3.595]))
    np.testing.assert_allclose(X["log_amount"].to_numpy(), np.log1p(np.array([75.887, 16.495, 42.294, 3.595])))

    # 3. Confirms canonical amount_zscore is produced through existing M2 implementation (Welford on amount_minor)
    assert not X["amount_zscore"].isna().any()
    assert X.iloc[0]["amount_zscore"] == 0.0
    assert X.iloc[1]["amount_zscore"] == 0.0
    # Mean of [7589, 1650] = 4619.5, std = 4199.505923320668
    expected_zscore_2 = (4229.0 - 4619.5) / 4199.505923320668
    assert np.isclose(X.iloc[2]["amount_zscore"], expected_zscore_2, rtol=1e-5)
