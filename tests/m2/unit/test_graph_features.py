"""Unit tests for AegisFlow M2 Hourly Graph Feature Job (aegis_ml.features.graph_features).

Tests:
- Exact UTC snapshot timestamp normalization and hourly bucket helpers
- Undirected entity graph construction with canonical prefixes (`acct:`, `dev:`, `ip:`, `mer:`)
- Direction-invariant and order-invariant duplicate-edge aggregation and invalid-weight validation
- Fraud-personalised weighted PageRank (including exact 0.0 without fraud seeds & disconnected 0.0)
- Unweighted multi-source BFS hop distances ignoring large edge strengths
- In-memory SQLite regression tests for historical edge-weight reconstruction & fraud-label eligibility
- Config / .env URL resolution without hardcoded secrets, CLI --redis-url wiring, resource cleanup,
  and partial Postgres/Redis publication failure observability + safe retries
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import uuid

import fakeredis
import networkx as nx
import pytest
import redis as redis_lib
from sqlalchemy import create_engine, text

from aegis_ml.features import definitions as D
from aegis_ml.features.graph_features import (
    ENTITY_PREFIXES,
    GraphFeatureSnapshot,
    HistoricalEdgeReconstructionError,
    build_graph_from_edges,
    compute_graph_features,
    fetch_active_account_ids,
    fetch_entity_edges,
    fetch_fraud_account_ids,
    hourly_bucket_for,
    main,
    make_node_id,
    normalize_snapshot_time,
    parse_args,
    publish_graph_features_to_redis,
    resolve_database_url,
    resolve_redis_url,
    run_job,
)
import ingestion.online_features as OF


# ============================================================================
# 1. Timestamp Normalization & Bucket Tests (Remediation E)
# ============================================================================


def test_normalize_snapshot_time_none_preserves_exact_current_utc_instant():
    before = datetime.now(timezone.utc)
    t_norm = normalize_snapshot_time(None)
    after = datetime.now(timezone.utc)

    assert t_norm.tzinfo == timezone.utc
    assert before <= t_norm <= after

    bucket = hourly_bucket_for(t_norm)
    assert bucket.tzinfo == timezone.utc
    assert bucket == t_norm.replace(minute=0, second=0, microsecond=0)


def test_normalize_snapshot_time_preserves_exact_historical_timestamp_and_converts_tz():
    raw_utc = "2026-11-04T15:43:21+00:00"
    assert normalize_snapshot_time(raw_utc) == datetime(2026, 11, 4, 15, 43, 21, tzinfo=timezone.utc)

    # Non-UTC offset (+05:30 -> 10:13:21Z exact)
    ist = timezone(timedelta(hours=5, minutes=30))
    dt_ist = datetime(2026, 11, 4, 15, 43, 21, tzinfo=ist)
    assert normalize_snapshot_time(dt_ist) == datetime(2026, 11, 4, 10, 13, 21, tzinfo=timezone.utc)


def test_normalize_snapshot_time_naive_assumed_utc_without_truncation():
    naive = datetime(2026, 11, 4, 18, 25, 30)
    t_norm = normalize_snapshot_time(naive)
    assert t_norm == datetime(2026, 11, 4, 18, 25, 30, tzinfo=timezone.utc)


def test_normalize_snapshot_time_invalid_type_raises():
    with pytest.raises(TypeError, match="Invalid timestamp type"):
        normalize_snapshot_time(123456789)  # type: ignore


# ============================================================================
# 2. Undirected Graph Construction & Edge Validation Tests (Remediation A)
# ============================================================================


def test_make_node_id_preserves_canonical_entity_prefixes():
    assert ENTITY_PREFIXES["account"] == "acct"
    assert ENTITY_PREFIXES["device"] == "dev"
    assert ENTITY_PREFIXES["ip"] == "ip"
    assert ENTITY_PREFIXES["merchant"] == "mer"

    assert make_node_id("account", "a-1") == "acct:a-1"
    assert make_node_id("device", "d-1") == "dev:d-1"
    assert make_node_id("ip", "10.0.0.1") == "ip:10.0.0.1"
    assert make_node_id("merchant", "m-1") == "mer:m-1"


def test_build_graph_from_edge_tuples_and_weights():
    edges = [
        ("account", "acc-1", "device", "dev-1", 1.5),
        ("account", "acc-1", "ip", "10.0.0.1", 1.0),
        ("account", "acc-2", "device", "dev-1", 2.0),
    ]
    G = build_graph_from_edges(edges)
    assert isinstance(G, nx.Graph)
    assert not G.is_directed()
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 3

    assert G.has_edge("acct:acc-1", "dev:dev-1")
    assert G.has_edge("dev:dev-1", "acct:acc-1")
    assert G["acct:acc-1"]["dev:dev-1"]["weight"] == pytest.approx(1.5)


def test_build_graph_from_edge_mappings():
    edges = [
        {"src_type": "account", "src_id": "acc-1", "dst_type": "merchant", "dst_id": "m-1", "weight": 2.5},
        {"src_type": "account", "src_id": "acc-2", "dst_type": "account", "dst_id": "acc-1", "weight": 1.0},
    ]
    G = build_graph_from_edges(edges)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 3
    assert G["acct:acc-1"]["mer:m-1"]["weight"] == pytest.approx(2.5)
    assert G["acct:acc-2"]["acct:acc-1"]["weight"] == pytest.approx(1.0)


def test_build_graph_consolidates_duplicate_and_reversed_edges():
    edges = [
        ("account", "acc-1", "device", "dev-1", 1.0),
        ("device", "dev-1", "account", "acc-1", 2.5),
    ]
    G = build_graph_from_edges(edges)
    assert G.number_of_edges() == 1
    assert G["acct:acc-1"]["dev:dev-1"]["weight"] == pytest.approx(3.5)


def test_build_graph_weight_aggregation_and_personalized_pagerank_parity():
    """Multiple edges between endpoints aggregate additively and PageRank reflects aggregated strength."""
    edges = [
        ("account", "acc-1", "account", "acc-2", 2.0),
        ("account", "acc-1", "account", "acc-2", 3.0),
        ("account", "acc-1", "account", "acc-3", 5.0),
    ]
    G = build_graph_from_edges(edges)

    assert G.number_of_edges() == 2
    assert G["acct:acc-1"]["acct:acc-2"]["weight"] == pytest.approx(5.0)
    assert G["acct:acc-1"]["acct:acc-3"]["weight"] == pytest.approx(5.0)

    snaps = {
        s.account_id: s
        for s in compute_graph_features(G, ["acc-1", "acc-2", "acc-3"], fraud_account_ids=["acc-1"])
    }
    assert snaps["acc-2"].risk_rank == pytest.approx(snaps["acc-3"].risk_rank)
    assert snaps["acc-2"].risk_rank > 0.2


def test_build_graph_handles_missing_and_invalid_weights():
    # None defaults safely to 1.0
    G_none = build_graph_from_edges([("account", "a1", "device", "d1", None)])
    assert G_none["acct:a1"]["dev:d1"]["weight"] == 1.0

    for bad_w in (0.0, -2.5, float("nan"), float("inf"), True, "not-a-float"):
        with pytest.raises(ValueError):
            build_graph_from_edges([("account", "a1", "device", "d1", bad_w)])

    with pytest.raises(ValueError):
        build_graph_from_edges([("account", "", "device", "d1", 1.0)])


def test_build_graph_empty_list():
    G = build_graph_from_edges([])
    assert isinstance(G, nx.Graph)
    assert not G.is_directed()
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


# ============================================================================
# 3. Fraud-Personalised PageRank & Undirected Feature Tests (Remediations A, B)
# ============================================================================


def test_empty_graph_and_accounts():
    G = nx.Graph()
    results = compute_graph_features(G, account_ids=[])
    assert results == []


def test_empty_graph_with_isolated_accounts_falls_back_to_defaults():
    G = nx.Graph()
    acc1 = str(uuid.uuid4())
    acc2 = str(uuid.uuid4())
    results = compute_graph_features(G, account_ids=[acc1, acc2])

    assert len(results) == 2
    for r in results:
        assert isinstance(r, GraphFeatureSnapshot)
        assert r.pagerank == 0.0
        assert r.degree == 0
        assert r.in_degree == 0
        assert r.out_degree == 0
        assert r.clustering_coefficient == 0.0
        assert r.component_size == 1.0
        assert r.dist_to_fraud == -1.0
        assert r.risk_rank == 0.0
        assert r.features == {
            "component_size": 1.0,
            "dist_to_fraud": -1.0,
            "risk_rank": 0.0,
        }


def test_isolated_account_marked_as_fraud():
    G = nx.Graph()
    acc_fraud = str(uuid.uuid4())
    acc_clean = str(uuid.uuid4())

    results = compute_graph_features(
        G,
        account_ids=[acc_fraud, acc_clean],
        fraud_account_ids=[acc_fraud],
    )

    r_fraud = next(r for r in results if r.account_id == acc_fraud)
    r_clean = next(r for r in results if r.account_id == acc_clean)

    assert r_fraud.dist_to_fraud == 0.0
    assert r_fraud.component_size == 1.0
    assert r_fraud.pagerank == 0.0
    assert r_fraud.risk_rank == 0.0
    assert r_clean.dist_to_fraud == -1.0
    assert r_clean.risk_rank == 0.0


def test_empty_fraud_seeds_produce_exact_zero_risk_rank_on_connected_graph():
    """Verify no global PageRank fallback occurs when eligible fraud seeds are empty."""
    edges = [
        ("account", "A", "device", "D", 1.0),
        ("account", "B", "device", "D", 4.0),
        ("account", "C", "device", "D", 1.0),
        ("account", "E", "device", "F", 1.0),
    ]
    G = build_graph_from_edges(edges)

    for seeds in (None, []):
        results = compute_graph_features(G, ["A", "B", "C", "E"], fraud_account_ids=seeds)
        assert all(r.dist_to_fraud == -1.0 for r in results)
        assert all(r.risk_rank == 0.0 for r in results)
        assert all(r.pagerank == 0.0 for r in results)


def test_personalized_pagerank_ring_and_disconnected_zero_rank():
    """Known fraud seeds drive personalised PageRank; disconnected components receive 0.0."""
    edges = [
        ("account", "A", "device", "D", 1.0),
        ("account", "B", "device", "D", 3.0),
        ("account", "C", "device", "D", 1.0),
        ("account", "E", "device", "F", 1.0),
        ("account", "B", "device", "D", 1.0),
    ]
    G = build_graph_from_edges(edges)
    by_acc = {r.account_id: r for r in compute_graph_features(G, ["A", "B", "C", "E"], ["A"])}

    assert by_acc["A"].component_size == 4.0
    assert by_acc["B"].component_size == 4.0
    assert by_acc["C"].component_size == 4.0
    assert by_acc["E"].component_size == 2.0

    assert [by_acc[a].dist_to_fraud for a in ("A", "B", "C", "E")] == [0.0, 2.0, 2.0, -1.0]

    # Disconnected account E must receive exact 0.0 (no artificial global PageRank)
    assert by_acc["E"].risk_rank == 0.0
    assert by_acc["A"].risk_rank > 0.0
    assert 0.0 < by_acc["C"].risk_rank < by_acc["B"].risk_rank


def test_hop_distance_ignores_heavy_edge_strength():
    heavy = [
        ("account", "A", "device", "D", 1000.0),
        ("account", "B", "device", "D", 1000.0),
    ]
    G = build_graph_from_edges(heavy)
    by_acc = {r.account_id: r for r in compute_graph_features(G, ["A", "B"], ["A"])}
    assert by_acc["A"].dist_to_fraud == 0.0
    assert by_acc["B"].dist_to_fraud == 2.0


def test_reversing_edge_directions_yields_identical_features():
    """Undirected feature computation must be invariant to reversing edge endpoint directions."""
    t_snap = datetime(2026, 11, 4, 15, 0, 0, tzinfo=timezone.utc)
    forward_edges = [
        ("account", "acc-fraud", "device", "dev-1", 1.2),
        ("account", "acc-mid", "device", "dev-1", 2.8),
        ("account", "acc-leaf", "account", "acc-mid", 1.5),
        ("account", "acc-iso-comp", "merchant", "mer-1", 3.0),
    ]
    reversed_edges = [(d_t, d_id, s_t, s_id, w) for (s_t, s_id, d_t, d_id, w) in reversed(forward_edges)]

    G_fwd = build_graph_from_edges(forward_edges)
    G_rev = build_graph_from_edges(reversed_edges)

    accs = ["acc-fraud", "acc-mid", "acc-leaf", "acc-iso-comp"]
    res_fwd = compute_graph_features(G_fwd, accs, ["acc-fraud"], snapshot_at=t_snap)
    res_rev = compute_graph_features(G_rev, accs, ["acc-fraud"], snapshot_at=t_snap)

    for rf, rr in zip(res_fwd, res_rev):
        assert rf.account_id == rr.account_id
        assert rf.degree == rr.degree
        assert rf.in_degree == rr.in_degree == rf.degree
        assert rf.out_degree == rr.out_degree == rf.degree
        assert rf.component_size == rr.component_size
        assert rf.dist_to_fraud == rr.dist_to_fraud
        assert rf.clustering_coefficient == pytest.approx(rr.clustering_coefficient)
        assert rf.risk_rank == pytest.approx(rr.risk_rank)


def test_synthetic_network_topology_and_metrics():
    """Verify undirected degrees, multi-hop fraud distances, component sizes, and personalised PageRank."""
    acc_fraud = "acc-fraud-1"
    acc_shared = "acc-shared-2"
    acc_transfer = "acc-transfer-3"
    acc_isolated = "acc-isolated-4"

    edges = [
        ("account", acc_fraud, "device", "dev-1", 1.0),
        ("account", acc_shared, "device", "dev-1", 1.0),
        ("account", acc_transfer, "account", acc_shared, 1.0),
    ]
    G = build_graph_from_edges(edges)

    results = compute_graph_features(
        G=G,
        account_ids=[acc_fraud, acc_shared, acc_transfer, acc_isolated],
        fraud_account_ids=[acc_fraud],
    )
    by_acc = {r.account_id: r for r in results}

    assert by_acc[acc_fraud].dist_to_fraud == 0.0
    assert by_acc[acc_shared].dist_to_fraud == 2.0
    assert by_acc[acc_transfer].dist_to_fraud == 3.0
    assert by_acc[acc_isolated].dist_to_fraud == -1.0

    assert by_acc[acc_fraud].component_size == 4.0
    assert by_acc[acc_shared].component_size == 4.0
    assert by_acc[acc_transfer].component_size == 4.0
    assert by_acc[acc_isolated].component_size == 1.0

    # Symmetric undirected degrees
    assert by_acc[acc_transfer].degree == 1
    assert by_acc[acc_transfer].in_degree == 1
    assert by_acc[acc_transfer].out_degree == 1

    assert by_acc[acc_shared].degree == 2
    assert by_acc[acc_shared].in_degree == 2
    assert by_acc[acc_shared].out_degree == 2

    assert by_acc[acc_isolated].degree == 0

    for acc in [acc_fraud, acc_shared, acc_transfer]:
        assert by_acc[acc].pagerank > 0.0
        assert by_acc[acc].risk_rank == by_acc[acc].pagerank
    assert by_acc[acc_isolated].pagerank == 0.0
    assert by_acc[acc_isolated].risk_rank == 0.0


def test_triangle_clustering_coefficient():
    """Verify clustering coefficient on a 3-account triangle is 1.0 and risk_rank=0.0 without fraud."""
    edges = [
        ("account", "acc-tri-a", "account", "acc-tri-b", 1.0),
        ("account", "acc-tri-b", "account", "acc-tri-c", 1.0),
        ("account", "acc-tri-c", "account", "acc-tri-a", 1.0),
    ]
    G = build_graph_from_edges(edges)
    results = compute_graph_features(G, account_ids=["acc-tri-a", "acc-tri-b", "acc-tri-c"])
    for r in results:
        assert r.clustering_coefficient == pytest.approx(1.0)
        assert r.component_size == 3.0
        assert r.risk_rank == 0.0


def test_multiple_fraud_accounts_selects_shortest_distance():
    """Multi-source BFS selects minimum unweighted hop distance when multiple fraud accounts exist."""
    f1 = "f-1"
    f2 = "f-2"
    target = "acc-target"

    edges = [
        ("account", target, "account", f1, 1.0),
        ("account", target, "device", "d-1", 1.0),
        ("device", "d-1", "device", "d-2", 1.0),
        ("device", "d-2", "account", f2, 1.0),
    ]
    G = build_graph_from_edges(edges)

    results = compute_graph_features(
        G=G,
        account_ids=[target],
        fraud_account_ids=[f1, f2],
    )
    assert len(results) == 1
    assert results[0].dist_to_fraud == 1.0


def test_check_constraint_bounds_enforcement():
    """Verify compute_graph_features enforces finite check-constraint bounds."""
    G = nx.Graph()
    G.add_node("acct:acc-clamp")
    results = compute_graph_features(G, account_ids=["acc-clamp"], fraud_account_ids=["acc-clamp"])
    assert len(results) == 1
    r = results[0]
    assert 0.0 <= r.pagerank < float("inf")
    assert 0.0 <= r.clustering_coefficient <= 1.0
    assert r.degree >= 0
    assert r.in_degree >= 0
    assert r.out_degree >= 0


# ============================================================================
# 4. In-Memory Temporal Eligibility & Historical Reconstruction Tests (C, D)
# ============================================================================


def _create_sqlite_graph_schema(engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE accounts (
                    account_id text PRIMARY KEY,
                    status text NOT NULL
                );
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE transactions (
                    txn_id text PRIMARY KEY,
                    account_id text NOT NULL,
                    destination_account_id text,
                    device_id text,
                    ip_id text,
                    merchant_id text,
                    channel text NOT NULL,
                    occurred_at timestamp NOT NULL
                );
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE entity_edges (
                    src_type text NOT NULL,
                    src_id text NOT NULL,
                    dst_type text NOT NULL,
                    dst_id text NOT NULL,
                    edge_type text NOT NULL,
                    weight float NOT NULL,
                    first_seen_at timestamp NOT NULL,
                    last_seen_at timestamp NOT NULL
                );
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE ground_truth_labels (
                    label_id text PRIMARY KEY,
                    txn_id text,
                    account_id text,
                    label_target text NOT NULL,
                    is_fraud boolean NOT NULL,
                    confidence float NOT NULL,
                    labeled_at timestamp NOT NULL,
                    incident_start_at timestamp,
                    incident_end_at timestamp
                );
                """
            )
        )


def test_fetch_entity_edges_reconstructs_spanning_txn_weights_and_blocks_unbacked():
    engine = create_engine("sqlite:///:memory:")
    _create_sqlite_graph_schema(engine)

    t0 = datetime(2026, 11, 4, 10, 0, 0, tzinfo=timezone.utc)
    t_snap_1 = datetime(2026, 11, 4, 10, 30, 0, tzinfo=timezone.utc)
    t_later = datetime(2026, 11, 4, 11, 0, 0, tzinfo=timezone.utc)
    t_snap_2 = datetime(2026, 11, 4, 11, 30, 0, tzinfo=timezone.utc)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO transactions (txn_id, account_id, device_id, channel, occurred_at) VALUES
                    ('tx-1', 'acc-1', 'dev-1', 'card', :t0),
                    ('tx-2', 'acc-1', 'dev-1', 'card', :t_later);
                """
            ),
            {"t0": t0, "t_later": t_later},
        )
        conn.execute(
            text(
                """
                INSERT INTO entity_edges
                    (src_type, src_id, dst_type, dst_id, edge_type, weight, first_seen_at, last_seen_at)
                VALUES
                    ('account', 'acc-1', 'device', 'dev-1', 'uses_device', 2.0, :t0, :t_later);
                """
            ),
            {"t0": t0, "t_later": t_later},
        )

    # Reconstructing T1 must recover weight 1.0 (zero future contribution)
    edges_t1 = fetch_entity_edges(engine, snapshot_at=t_snap_1)
    assert edges_t1 == [("account", "acc-1", "device", "dev-1", 1.0)]

    # Later snapshot T2 sees accumulated weight 2.0
    edges_t2 = fetch_entity_edges(engine, snapshot_at=t_snap_2)
    assert edges_t2 == [("account", "acc-1", "device", "dev-1", 2.0)]

    # Adding an unbacked spanning non-transaction edge must fail closed rather than leaking weight
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO entity_edges
                    (src_type, src_id, dst_type, dst_id, edge_type, weight, first_seen_at, last_seen_at)
                VALUES
                    ('account', 'acc-1', 'account', 'acc-2', 'shares_card_with', 4.0, :t0, :t_later);
                """
            ),
            {"t0": t0, "t_later": t_later},
        )

    with pytest.raises(HistoricalEdgeReconstructionError):
        fetch_entity_edges(engine, snapshot_at=t_snap_1)

    engine.dispose()


def test_fetch_fraud_account_ids_temporal_incident_and_txn_guards():
    engine = create_engine("sqlite:///:memory:")
    _create_sqlite_graph_schema(engine)

    t_past = datetime(2026, 11, 1, 9, 0, 0, tzinfo=timezone.utc)
    t_inc_end = datetime(2026, 11, 2, 12, 0, 0, tzinfo=timezone.utc)
    t_conf_past = datetime(2026, 11, 3, 12, 0, 0, tzinfo=timezone.utc)
    t_snap = datetime(2026, 11, 4, 15, 0, 0, tzinfo=timezone.utc)
    t_future = datetime(2026, 11, 5, 10, 0, 0, tzinfo=timezone.utc)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO transactions (txn_id, account_id, channel, occurred_at) VALUES
                    ('tx-elig', 'acc-txn-elig', 'card', :t_past),
                    ('tx-future', 'acc-txn-future', 'card', :t_future),
                    ('tx-oob', 'acc-txn-oob', 'card', :t_past);
                """
            ),
            {"t_past": t_past, "t_future": t_future},
        )
        conn.execute(
            text(
                """
                INSERT INTO ground_truth_labels
                    (label_id, txn_id, account_id, label_target, is_fraud, confidence, labeled_at, incident_start_at, incident_end_at)
                VALUES
                    -- Eligible timeless account fraud
                    ('l1', NULL, 'acc-timeless', 'account', 1, 0.95, :t_conf_past, NULL, NULL),
                    -- Eligible bounded completed & confirmed account fraud
                    ('l2', NULL, 'acc-bounded-ok', 'account', 1, 0.90, :t_conf_past, :t_past, :t_inc_end),
                    -- Ineligible: confirmed after snapshot
                    ('l3', NULL, 'acc-late-label', 'account', 1, 1.0, :t_future, :t_past, :t_inc_end),
                    -- Ineligible: future incident_start_at
                    ('l4', NULL, 'acc-future-inc', 'account', 1, 1.0, :t_conf_past, :t_future, :t_future),
                    -- Ineligible: sub-threshold confidence (< 0.85)
                    ('l5', NULL, 'acc-low-conf', 'account', 1, 0.40, :t_conf_past, NULL, NULL),
                    -- Eligible transaction-backed fraud
                    ('l6', 'tx-elig', NULL, 'transaction', 1, 0.99, :t_conf_past, :t_past, :t_inc_end),
                    -- Ineligible: transaction occurred AFTER snapshot even though labeled_at <= snapshot
                    ('l7', 'tx-future', NULL, 'transaction', 1, 0.99, :t_conf_past, NULL, NULL),
                    -- Ineligible: transaction occurred outside label incident window
                    ('l8', 'tx-oob', NULL, 'transaction', 1, 0.99, :t_conf_past, :t_inc_end, :t_conf_past);
                """
            ),
            {"t_past": t_past, "t_inc_end": t_inc_end, "t_conf_past": t_conf_past, "t_future": t_future},
        )

    fraud_ids = fetch_fraud_account_ids(engine, snapshot_at=t_snap)
    assert fraud_ids == {"acc-timeless", "acc-bounded-ok", "acc-txn-elig"}

    # Earlier snapshot before any confirmation -> empty set
    assert fetch_fraud_account_ids(engine, snapshot_at=t_past) == set()
    engine.dispose()


# ============================================================================
# 5. CLI Wiring, Configuration, Resource Lifecycle & Partial Failure Tests (F, G, H)
# ============================================================================


def test_parse_args_defaults():
    args = parse_args([])
    assert args.once is False
    assert args.snapshot_at is None
    assert args.interval_s == 3600
    assert args.db_url is None
    assert args.redis_url is None
    assert args.skip_redis is False
    assert args.batch_size == 500
    assert args.include_closed is False


def test_parse_args_custom_values():
    args = parse_args(
        [
            "--once",
            "--snapshot-at",
            "2026-11-04T12:15:00Z",
            "--interval-s",
            "1800",
            "--db-url",
            "postgresql://user:pass@db:5432/test",
            "--redis-url",
            "redis://redis:6379/2",
            "--skip-redis",
            "--batch-size",
            "100",
            "--include-closed",
        ]
    )
    assert args.once is True
    assert args.snapshot_at == "2026-11-04T12:15:00Z"
    assert args.interval_s == 1800
    assert args.db_url == "postgresql://user:pass@db:5432/test"
    assert args.redis_url == "redis://redis:6379/2"
    assert args.skip_redis is True
    assert args.batch_size == 100
    assert args.include_closed is True


def test_resolve_urls_respect_overrides_and_env_without_hardcoded_passwords(monkeypatch):
    # Prevent reading repo .env in this isolated config test
    monkeypatch.setattr("aegis_ml.features.graph_features._load_dotenv", lambda: None)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.delenv("DB_PASS", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)
    monkeypatch.delenv("REDIS_DB", raising=False)

    with pytest.raises(ValueError, match="Database connection URL is not configured"):
        resolve_database_url()

    assert resolve_redis_url() == "redis://localhost:6380/0"

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5433/aegisflow")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6380/1")
    assert resolve_database_url() == "postgresql+psycopg://u:p@localhost:5433/aegisflow"
    assert resolve_redis_url() == "redis://localhost:6380/1"

    assert (
        resolve_database_url("postgresql://cli_user:cli_pass@h:5433/db")
        == "postgresql+psycopg://cli_user:cli_pass@h:5433/db"
    )
    assert resolve_redis_url("redis://cli-host:6390/4") == "redis://cli-host:6390/4"


def test_cli_main_once_passes_redis_url_and_handles_partial_failure():
    with patch("aegis_ml.features.graph_features.run_job", return_value={"status": "success"}) as mock_run:
        exit_code = main(["--once", "--redis-url", "redis://custom:6399/3", "--batch-size", "250"])
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["redis_url"] == "redis://custom:6399/3"
        assert mock_run.call_args.kwargs["batch_size"] == 250

    with patch(
        "aegis_ml.features.graph_features.run_job",
        return_value={"status": "partial_failure", "publication_status": "postgres_only", "redis_error": "down"},
    ):
        assert main(["--once"]) == 1

    with patch("aegis_ml.features.graph_features.run_job", side_effect=RuntimeError("Database failure")):
        assert main(["--once"]) == 1


def test_cli_run_job_uses_redis_url_publishes_contract_and_closes_owned_clients(tmp_path: Path):
    """End-to-end mocked CLI test: --redis-url -> Redis.from_url -> HSET + TTL=7200 -> close()."""
    db_file = tmp_path / "graph_cli.db"
    sqlite_url = f"sqlite:///{db_file}"
    setup_engine = create_engine(sqlite_url)
    _create_sqlite_graph_schema(setup_engine)

    with setup_engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE graph_feature_snapshots (
                    account_id text NOT NULL,
                    snapshot_at timestamp NOT NULL,
                    pagerank float NOT NULL,
                    degree integer NOT NULL,
                    in_degree integer NOT NULL,
                    out_degree integer NOT NULL,
                    clustering_coefficient float NOT NULL,
                    features text NOT NULL,
                    created_at timestamp,
                    PRIMARY KEY (account_id, snapshot_at)
                );
                """
            )
        )
        t_edge = datetime(2026, 11, 4, 14, 0, 0, tzinfo=timezone.utc)
        conn.execute(
            text("INSERT INTO accounts (account_id, status) VALUES ('acc-1', 'active'), ('acc-2', 'active');")
        )
        conn.execute(
            text(
                """
                INSERT INTO entity_edges
                    (src_type, src_id, dst_type, dst_id, edge_type, weight, first_seen_at, last_seen_at)
                VALUES
                    ('account', 'acc-1', 'device', 'dev-1', 'uses_device', 1.0, :te, :te),
                    ('account', 'acc-2', 'device', 'dev-1', 'uses_device', 1.0, :te, :te);
                """
            ),
            {"te": t_edge},
        )
        conn.execute(
            text(
                """
                INSERT INTO ground_truth_labels
                    (label_id, account_id, label_target, is_fraud, confidence, labeled_at)
                VALUES
                    ('l1', 'acc-1', 'account', 1, 0.95, :te);
                """
            ),
            {"te": t_edge},
        )
    setup_engine.dispose()

    fake_r = fakeredis.FakeRedis()
    real_close = fake_r.close
    close_calls = 0

    def tracked_close():
        nonlocal close_calls
        close_calls += 1
        return real_close()

    fake_r.close = tracked_close  # type: ignore[method-assign]

    snap_mid_hour = "2026-11-04T14:27:19Z"
    with patch("aegis_ml.features.graph_features.redis.Redis.from_url", return_value=fake_r) as mock_from_url:
        exit_code = main(
            [
                "--once",
                "--db-url",
                sqlite_url,
                "--redis-url",
                "redis://cli-redis:6391/5",
                "--snapshot-at",
                snap_mid_hour,
            ]
        )

    assert exit_code == 0
    mock_from_url.assert_called_once_with("redis://cli-redis:6391/5", decode_responses=False)
    assert close_calls == 1

    # Verify Redis hash contract & TTL == 7200
    for aid, expected_dist in (("acc-1", 0.0), ("acc-2", 2.0)):
        key = f"gfeat:acct:{aid}"
        assert fake_r.ttl(key) == pytest.approx(D.GRAPH_TTL_S, abs=5)
        raw = fake_r.hgetall(key)
        assert b"computed_at" in raw
        assert raw[b"computed_at"] == b"2026-11-04T14:27:19+00:00"
        feats = OF.read_graph(fake_r, aid)
        assert feats["graph_component_size"] == 3.0
        assert feats["graph_dist_to_fraud"] == expected_dist
        assert feats["graph_risk_rank"] > 0.0


def test_run_job_does_not_close_caller_owned_resources_and_reports_partial_redis_failure(tmp_path: Path):
    db_file = tmp_path / "graph_partial.db"
    engine = create_engine(f"sqlite:///{db_file}")
    _create_sqlite_graph_schema(engine)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE graph_feature_snapshots (
                    account_id text NOT NULL,
                    snapshot_at timestamp NOT NULL,
                    pagerank float NOT NULL,
                    degree integer NOT NULL,
                    in_degree integer NOT NULL,
                    out_degree integer NOT NULL,
                    clustering_coefficient float NOT NULL,
                    features text NOT NULL,
                    created_at timestamp,
                    PRIMARY KEY (account_id, snapshot_at)
                );
                """
            )
        )
        conn.execute(
            text("INSERT INTO accounts (account_id, status) VALUES ('acc-1', 'active'), ('acc-2', 'active');")
        )

    caller_redis = fakeredis.FakeRedis()
    caller_redis_close = MagicMock()
    caller_redis.close = caller_redis_close  # type: ignore[method-assign]

    # Simulate chunk 1 succeeding and chunk 2 failing in Redis
    real_pipeline = caller_redis.pipeline
    call_idx = 0

    class FailingSecondPipeline:
        def __init__(self, real_pipe):
            self._real_pipe = real_pipe
            self._fail = call_idx >= 2

        def hset(self, *args, **kwargs):
            return self._real_pipe.hset(*args, **kwargs)

        def expire(self, *args, **kwargs):
            return self._real_pipe.expire(*args, **kwargs)

        def execute(self):
            if self._fail:
                raise redis_lib.ConnectionError("Simulated mid-batch Redis failure")
            return self._real_pipe.execute()

    def flaky_pipeline(*args, **kwargs):
        nonlocal call_idx
        call_idx += 1
        return FailingSecondPipeline(real_pipeline(*args, **kwargs))

    snap_ts = datetime(2026, 11, 4, 15, 22, 10, tzinfo=timezone.utc)
    with patch.object(caller_redis, "pipeline", side_effect=flaky_pipeline):
        summary = run_job(
            engine=engine,
            redis_client=caller_redis,
            snapshot_at=snap_ts,
            batch_size=1,
        )

    assert summary["status"] == "partial_failure"
    assert summary["publication_status"] == "partial_redis"
    assert summary["postgres_persisted"] is True
    assert summary["persisted_snapshots"] == 2
    assert summary["redis_status"] == "partial"
    assert summary["redis_published"] == 1
    assert summary["redis_failed"] == 1
    assert summary["retryable"] is True
    assert "ConnectionError" in summary["redis_error"]

    # Caller-owned Redis client & Engine must remain open
    caller_redis_close.assert_not_called()

    # Safe retry with same snapshot_at completes without duplicate DB rows
    retry_summary = run_job(
        engine=engine,
        redis_client=caller_redis,
        snapshot_at=snap_ts,
        batch_size=1,
    )
    assert retry_summary["status"] == "success"
    assert retry_summary["publication_status"] == "complete"
    assert retry_summary["persisted_snapshots"] == 2
    assert retry_summary["redis_published"] == 2
    assert retry_summary["redis_failed"] == 0

    with engine.connect() as conn:
        row_cnt = conn.execute(text("SELECT count(*) FROM graph_feature_snapshots;")).scalar()
        assert row_cnt == 2

    engine.dispose()


# ============================================================================
# 6. Account Status Semantics Tests
# ============================================================================


def test_fetch_active_account_ids_all_four_statuses():
    """Verify account status semantics: active, dormant, and frozen are included; closed is excluded."""
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE accounts (
                    account_id text PRIMARY KEY,
                    status text NOT NULL
                );
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO accounts (account_id, status) VALUES
                    ('acc-act-1', 'active'),
                    ('acc-dorm-2', 'dormant'),
                    ('acc-froz-3', 'frozen'),
                    ('acc-clos-4', 'closed');
                """
            )
        )

    active_ids = fetch_active_account_ids(engine, active_only=True)
    assert sorted(active_ids) == ["acc-act-1", "acc-dorm-2", "acc-froz-3"]
    assert "acc-clos-4" not in active_ids

    all_ids = fetch_active_account_ids(engine, active_only=False)
    assert sorted(all_ids) == ["acc-act-1", "acc-clos-4", "acc-dorm-2", "acc-froz-3"]
    engine.dispose()


def test_fetch_active_account_ids_without_status_column():
    """Verify that tables lacking a status column return all accounts safely."""
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE accounts (
                    account_id text PRIMARY KEY
                );
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO accounts (account_id) VALUES
                    ('acc-legacy-1'),
                    ('acc-legacy-2');
                """
            )
        )

    accs = fetch_active_account_ids(engine, active_only=True)
    assert sorted(accs) == ["acc-legacy-1", "acc-legacy-2"]
    engine.dispose()


def test_build_graph_collapses_distinct_edge_types_with_mappings_and_tuples():
    """Distinct edge_type values between identical endpoints collapse onto one undirected edge."""
    edge_mappings = [
        {
            "src_type": "account",
            "src_id": "acc-A",
            "dst_type": "account",
            "dst_id": "acc-B",
            "edge_type": "transfers_to",
            "weight": 2.5,
        },
        {
            "src_type": "account",
            "src_id": "acc-B",
            "dst_type": "account",
            "dst_id": "acc-A",
            "edge_type": "shares_card_with",
            "weight": 3.5,
        },
    ]
    G_map = build_graph_from_edges(edge_mappings)
    assert G_map.number_of_nodes() == 2
    assert G_map.number_of_edges() == 1
    assert G_map["acct:acc-A"]["acct:acc-B"]["weight"] == pytest.approx(6.0)

    edge_tuples_5 = [
        ("account", "acc-A", "account", "acc-B", 2.5),
        ("account", "acc-B", "account", "acc-A", 3.5),
    ]
    G_tup5 = build_graph_from_edges(edge_tuples_5)
    assert G_tup5.number_of_nodes() == 2
    assert G_tup5.number_of_edges() == 1
    assert G_tup5["acct:acc-A"]["acct:acc-B"]["weight"] == pytest.approx(6.0)

    edge_tuples_6 = [
        ("account", "acc-A", "account", "acc-B", 2.5, "transfers_to"),
        ("account", "acc-A", "account", "acc-B", 3.5, "co_signed"),
    ]
    G_tup6 = build_graph_from_edges(edge_tuples_6)
    assert G_tup6.number_of_nodes() == 2
    assert G_tup6.number_of_edges() == 1
    assert G_tup6["acct:acc-A"]["acct:acc-B"]["weight"] == pytest.approx(6.0)
