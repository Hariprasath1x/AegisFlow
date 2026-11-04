"""Unit tests for AegisFlow M2 Hourly Graph Feature Job (aegis_ml.features.graph_features).

Tests:
- Snapshot timestamp normalization
- Graph construction from raw edge tuples and mappings
- Weight consolidation on duplicate edges
- Feature calculations on empty graphs and isolated accounts
- Multi-hop fraud distance BFS with multiple fraud sources
- Clustering coefficient and component size verification
- Canonical feature dictionary formatting
- CLI argument parsing and error handling
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch
import uuid

import networkx as nx
import pytest
from sqlalchemy import create_engine, text

from aegis_ml.features import definitions as D
from aegis_ml.features.graph_features import (
    GraphFeatureSnapshot,
    build_graph_from_edges,
    compute_graph_features,
    fetch_active_account_ids,
    main,
    normalize_snapshot_time,
    parse_args,
)


# ============================================================================
# 1. Timestamp Normalization Tests
# ============================================================================


def test_normalize_snapshot_time_none_returns_current_utc_hour():
    t_norm = normalize_snapshot_time(None)
    assert t_norm.tzinfo == timezone.utc
    assert t_norm.minute == 0
    assert t_norm.second == 0
    assert t_norm.microsecond == 0


def test_normalize_snapshot_time_iso_string():
    raw_str = "2026-11-04T15:43:21+00:00"
    t_norm = normalize_snapshot_time(raw_str)
    assert t_norm == datetime(2026, 11, 4, 15, 0, 0, tzinfo=timezone.utc)


def test_normalize_snapshot_time_naive_assumed_utc():
    naive = datetime(2026, 11, 4, 18, 25, 30)
    t_norm = normalize_snapshot_time(naive)
    assert t_norm == datetime(2026, 11, 4, 18, 0, 0, tzinfo=timezone.utc)


def test_normalize_snapshot_time_invalid_type_raises():
    with pytest.raises(TypeError, match="Invalid timestamp type"):
        normalize_snapshot_time(123456789)  # type: ignore


# ============================================================================
# 2. Graph Construction Tests
# ============================================================================


def test_build_graph_from_edge_tuples_and_weights():
    edges = [
        ("account", "acc-1", "device", "dev-1", 1.5),
        ("account", "acc-1", "ip", "10.0.0.1", 1.0),
        ("account", "acc-2", "device", "dev-1", 2.0),
    ]
    G = build_graph_from_edges(edges)
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 3

    assert G.has_edge(("account", "acc-1"), ("device", "dev-1"))
    assert G[("account", "acc-1")][("device", "dev-1")]["weight"] == 1.5


def test_build_graph_from_edge_mappings():
    edges = [
        {"src_type": "account", "src_id": "acc-1", "dst_type": "merchant", "dst_id": "m-1", "weight": 2.5},
        {"src_type": "account", "src_id": "acc-2", "dst_type": "account", "dst_id": "acc-1", "weight": 1.0},
    ]
    G = build_graph_from_edges(edges)
    assert G.number_of_nodes() == 3
    assert G[("account", "acc-1")][("merchant", "m-1")]["weight"] == 2.5
    assert G[("account", "acc-2")][("account", "acc-1")]["weight"] == 1.0


def test_build_graph_consolidates_duplicate_edges():
    edges = [
        ("account", "acc-1", "device", "dev-1", 1.0),
        ("account", "acc-1", "device", "dev-1", 2.5),
    ]
    G = build_graph_from_edges(edges)
    assert G.number_of_edges() == 1
    assert G[("account", "acc-1")][("device", "dev-1")]["weight"] == 3.5


def test_build_graph_weight_aggregation_and_pagerank_parity():
    """Verify that multiple edges between same endpoints aggregate additively and PageRank uses the aggregated weight."""
    # acc-1 has two distinct relationships to acc-2 (weights 2.0 and 3.0, total 5.0)
    # acc-1 has one relationship to acc-3 (weight 5.0)
    edges = [
        ("account", "acc-1", "account", "acc-2", 2.0),
        ("account", "acc-1", "account", "acc-2", 3.0),
        ("account", "acc-1", "account", "acc-3", 5.0),
    ]
    G = build_graph_from_edges(edges)

    # 1. Total edge count is 2 (acc-1 -> acc-2 and acc-1 -> acc-3)
    assert G.number_of_edges() == 2

    # 2. Additive weight aggregation without silent overwriting
    assert G[("account", "acc-1")][("account", "acc-2")]["weight"] == 5.0
    assert G[("account", "acc-1")][("account", "acc-3")]["weight"] == 5.0

    # 3. PageRank treats both targets equally because both receive equal total weight
    pr = nx.pagerank(G, weight="weight")
    assert pr[("account", "acc-2")] == pytest.approx(pr[("account", "acc-3")])
    assert pr[("account", "acc-2")] > 0.35


def test_build_graph_empty_list():
    G = build_graph_from_edges([])
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


# ============================================================================
# 3. Graph Feature Calculation Tests
# ============================================================================


def test_empty_graph_and_accounts():
    G = nx.DiGraph()
    results = compute_graph_features(G, account_ids=[])
    assert results == []


def test_empty_graph_with_isolated_accounts_falls_back_to_defaults():
    G = nx.DiGraph()
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
    G = nx.DiGraph()
    acc_fraud = str(uuid.uuid4())
    acc_clean = str(uuid.uuid4())

    results = compute_graph_features(
        G,
        account_ids=[acc_fraud, acc_clean],
        fraud_account_ids=[acc_fraud],
    )

    r_fraud = next(r for r in results if r.account_id == acc_fraud)
    r_clean = next(r for r in results if r.account_id == acc_clean)

    # Isolated fraud account has dist_to_fraud=0.0 (distance to self is 0)
    assert r_fraud.dist_to_fraud == 0.0
    assert r_fraud.component_size == 1.0
    assert r_fraud.pagerank == 0.0

    # Clean account is unreachable from fraud
    assert r_clean.dist_to_fraud == -1.0


def test_synthetic_network_topology_and_metrics():
    """Verify exact degrees, multi-hop fraud distances, component sizes, and PageRank."""
    acc_fraud = "acc-fraud-1"
    acc_shared = "acc-shared-2"
    acc_transfer = "acc-transfer-3"
    acc_isolated = "acc-isolated-4"

    # Topology:
    # acc_fraud -> dev_1
    # acc_shared -> dev_1
    # acc_transfer -> acc_shared (transfer_to)
    # dev_1 connects acc_fraud and acc_shared (2 hops)
    # acc_transfer is 1 hop to acc_shared, 2 hops to dev_1, 3 hops to acc_fraud
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

    # 1. Distances to fraud
    assert by_acc[acc_fraud].dist_to_fraud == 0.0
    assert by_acc[acc_shared].dist_to_fraud == 2.0  # acc_shared -> dev_1 -> acc_fraud
    assert by_acc[acc_transfer].dist_to_fraud == 3.0  # acc_transfer -> acc_shared -> dev_1 -> acc_fraud
    assert by_acc[acc_isolated].dist_to_fraud == -1.0  # disconnected

    # 2. Connected Component Sizes
    # Component contains: acc_fraud, dev_1, acc_shared, acc_transfer (4 entities)
    assert by_acc[acc_fraud].component_size == 4.0
    assert by_acc[acc_shared].component_size == 4.0
    assert by_acc[acc_transfer].component_size == 4.0
    assert by_acc[acc_isolated].component_size == 1.0

    # 3. Directed Degrees
    # acc_transfer has 1 outgoing edge to acc_shared
    assert by_acc[acc_transfer].out_degree == 1
    assert by_acc[acc_transfer].in_degree == 0
    assert by_acc[acc_transfer].degree == 1

    # acc_shared has 1 incoming edge (from acc_transfer) and 1 outgoing edge (to dev_1)
    assert by_acc[acc_shared].in_degree == 1
    assert by_acc[acc_shared].out_degree == 1
    assert by_acc[acc_shared].degree == 2

    # acc_isolated has 0 degree
    assert by_acc[acc_isolated].degree == 0

    # 4. PageRank & Risk Rank
    for acc in [acc_fraud, acc_shared, acc_transfer]:
        assert by_acc[acc].pagerank > 0.0
        assert by_acc[acc].risk_rank == by_acc[acc].pagerank
    assert by_acc[acc_isolated].pagerank == 0.0
    assert by_acc[acc_isolated].risk_rank == 0.0


def test_triangle_clustering_coefficient():
    """Verify clustering coefficient on a 3-account triangle is exactly 1.0."""
    acc_a = "acc-tri-a"
    acc_b = "acc-tri-b"
    acc_c = "acc-tri-c"

    edges = [
        ("account", acc_a, "account", acc_b, 1.0),
        ("account", acc_b, "account", acc_c, 1.0),
        ("account", acc_c, "account", acc_a, 1.0),
    ]
    G = build_graph_from_edges(edges)
    results = compute_graph_features(G, account_ids=[acc_a, acc_b, acc_c])
    for r in results:
        assert r.clustering_coefficient == pytest.approx(1.0)
        assert r.component_size == 3.0


def test_multiple_fraud_accounts_selects_shortest_distance():
    """Multi-source BFS selects minimum hop distance when multiple fraud accounts exist."""
    f1 = "f-1"
    f2 = "f-2"
    target = "acc-target"

    # f1 is 1 hop from target (direct transfer)
    # f2 is 4 hops from target via devices
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
    """Verify that compute_graph_features strictly clamps within DB check constraint limits."""
    G = nx.DiGraph()
    # Add a node
    G.add_node(("account", "acc-clamp"))
    results = compute_graph_features(G, account_ids=["acc-clamp"])
    assert len(results) == 1
    r = results[0]
    assert 0.0 <= r.pagerank < float("inf")
    assert 0.0 <= r.clustering_coefficient <= 1.0
    assert r.degree >= 0
    assert r.in_degree >= 0
    assert r.out_degree >= 0


# ============================================================================
# 4. CLI Argument Parsing & Error Handling Tests
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
    args = parse_args([
        "--once",
        "--snapshot-at", "2026-11-04T12:00:00Z",
        "--interval-s", "1800",
        "--db-url", "postgresql://user:pass@db:5432/test",
        "--redis-url", "redis://redis:6379/2",
        "--skip-redis",
        "--batch-size", "100",
        "--include-closed",
    ])
    assert args.once is True
    assert args.snapshot_at == "2026-11-04T12:00:00Z"
    assert args.interval_s == 1800
    assert args.db_url == "postgresql://user:pass@db:5432/test"
    assert args.redis_url == "redis://redis:6379/2"
    assert args.skip_redis is True
    assert args.batch_size == 100
    assert args.include_closed is True


def test_cli_main_once_success():
    with patch("aegis_ml.features.graph_features.run_job", return_value={"status": "success"}) as mock_run:
        exit_code = main(["--once", "--skip-redis"])
        assert exit_code == 0
        mock_run.assert_called_once()


def test_cli_main_once_failure():
    with patch("aegis_ml.features.graph_features.run_job", side_effect=RuntimeError("Database failure")):
        exit_code = main(["--once"])
        assert exit_code == 1


# ============================================================================
# 5. Account Status Semantics Tests
# ============================================================================


def test_fetch_active_account_ids_all_four_statuses():
    """Verify account status semantics: active, dormant, and frozen are included; closed is excluded."""
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE accounts (
                account_id text PRIMARY KEY,
                status text NOT NULL
            );
        """))
        conn.execute(text("""
            INSERT INTO accounts (account_id, status) VALUES
                ('acc-act-1', 'active'),
                ('acc-dorm-2', 'dormant'),
                ('acc-froz-3', 'frozen'),
                ('acc-clos-4', 'closed');
        """))

    # When active_only=True (default), 'closed' is excluded; 'active', 'dormant', 'frozen' are returned
    active_ids = fetch_active_account_ids(engine, active_only=True)
    assert sorted(active_ids) == ["acc-act-1", "acc-dorm-2", "acc-froz-3"]
    assert "acc-clos-4" not in active_ids

    # When active_only=False, all four statuses are returned
    all_ids = fetch_active_account_ids(engine, active_only=False)
    assert sorted(all_ids) == ["acc-act-1", "acc-clos-4", "acc-dorm-2", "acc-froz-3"]


def test_fetch_active_account_ids_without_status_column():
    """Verify that tables lacking a status column return all accounts safely."""
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE accounts (
                account_id text PRIMARY KEY
            );
        """))
        conn.execute(text("""
            INSERT INTO accounts (account_id) VALUES
                ('acc-legacy-1'),
                ('acc-legacy-2');
        """))

    accs = fetch_active_account_ids(engine, active_only=True)
    assert sorted(accs) == ["acc-legacy-1", "acc-legacy-2"]


def test_build_graph_collapses_distinct_edge_types_with_mappings_and_tuples():
    """Verify that build_graph_from_edges collapses different edge_type values into a single directed edge.

    Demonstrates intentional loss of relationship-type distinctions in exchange for
    a well-defined scalar DiGraph for PageRank, degree, and clustering computation.
    """
    # 1. Using mapping format with explicit edge_type
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
            "src_id": "acc-A",
            "dst_type": "account",
            "dst_id": "acc-B",
            "edge_type": "shares_card_with",
            "weight": 3.5,
        },
    ]
    G_map = build_graph_from_edges(edge_mappings)
    assert G_map.number_of_nodes() == 2
    assert G_map.number_of_edges() == 1
    edge_data = G_map[("account", "acc-A")][("account", "acc-B")]
    assert edge_data["weight"] == pytest.approx(6.0)

    # 2. Using tuple format with 5 elements (standard extractor output)
    edge_tuples_5 = [
        ("account", "acc-A", "account", "acc-B", 2.5),
        ("account", "acc-A", "account", "acc-B", 3.5),
    ]
    G_tup5 = build_graph_from_edges(edge_tuples_5)
    assert G_tup5.number_of_nodes() == 2
    assert G_tup5.number_of_edges() == 1
    assert G_tup5[("account", "acc-A")][("account", "acc-B")]["weight"] == pytest.approx(6.0)

    # 3. Using tuple format with 6 elements (including edge_type)
    edge_tuples_6 = [
        ("account", "acc-A", "account", "acc-B", 2.5, "transfers_to"),
        ("account", "acc-A", "account", "acc-B", 3.5, "co_signed"),
    ]
    G_tup6 = build_graph_from_edges(edge_tuples_6)
    assert G_tup6.number_of_nodes() == 2
    assert G_tup6.number_of_edges() == 1
    assert G_tup6[("account", "acc-A")][("account", "acc-B")]["weight"] == pytest.approx(6.0)
