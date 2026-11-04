"""AegisFlow M2 — Hourly Graph Feature Job.

Extracts entity relationships from PostgreSQL `entity_edges` and known fraud from
`ground_truth_labels`, constructs an entity graph with NetworkX, computes canonical
graph features (`component_size`, `dist_to_fraud`, `risk_rank`, `pagerank`, `degree`,
`in_degree`, `out_degree`, `clustering_coefficient`), persists snapshots to
`graph_feature_snapshots`, and publishes pre-computed features to Redis hashes
`gfeat:acct:{account_id}` with TTL `GRAPH_TTL_S` (2 hours).

Entry point:
    python -m aegis_ml.features.graph_features --once
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
import signal
import sys
import time
from typing import Any, Mapping, Sequence
from uuid import UUID

from loguru import logger
import networkx as nx
import redis
from sqlalchemy import Engine, create_engine, inspect, text

from aegis_ml.features import definitions as D


# ============================================================================
# 1. Data Models & Timestamp Normalization
# ============================================================================


@dataclass(frozen=True)
class GraphFeatureSnapshot:
    """Pre-computed graph features for an account at a specific hourly snapshot."""

    account_id: str
    snapshot_at: datetime
    pagerank: float
    degree: int
    in_degree: int
    out_degree: int
    clustering_coefficient: float
    component_size: float
    dist_to_fraud: float
    risk_rank: float
    features: dict[str, float]


def normalize_snapshot_time(dt: datetime | str | None = None) -> datetime:
    """Normalize a snapshot time to UTC, truncated to the current/target hour.

    If dt is None, returns current UTC time truncated to the hour.
    If dt is a string, parses ISO-8601 format and normalizes to UTC hour.
    """
    if dt is None:
        target = datetime.now(timezone.utc)
    elif isinstance(dt, str):
        target = datetime.fromisoformat(dt)
    elif isinstance(dt, datetime):
        target = dt
    else:
        raise TypeError(f"Invalid timestamp type: {type(dt)!r}. Expected datetime, str, or None.")

    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    else:
        target = target.astimezone(timezone.utc)

    return target.replace(minute=0, second=0, microsecond=0)


def normalize_db_url(url: str) -> str:
    """Ensure PostgreSQL URL uses psycopg3 driver if unspecified."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


# ============================================================================
# 2. Graph Construction & Feature Computation (Pure NetworkX)
# ============================================================================


def build_graph_from_edges(
    edges: Sequence[tuple[str, str, str, str, float] | tuple[str, str, str, str, float, str] | Mapping[str, Any]],
) -> nx.DiGraph:
    """Construct a directed NetworkX graph from raw edge records.

    Node representation:
        (entity_type, str(entity_id)) — e.g. ("account", "uuid"), ("device", "uuid").

    Weight Aggregation Policy & Intentional Loss of Relationship-Type Distinctions:
        In PostgreSQL `entity_edges`, the unique constraint covers:
            (src_type, src_id, dst_type, dst_id, edge_type).
        Multiple database records can share identical source and destination endpoints
        if they represent distinct relationship types (e.g. 'transfers_to' and 'shares_card_with').

        The graph feature calculation layer builds a simple directed graph (nx.DiGraph)
        rather than a multigraph. This is an intentional design decision:
        1. NetworkX algorithms such as `clustering` coefficient are mathematically and
           computationally defined only on simple graphs (NetworkX raises
           NetworkXNotImplemented on multigraphs).
        2. Directed PageRank uses edge weights as random-walk transition probabilities;
           multiple interactions between identical entities represent higher connection
           affinity and flow capacity.
        3. Collapsing multiple relationship types into a single directed edge with cumulative
           weight:
               W(u, v) = sum(weight)
           preserves total relational connection strength.
        4. Intentional Trade-off: Individual relationship-type distinctions are intentionally
           collapsed in this topological projection in exchange for well-defined centrality,
           clustering, and component metrics.

    Stability of Aggregation Order:
        Floating-point addition in IEEE 754 arithmetic is commutative but non-associative.
        To ensure bit-exact, reproducible edge weights and PageRank scores across executions,
        `fetch_entity_edges()` orders query results deterministically via an explicit SQL
        ORDER BY clause:
            ORDER BY src_type, src_id, dst_type, dst_id, edge_type.
        The SQL ORDER BY clause guarantees result set ordering; index selection is an optimizer
        decision.
    """
    G = nx.DiGraph()
    for e in edges:
        if isinstance(e, Mapping):
            src_type = str(e["src_type"])
            src_id = str(e["src_id"])
            dst_type = str(e["dst_type"])
            dst_id = str(e["dst_id"])
            weight = float(e.get("weight", 1.0))
        else:
            src_type = str(e[0])
            src_id = str(e[1])
            dst_type = str(e[2])
            dst_id = str(e[3])
            weight = float(e[4])

        src_node = (src_type, src_id)
        dst_node = (dst_type, dst_id)

        # In case of duplicate edges across multiple queries or types, sum weights additively
        if G.has_edge(src_node, dst_node):
            G[src_node][dst_node]["weight"] += weight
        else:
            G.add_edge(src_node, dst_node, weight=weight)

    return G


def compute_graph_features(
    G: nx.DiGraph,
    account_ids: Sequence[str | UUID],
    fraud_account_ids: Sequence[str | UUID] | set[str | UUID] | None = None,
    snapshot_at: datetime | None = None,
) -> list[GraphFeatureSnapshot]:
    """Compute all canonical graph features for the specified accounts.

    Algorithms:
    - Directed PageRank: nx.pagerank(G, alpha=0.85, weight='weight')
    - Directed degrees: in_degree, out_degree, total degree (in + out)
    - Clustering: nx.clustering(G_undirected)
    - Component size: |C_undirected| containing the account
    - Distance to fraud: shortest hop path length from any known fraud node
      via multi-source BFS on the undirected entity graph.
    """
    norm_snapshot_at = normalize_snapshot_time(snapshot_at)
    fraud_set: set[str] = {str(f) for f in (fraud_account_ids or set())}

    # Undirected view for symmetric relationships (device sharing, component size, hop distance)
    G_undir = G.to_undirected(as_view=True)

    # 1. PageRank computation (directed)
    pagerank_scores: dict[tuple[str, str], float] = {}
    if len(G) > 0:
        try:
            pagerank_scores = nx.pagerank(
                G,
                alpha=0.85,
                weight="weight",
                max_iter=200,
                tol=1e-6,
            )
        except Exception as exc:
            logger.warning(f"PageRank computation failed ({exc}); falling back to zero scores")
            pagerank_scores = {}

    # 2. Local clustering coefficient (undirected)
    clustering_scores: dict[tuple[str, str], float] = {}
    if len(G_undir) > 0:
        try:
            clustering_scores = nx.clustering(G_undir)
        except Exception as exc:
            logger.warning(f"Clustering computation failed ({exc}); falling back to zero")
            clustering_scores = {}

    # 3. Connected component sizes (undirected)
    component_sizes: dict[tuple[str, str], float] = {}
    if len(G_undir) > 0:
        for comp in nx.connected_components(G_undir):
            comp_len = float(len(comp))
            for node in comp:
                component_sizes[node] = comp_len

    # 4. Multi-source BFS for distance to fraud (undirected hop distance)
    dist_to_fraud_map: dict[tuple[str, str], float] = {}
    fraud_nodes_in_graph = [
        ("account", fid) for fid in fraud_set if ("account", fid) in G_undir
    ]
    if fraud_nodes_in_graph:
        queue: deque[tuple[str, str]] = deque(fraud_nodes_in_graph)
        for fn in fraud_nodes_in_graph:
            dist_to_fraud_map[fn] = 0.0

        while queue:
            curr = queue.popleft()
            curr_dist = dist_to_fraud_map[curr]
            for neighbor in G_undir.neighbors(curr):
                if neighbor not in dist_to_fraud_map:
                    dist_to_fraud_map[neighbor] = curr_dist + 1.0
                    queue.append(neighbor)

    # 5. Assemble snapshots for each target account
    results: list[GraphFeatureSnapshot] = []
    for raw_acc_id in account_ids:
        acc_str = str(raw_acc_id)
        acc_node = ("account", acc_str)
        in_g = acc_node in G

        if in_g:
            deg = int(G.degree(acc_node))
            in_deg = int(G.in_degree(acc_node))
            out_deg = int(G.out_degree(acc_node))
            pr = float(pagerank_scores.get(acc_node, 0.0))
            cc = float(clustering_scores.get(acc_node, 0.0))
            comp_sz = float(component_sizes.get(acc_node, 1.0))
        else:
            deg = 0
            in_deg = 0
            out_deg = 0
            pr = 0.0
            cc = 0.0
            comp_sz = 1.0

        # Distance to fraud calculation
        if acc_str in fraud_set:
            d_fraud = 0.0
        elif in_g and acc_node in dist_to_fraud_map:
            d_fraud = float(dist_to_fraud_map[acc_node])
        else:
            d_fraud = -1.0

        # Canonical risk rank is PageRank
        r_rank = pr

        # Enforce domain check constraints and finite bounds
        pr_clean = max(0.0, pr) if math.isfinite(pr) else 0.0
        cc_clean = max(0.0, min(1.0, cc)) if math.isfinite(cc) else 0.0
        deg_clean = max(0, deg)
        in_deg_clean = max(0, in_deg)
        out_deg_clean = max(0, out_deg)

        features_dict = {
            "component_size": comp_sz,
            "dist_to_fraud": d_fraud,
            "risk_rank": r_rank,
        }

        results.append(
            GraphFeatureSnapshot(
                account_id=acc_str,
                snapshot_at=norm_snapshot_at,
                pagerank=pr_clean,
                degree=deg_clean,
                in_degree=in_deg_clean,
                out_degree=out_deg_clean,
                clustering_coefficient=cc_clean,
                component_size=comp_sz,
                dist_to_fraud=d_fraud,
                risk_rank=r_rank,
                features=features_dict,
            )
        )

    return results


# ============================================================================
# 3. Database Extraction & Persistence (PostgreSQL)
# ============================================================================


def fetch_active_account_ids(engine: Engine | str, active_only: bool = True) -> list[str]:
    """Fetch account UUID strings from accounts table.

    Status semantics:
    - In AegisFlow, account status domain is ('active', 'dormant', 'frozen', 'closed').
    - When active_only=True (default), closed accounts ('closed') are excluded, while
      'active', 'dormant', and 'frozen' accounts are included.
      Rationale:
      * 'dormant' accounts require graph features to evaluate reactivation risk (Rule R006).
      * 'frozen' accounts are high-risk nodes actively tracked by compliance and investigators.
      * Only legally terminated 'closed' accounts cannot process new transactions.
    - When active_only=False, all accounts are returned regardless of status.
    """
    engine_obj = create_engine(normalize_db_url(engine)) if isinstance(engine, str) else engine
    with engine_obj.connect() as conn:
        inspector = inspect(conn)
        cols = {c["name"] for c in inspector.get_columns("accounts")}
        if active_only and "status" in cols:
            stmt = text("SELECT CAST(account_id AS text) FROM accounts WHERE status != 'closed' ORDER BY account_id;")
        else:
            stmt = text("SELECT CAST(account_id AS text) FROM accounts ORDER BY account_id;")
        rows = conn.execute(stmt).scalars().all()
        return [str(r) for r in rows if r]


def fetch_entity_edges(
    engine: Engine | str,
    snapshot_at: datetime | None = None,
) -> list[tuple[str, str, str, str, float]]:
    """Fetch entity edges created on or before snapshot_at.

    Ordered deterministically via explicit SQL `ORDER BY src_type, src_id, dst_type, dst_id, edge_type`
    to guarantee consistent floating-point summation order when multiple relationship
    types exist between the same entity pairs. The SQL ORDER BY guarantees result ordering;
    index selection is an optimizer decision.

    Returns: list of (src_type, src_id, dst_type, dst_id, weight)
    """
    engine_obj = create_engine(normalize_db_url(engine)) if isinstance(engine, str) else engine
    with engine_obj.connect() as conn:
        stmt = text("""
            SELECT src_type, src_id, dst_type, dst_id, weight
            FROM entity_edges
            WHERE (:snapshot_at IS NULL OR first_seen_at <= :snapshot_at)
            ORDER BY src_type, src_id, dst_type, dst_id, edge_type;
        """)
        rows = conn.execute(stmt, {"snapshot_at": snapshot_at}).fetchall()
        return [
            (str(r[0]), str(r[1]), str(r[2]), str(r[3]), float(r[4]))
            for r in rows
        ]


def fetch_fraud_account_ids(
    engine: Engine | str,
    snapshot_at: datetime | None = None,
) -> set[str]:
    """Fetch distinct fraud account IDs from ground_truth_labels on or before snapshot_at."""
    fraud_accounts: set[str] = set()
    engine_obj = create_engine(normalize_db_url(engine)) if isinstance(engine, str) else engine
    with engine_obj.connect() as conn:
        # 1. Accounts explicitly attached to ground_truth_labels
        stmt_direct = text("""
            SELECT DISTINCT CAST(account_id AS text) AS acc_id
            FROM ground_truth_labels
            WHERE is_fraud IS TRUE
              AND account_id IS NOT NULL
              AND (:snapshot_at IS NULL OR labeled_at <= :snapshot_at);
        """)
        direct_rows = conn.execute(stmt_direct, {"snapshot_at": snapshot_at}).scalars().all()
        for r in direct_rows:
            if r:
                fraud_accounts.add(str(r))

        # 2. Orphan transaction labels: resolve account_id through transactions table
        stmt_orphan_txns = text("""
            SELECT DISTINCT gtl.txn_id
            FROM ground_truth_labels gtl
            WHERE gtl.is_fraud IS TRUE
              AND gtl.account_id IS NULL
              AND gtl.txn_id IS NOT NULL
              AND (:snapshot_at IS NULL OR gtl.labeled_at <= :snapshot_at);
        """)
        orphan_txn_ids = conn.execute(stmt_orphan_txns, {"snapshot_at": snapshot_at}).scalars().all()
        if orphan_txn_ids:
            inspector = inspect(conn)
            cols = {c["name"] for c in inspector.get_columns("transactions")}
            pk_col = "txn_id" if "txn_id" in cols else "id"
            stmt_resolve = text(f"""
                SELECT DISTINCT CAST(account_id AS text) AS acc_id
                FROM transactions
                WHERE {pk_col} = ANY(:txn_ids) AND account_id IS NOT NULL;
            """)
            resolved_rows = conn.execute(stmt_resolve, {"txn_ids": orphan_txn_ids}).scalars().all()
            for r in resolved_rows:
                if r:
                    fraud_accounts.add(str(r))

    return fraud_accounts


def persist_graph_snapshots(
    engine: Engine | str,
    snapshots: Sequence[GraphFeatureSnapshot],
    batch_size: int = 500,
) -> int:
    """Atomically upsert graph snapshots to graph_feature_snapshots in PostgreSQL.

    Idempotent on (account_id, snapshot_at).
    Returns count of persisted records.
    """
    if not snapshots:
        return 0

    engine_obj = create_engine(normalize_db_url(engine)) if isinstance(engine, str) else engine

    stmt = text("""
        INSERT INTO graph_feature_snapshots (
            account_id, snapshot_at, pagerank, degree, in_degree, out_degree,
            clustering_coefficient, features, created_at
        ) VALUES (
            :account_id, :snapshot_at, :pagerank, :degree, :in_degree, :out_degree,
            :clustering_coefficient, CAST(:features AS jsonb), NOW()
        )
        ON CONFLICT (account_id, snapshot_at)
        DO UPDATE SET
            pagerank = EXCLUDED.pagerank,
            degree = EXCLUDED.degree,
            in_degree = EXCLUDED.in_degree,
            out_degree = EXCLUDED.out_degree,
            clustering_coefficient = EXCLUDED.clustering_coefficient,
            features = EXCLUDED.features,
            created_at = NOW();
    """)

    total_persisted = 0
    with engine_obj.begin() as conn:
        for i in range(0, len(snapshots), batch_size):
            chunk = snapshots[i : i + batch_size]
            batch_params = [
                {
                    "account_id": s.account_id,
                    "snapshot_at": s.snapshot_at,
                    "pagerank": s.pagerank,
                    "degree": s.degree,
                    "in_degree": s.in_degree,
                    "out_degree": s.out_degree,
                    "clustering_coefficient": s.clustering_coefficient,
                    "features": json.dumps(s.features),
                }
                for s in chunk
            ]
            conn.execute(stmt, batch_params)
            total_persisted += len(chunk)

    return total_persisted


# ============================================================================
# 4. Redis Online Feature Publishing
# ============================================================================


def publish_graph_features_to_redis(
    redis_client: redis.Redis,
    snapshots: Sequence[GraphFeatureSnapshot],
    ttl_s: int = D.GRAPH_TTL_S,
    batch_size: int = 500,
) -> int:
    """Publish graph features to Redis hash keys 'gfeat:acct:{account_id}' with TTL.

    Key fields:
        component_size -> str
        dist_to_fraud  -> str
        risk_rank      -> str

    Returns count of published accounts.
    """
    if not snapshots:
        return 0

    total_published = 0
    for i in range(0, len(snapshots), batch_size):
        chunk = snapshots[i : i + batch_size]
        pipe = redis_client.pipeline(transaction=False)
        for s in chunk:
            key = f"gfeat:acct:{s.account_id}"
            pipe.hset(
                key,
                mapping={
                    "component_size": f"{s.component_size:.17g}",
                    "dist_to_fraud": f"{s.dist_to_fraud:.17g}",
                    "risk_rank": f"{s.risk_rank:.17g}",
                },
            )
            pipe.expire(key, ttl_s)
        pipe.execute()
        total_published += len(chunk)

    return total_published


# ============================================================================
# 5. Job Orchestration & Execution
# ============================================================================


def run_job(
    engine: Engine | str | None = None,
    redis_client: redis.Redis | None = None,
    snapshot_at: datetime | str | None = None,
    skip_redis: bool = False,
    batch_size: int = 500,
    active_only: bool = True,
) -> dict[str, Any]:
    """Execute a single hourly graph feature snapshot cycle.

    Steps:
    1. Extract active accounts, entity edges, and fraud labels up to snapshot_at.
    2. Build NetworkX directed graph and undirected view.
    3. Compute canonical graph features for all active accounts.
    4. Upsert snapshots into graph_feature_snapshots in PostgreSQL.
    5. Publish feature hashes to Redis with 2-hour TTL (unless skip_redis=True).

    Returns execution summary metrics dictionary.
    """
    t0 = time.perf_counter()
    norm_snapshot_at = normalize_snapshot_time(snapshot_at)
    logger.info(f"Starting graph feature job for snapshot_at={norm_snapshot_at.isoformat()}")

    # Setup database engine
    if engine is None:
        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://aegisflow:aegisflow@localhost:5432/aegisflow",
        )
        engine_obj = create_engine(normalize_db_url(db_url), pool_pre_ping=True)
    elif isinstance(engine, str):
        engine_obj = create_engine(normalize_db_url(engine), pool_pre_ping=True)
    else:
        engine_obj = engine

    # Setup Redis client
    r_client = redis_client
    if not skip_redis and r_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r_client = redis.Redis.from_url(redis_url, decode_responses=False)

    try:
        # Phase A: Data Extraction
        t_extract = time.perf_counter()
        account_ids = fetch_active_account_ids(engine_obj, active_only=active_only)
        edges = fetch_entity_edges(engine_obj, norm_snapshot_at)
        fraud_accounts = fetch_fraud_account_ids(engine_obj, norm_snapshot_at)
        duration_extract = time.perf_counter() - t_extract
        logger.info(
            f"Extracted {len(account_ids)} accounts, {len(edges)} edges, "
            f"{len(fraud_accounts)} fraud accounts in {duration_extract:.3f}s"
        )

        # Phase B: Graph Construction & Feature Computation
        t_comp = time.perf_counter()
        G = build_graph_from_edges(edges)
        snapshots = compute_graph_features(
            G=G,
            account_ids=account_ids,
            fraud_account_ids=fraud_accounts,
            snapshot_at=norm_snapshot_at,
        )
        duration_comp = time.perf_counter() - t_comp
        logger.info(
            f"Graph constructed ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges) "
            f"and {len(snapshots)} snapshots computed in {duration_comp:.3f}s"
        )

        # Phase C: PostgreSQL Snapshot Persistence
        t_db = time.perf_counter()
        persisted_count = persist_graph_snapshots(
            engine=engine_obj,
            snapshots=snapshots,
            batch_size=batch_size,
        )
        duration_db = time.perf_counter() - t_db
        logger.info(f"Persisted {persisted_count} snapshots to PostgreSQL in {duration_db:.3f}s")

        # Phase D: Redis Online Feature Publishing
        published_count = 0
        duration_redis = 0.0
        if not skip_redis and r_client is not None:
            t_redis = time.perf_counter()
            published_count = publish_graph_features_to_redis(
                redis_client=r_client,
                snapshots=snapshots,
                ttl_s=D.GRAPH_TTL_S,
                batch_size=batch_size,
            )
            duration_redis = time.perf_counter() - t_redis
            logger.info(f"Published {published_count} accounts to Redis in {duration_redis:.3f}s")

        total_duration = time.perf_counter() - t0
        summary = {
            "status": "success",
            "snapshot_at": norm_snapshot_at.isoformat(),
            "num_accounts": len(account_ids),
            "num_edges": len(edges),
            "num_graph_nodes": G.number_of_nodes(),
            "num_graph_edges": G.number_of_edges(),
            "num_fraud_accounts": len(fraud_accounts),
            "persisted_snapshots": persisted_count,
            "redis_published": published_count,
            "duration_s": total_duration,
        }
        logger.info(f"Completed graph feature job successfully in {total_duration:.3f}s")
        return summary

    finally:
        # Dispose engine if we created it locally
        if engine is None and isinstance(engine_obj, Engine):
            engine_obj.dispose()


# ============================================================================
# 6. CLI & Daemon Loop
# ============================================================================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for graph feature job."""
    parser = argparse.ArgumentParser(
        description="AegisFlow Hourly Graph Feature Job (M2 Step 2.9)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Execute single snapshot computation cycle and exit immediately.",
    )
    parser.add_argument(
        "--snapshot-at",
        type=str,
        default=None,
        help="Optional ISO-8601 UTC timestamp for snapshot (default: current UTC hour).",
    )
    parser.add_argument(
        "--interval-s",
        type=int,
        default=3600,
        help="Interval in seconds between hourly runs in daemon mode (default: 3600).",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="PostgreSQL connection URL override.",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default=None,
        help="Redis connection URL override.",
    )
    parser.add_argument(
        "--skip-redis",
        action="store_true",
        help="Skip publishing to Redis.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for database upserts and Redis pipeline writes (default: 500).",
    )
    parser.add_argument(
        "--include-closed",
        action="store_true",
        help="Include closed accounts in snapshot generation (default: False).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Hourly Graph Feature Job."""
    args = parse_args(argv)
    active_only = not args.include_closed

    if args.once:
        try:
            run_job(
                engine=args.db_url,
                redis_client=None,
                snapshot_at=args.snapshot_at,
                skip_redis=args.skip_redis,
                batch_size=args.batch_size,
                active_only=active_only,
            )
            return 0
        except Exception as exc:
            logger.error(f"Graph feature job failed: {exc}")
            return 1

    # Daemon mode: handle signals for clean termination
    stop_signal_received = False

    def handle_signal(sig: int, frame: Any) -> None:
        nonlocal stop_signal_received
        logger.info(f"Received termination signal ({sig}), shutting down daemon...")
        stop_signal_received = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info(f"Starting hourly graph feature job daemon (interval={args.interval_s}s)...")
    while not stop_signal_received:
        try:
            run_job(
                engine=args.db_url,
                redis_client=None,
                snapshot_at=None,
                skip_redis=args.skip_redis,
                batch_size=args.batch_size,
                active_only=active_only,
            )
        except Exception as exc:
            logger.error(f"Error during scheduled graph feature cycle: {exc}")

        # Sleep with sub-second polling to respond promptly to shutdown signals
        sleep_until = time.time() + args.interval_s
        while time.time() < sleep_until and not stop_signal_received:
            time.sleep(0.5)

    logger.info("Graph feature job daemon terminated gracefully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
