"""AegisFlow M2 — Hourly Graph Feature Job.

Extracts entity relationships from PostgreSQL `entity_edges` (with transaction-backed
historical edge-weight reconstruction via `transactions`) and temporally eligible
confirmed known-fraud accounts from `ground_truth_labels`, constructs an undirected
NetworkX entity graph (`acct:`, `dev:`, `ip:`, `mer:`), computes canonical undirected
graph features (`component_size`, `dist_to_fraud`, `risk_rank`, fraud-personalised
PageRank, `degree`, `clustering_coefficient`), persists snapshots to PostgreSQL
`graph_feature_snapshots`, and publishes pre-computed features to Redis hashes
`gfeat:acct:{account_id}` with TTL `GRAPH_TTL_S` (2 hours).

Entry point:
    python -m aegis_ml.features.graph_features --once
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import quote_plus
from uuid import UUID

from loguru import logger
import networkx as nx
import redis
from sqlalchemy import Engine, bindparam, create_engine, inspect, text

from aegis_ml.features import definitions as D


# ============================================================================
# Constants, Exceptions & Node Canonicalization
# ============================================================================

ENTITY_PREFIXES: dict[str, str] = {
    "account": "acct",
    "acct": "acct",
    "device": "dev",
    "dev": "dev",
    "ip": "ip",
    "merchant": "mer",
    "mer": "mer",
}

TXN_DERIVED_EDGE_TYPES: frozenset[str] = frozenset(
    {"uses_device", "uses_ip", "pays_merchant", "transfers_to"}
)

DEFAULT_FRAUD_CONFIDENCE_THRESHOLD: float = 0.85


class HistoricalEdgeReconstructionError(ValueError):
    """Raised when historical edge weights spanning snapshot_at cannot be reconstructed cleanly."""


def make_node_id(entity_type: str, entity_id: Any) -> str:
    """Format a prefixed canonical node identifier (`acct:`, `dev:`, `ip:`, `mer:`)."""
    if entity_type is None or not str(entity_type).strip():
        raise ValueError("entity_type must be a non-empty string.")
    if entity_id is None or not str(entity_id).strip():
        raise ValueError(f"entity_id for entity_type={entity_type!r} must be non-empty.")

    norm_type = str(entity_type).strip().lower()
    prefix = ENTITY_PREFIXES.get(norm_type, norm_type)
    norm_id = str(entity_id).strip()
    return f"{prefix}:{norm_id}"


# ============================================================================
# 1. Data Models, Configuration & Timestamp Normalization
# ============================================================================


@dataclass(frozen=True)
class GraphFeatureSnapshot:
    """Pre-computed graph features for an account at a specific point-in-time snapshot."""

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


def _load_dotenv() -> None:
    """Populate unset environment variables from repo-root .env if present."""
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


def normalize_db_url(url: str) -> str:
    """Ensure PostgreSQL URL uses psycopg3 driver if unspecified."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def resolve_database_url(db_url: str | None = None) -> str:
    """Resolve PostgreSQL connection URL from explicit arg, env, or .env without hardcoded secrets."""
    _load_dotenv()
    if db_url:
        return normalize_db_url(db_url)

    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return normalize_db_url(env_url)

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
        "or configure DB_USER/DB_PASS/DB_HOST/DB_PORT/DB_NAME via environment or .env."
    )


def resolve_redis_url(redis_url: str | None = None) -> str:
    """Resolve Redis connection URL from explicit arg, env, .env, or Docker host default."""
    _load_dotenv()
    if redis_url:
        return redis_url

    env_url = os.getenv("REDIS_URL")
    if env_url:
        return env_url

    host = os.getenv("REDIS_HOST", "localhost")
    port = os.getenv("REDIS_PORT", "6380")
    db = os.getenv("REDIS_DB", "0")
    return f"redis://{host}:{port}/{db}"


@contextmanager
def _managed_engine(engine: Engine | str) -> Iterator[Engine]:
    """Yield a SQLAlchemy Engine, disposing it afterwards only when created locally from a URL string."""
    if isinstance(engine, str):
        engine_obj = create_engine(normalize_db_url(engine), pool_pre_ping=True)
        try:
            yield engine_obj
        finally:
            engine_obj.dispose()
    else:
        yield engine


def hourly_bucket_for(dt: datetime) -> datetime:
    """Return the UTC top-of-hour bucket timestamp corresponding to dt."""
    utc_dt = normalize_snapshot_time(dt)
    return utc_dt.replace(minute=0, second=0, microsecond=0)


def normalize_snapshot_time(dt: datetime | str | None = None) -> datetime:
    """Normalize a snapshot timestamp to an exact timezone-aware UTC datetime.

    - If dt is None: returns exact current UTC timestamp (`datetime.now(timezone.utc)`).
    - If dt is a string: parses ISO-8601 format and converts to exact UTC.
    - If dt is a datetime: preserves exact instant in UTC (assumes UTC if naive).
    """
    if dt is None:
        return datetime.now(timezone.utc)
    if isinstance(dt, str):
        target = datetime.fromisoformat(dt)
    elif isinstance(dt, datetime):
        target = dt
    else:
        raise TypeError(f"Invalid timestamp type: {type(dt)!r}. Expected datetime, str, or None.")

    if target.tzinfo is None or target.tzinfo.utcoffset(target) is None:
        return target.replace(tzinfo=timezone.utc)
    return target.astimezone(timezone.utc)


# ============================================================================
# 2. Undirected Graph Construction & Feature Computation (Pure NetworkX)
# ============================================================================


def _validate_edge_weight(raw_weight: Any) -> float:
    """Validate and coerce an edge weight (`None` -> 1.0; finite and > 0 required)."""
    if raw_weight is None:
        return 1.0
    if isinstance(raw_weight, bool):
        raise ValueError(f"Invalid boolean edge weight: {raw_weight!r}")
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid edge weight {raw_weight!r}: {exc}") from exc

    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"Edge weight must be finite and > 0, got {weight!r}")
    return weight


def build_graph_from_edges(
    edges: Sequence[
        tuple[str, str, str, str]
        | tuple[str, str, str, str, Any]
        | tuple[str, str, str, str, Any, str]
        | Mapping[str, Any]
    ],
) -> nx.Graph:
    """Construct a simple undirected NetworkX graph from raw edge records.

    Node identifiers use canonical entity prefixes:
        `acct:{id}`, `dev:{id}`, `ip:{id}`, `mer:{id}`

    Multiple relationship observations or opposite directions between the same
    pair of entities are collapsed symmetrically onto one undirected edge whose
    weight equals `math.fsum(weights)` in sorted endpoint order, guaranteeing
    input-order and direction invariance.
    """
    grouped_weights: dict[tuple[str, str], list[float]] = {}

    for edge in edges:
        if isinstance(edge, Mapping):
            src_type = edge["src_type"]
            src_id = edge["src_id"]
            dst_type = edge["dst_type"]
            dst_id = edge["dst_id"]
            raw_weight = edge.get("weight", 1.0)
        else:
            if len(edge) < 4:
                raise ValueError(f"Edge tuple must have at least 4 elements, got {len(edge)}")
            src_type = edge[0]
            src_id = edge[1]
            dst_type = edge[2]
            dst_id = edge[3]
            raw_weight = edge[4] if len(edge) >= 5 else 1.0

        node_a = make_node_id(str(src_type), src_id)
        node_b = make_node_id(str(dst_type), dst_id)
        weight = _validate_edge_weight(raw_weight)

        u, v = (node_a, node_b) if node_a <= node_b else (node_b, node_a)
        grouped_weights.setdefault((u, v), []).append(weight)

    G = nx.Graph()
    for u, v in sorted(grouped_weights):
        total_weight = math.fsum(sorted(grouped_weights[(u, v)]))
        G.add_edge(u, v, weight=total_weight)

    return G


def _ensure_undirected_graph(G: nx.Graph) -> nx.Graph:
    """Return an undirected simple nx.Graph, summing reciprocal weights if given a DiGraph."""
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError("MultiGraph inputs are not supported; supply a simple nx.Graph.")
    if not G.is_directed():
        return G

    undir = nx.Graph()
    for node, attrs in G.nodes(data=True):
        undir.add_node(node, **attrs)
    grouped: dict[tuple[Any, Any], list[float]] = {}
    for u, v, data in G.edges(data=True):
        key = (u, v) if repr(u) <= repr(v) else (v, u)
        grouped.setdefault(key, []).append(_validate_edge_weight(data.get("weight", 1.0)))
    for (u, v), weights in grouped.items():
        undir.add_edge(u, v, weight=math.fsum(sorted(weights)))
    return undir


def _resolve_account_node(G: nx.Graph, account_id: str) -> Any:
    """Locate account node in G supporting prefixed strings (`acct:...`) and legacy tuples."""
    prefixed = make_node_id("account", account_id)
    if prefixed in G:
        return prefixed
    legacy = ("account", account_id)
    if legacy in G:
        return legacy
    return prefixed


def _compute_personalized_pagerank(
    G_undir: nx.Graph,
    fraud_nodes_in_graph: Sequence[Any],
    seeded_reachable_nodes: set[Any],
) -> dict[Any, float]:
    """Compute weighted personalised PageRank over components reachable from eligible fraud nodes.

    Returns `{}` (all accounts receive 0.0) when there are no eligible known-fraud seeds
    in the graph. Never falls back to global un-personalized PageRank.
    """
    unique_seeds = sorted(set(fraud_nodes_in_graph), key=repr)
    if not unique_seeds or not seeded_reachable_nodes:
        return {}

    G_seeded = G_undir.subgraph(sorted(seeded_reachable_nodes, key=repr))
    if len(G_seeded) == 0:
        return {}

    seed_set = set(unique_seeds)
    seed_mass = 1.0 / float(len(unique_seeds))
    personalization = {
        node: (seed_mass if node in seed_set else 0.0) for node in G_seeded.nodes()
    }
    total_mass = math.fsum(personalization.values())
    if total_mass <= 0.0:
        return {}
    if not math.isclose(total_mass, 1.0, rel_tol=0.0, abs_tol=1e-12):
        personalization = {n: v / total_mass for n, v in personalization.items()}

    try:
        return nx.pagerank(
            G_seeded,
            alpha=0.85,
            personalization=personalization,
            dangling=personalization,
            weight="weight",
            max_iter=200,
            tol=1e-6,
        )
    except nx.PowerIterationFailedConvergence:
        logger.warning(
            "SciPy PageRank hit max_iter=200; retrying with max_iter=1000 and nstart initialization"
        )
        return nx.pagerank(
            G_seeded,
            alpha=0.85,
            personalization=personalization,
            dangling=personalization,
            nstart=personalization,
            weight="weight",
            max_iter=1000,
            tol=1e-6,
            backend="networkx",
        )


def compute_graph_features(
    G: nx.Graph,
    account_ids: Sequence[str | UUID],
    fraud_account_ids: Sequence[str | UUID] | set[str | UUID] | None = None,
    snapshot_at: datetime | None = None,
) -> list[GraphFeatureSnapshot]:
    """Compute all canonical undirected graph features for the specified accounts.

    Algorithms (all computed over the undirected simple entity graph):
    - Undirected degree: `degree` (symmetric `in_degree == out_degree == degree`)
    - Local clustering: `nx.clustering(G_undir)`
    - Component size: `|C_undir|` containing the account (1.0 when isolated/absent)
    - Distance to fraud: unweighted multi-source shortest-path hop length from eligible
      known-fraud account nodes (`0.0` for eligible fraud accounts, `-1.0` when unreachable)
    - Risk rank / PageRank: weighted personalised PageRank seeded uniformly across eligible
      known-fraud account nodes present in `G` (`0.0` for all accounts when none exist or
      when an account is disconnected from all fraud seeds)
    """
    norm_snapshot_at = normalize_snapshot_time(snapshot_at)
    fraud_set: set[str] = {str(f) for f in (fraud_account_ids or set())}
    G_undir = _ensure_undirected_graph(G)

    # 1. Local clustering coefficient (undirected, unweighted topology)
    clustering_scores: dict[Any, float] = {}
    if len(G_undir) > 0:
        try:
            clustering_scores = nx.clustering(G_undir)
        except Exception as exc:
            logger.warning(f"Clustering computation failed ({exc}); falling back to zero")
            clustering_scores = {}

    # 2. Connected component sizes (undirected)
    component_sizes: dict[Any, float] = {}
    if len(G_undir) > 0:
        for comp in nx.connected_components(G_undir):
            comp_len = float(len(comp))
            for node in comp:
                component_sizes[node] = comp_len

    # 3. Multi-source BFS for unweighted hop distance to fraud
    dist_to_fraud_map: dict[Any, float] = {}
    fraud_nodes_in_graph: list[Any] = []
    for fid in sorted(fraud_set):
        node = _resolve_account_node(G_undir, fid)
        if node in G_undir:
            fraud_nodes_in_graph.append(node)

    if fraud_nodes_in_graph:
        queue: deque[Any] = deque(fraud_nodes_in_graph)
        for fn in fraud_nodes_in_graph:
            dist_to_fraud_map[fn] = 0.0

        while queue:
            curr = queue.popleft()
            curr_dist = dist_to_fraud_map[curr]
            for neighbor in G_undir.neighbors(curr):
                if neighbor not in dist_to_fraud_map:
                    dist_to_fraud_map[neighbor] = curr_dist + 1.0
                    queue.append(neighbor)

    # 4. Fraud-personalised PageRank over seeded components (0.0 if no fraud seeds in graph)
    pagerank_scores = _compute_personalized_pagerank(
        G_undir,
        fraud_nodes_in_graph=fraud_nodes_in_graph,
        seeded_reachable_nodes=set(dist_to_fraud_map.keys()),
    )

    # 5. Assemble snapshots for each target account
    results: list[GraphFeatureSnapshot] = []
    for raw_acc_id in account_ids:
        acc_str = str(raw_acc_id)
        acc_node = _resolve_account_node(G_undir, acc_str)
        in_g = acc_node in G_undir

        if in_g:
            deg = int(G_undir.degree(acc_node))
            in_deg = deg
            out_deg = deg
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

        if acc_str in fraud_set:
            d_fraud = 0.0
        elif in_g and acc_node in dist_to_fraud_map:
            d_fraud = float(dist_to_fraud_map[acc_node])
        else:
            d_fraud = -1.0

        pr_clean = max(0.0, pr) if math.isfinite(pr) else 0.0
        r_rank = pr_clean
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

    When active_only=True (default), 'closed' accounts are excluded while
    'active', 'dormant', and 'frozen' accounts remain included.
    """
    with _managed_engine(engine) as engine_obj:
        with engine_obj.connect() as conn:
            inspector = inspect(conn)
            cols = {c["name"] for c in inspector.get_columns("accounts")}
            if active_only and "status" in cols:
                stmt = text(
                    "SELECT CAST(account_id AS text) FROM accounts "
                    "WHERE status != 'closed' ORDER BY account_id;"
                )
            else:
                stmt = text("SELECT CAST(account_id AS text) FROM accounts ORDER BY account_id;")
            rows = conn.execute(stmt).scalars().all()
            return [str(r) for r in rows if r]


def _parse_db_timestamp(val: Any) -> datetime:
    """Coerce SQLite string or aware DB timestamp to timezone-aware UTC datetime."""
    if isinstance(val, datetime):
        return normalize_snapshot_time(val)
    if isinstance(val, str):
        return normalize_snapshot_time(val)
    raise TypeError(f"Unsupported timestamp value from database: {val!r}")


def _reconstruct_spanning_txn_weights(
    conn,
    spanning_edges: list[dict[str, Any]],
    snapshot_at: datetime,
) -> dict[tuple[str, str, str, str, str], float]:
    """Reconstruct exact as-of weights from `transactions` for edges with `last_seen_at > snapshot_at`.

    Only transaction-backed edges where `COUNT(all matching transactions) == entity_edges.weight`
    can be reconstructed without approximation. Raises `HistoricalEdgeReconstructionError`
    otherwise.
    """
    inspector = inspect(conn)
    if "transactions" not in set(inspector.get_table_names()):
        raise HistoricalEdgeReconstructionError(
            "Cannot reconstruct historical edge weights for edges spanning snapshot_at "
            "because the `transactions` table is missing."
        )

    txn_cols = {c["name"] for c in inspector.get_columns("transactions")}
    ts_col = "occurred_at" if "occurred_at" in txn_cols else "ts"
    is_pg = conn.dialect.name == "postgresql"

    non_txn_edges = [
        e for e in spanning_edges if e["edge_type"] not in TXN_DERIVED_EDGE_TYPES
    ]
    if non_txn_edges:
        sample = non_txn_edges[0]
        raise HistoricalEdgeReconstructionError(
            "Historical edge-weight reconstruction is unsupported for non-transaction edge "
            f"({sample['src_type']}:{sample['src_id']} -> {sample['dst_type']}:{sample['dst_id']} "
            f"[{sample['edge_type']}]) spanning snapshot_at={snapshot_at.isoformat()}."
        )

    edges_by_type: dict[str, list[dict[str, Any]]] = {}
    for edge in spanning_edges:
        edges_by_type.setdefault(edge["edge_type"], []).append(edge)

    type_specs: dict[str, tuple[str | None, str | None]] = {}
    if "device_id" in txn_cols:
        type_specs["uses_device"] = ("CAST(device_id AS text)", "device_id IS NOT NULL AND CAST(device_id AS text) <> ''")
    if "ip_id" in txn_cols:
        type_specs["uses_ip"] = ("CAST(ip_id AS text)", "ip_id IS NOT NULL")
    if "merchant_id" in txn_cols:
        type_specs["pays_merchant"] = ("CAST(merchant_id AS text)", "merchant_id IS NOT NULL")
    if "destination_account_id" in txn_cols:
        type_specs["transfers_to"] = (
            "CAST(destination_account_id AS text)",
            "channel = 'transfer' AND destination_account_id IS NOT NULL",
        )
    elif "counterparty_id" in txn_cols:
        type_specs["transfers_to"] = (
            "CAST(counterparty_id AS text)",
            "channel = 'transfer' AND counterparty_id IS NOT NULL",
        )

    reconstructed_counts: dict[tuple[str, str, str, str, str], tuple[int, int]] = {}

    for edge_type, group in edges_by_type.items():
        spec = type_specs.get(edge_type)
        if spec is None:
            raise HistoricalEdgeReconstructionError(
                f"Cannot reconstruct spanning edge_type={edge_type!r} without matching `transactions` column."
            )
        dst_expr, base_pred = spec

        unique_accounts = sorted({e["src_id"] for e in group})
        account_filter_params: dict[str, Any] = {"snapshot_at": snapshot_at}

        if is_pg:
            parsed_uuids: list[UUID] = []
            all_uuids = True
            for acc in unique_accounts:
                try:
                    parsed_uuids.append(UUID(acc))
                except ValueError:
                    all_uuids = False
                    break
            if all_uuids:
                acct_pred = "account_id IN :acc_ids"
                account_filter_params["acc_ids"] = parsed_uuids
            else:
                acct_pred = "CAST(account_id AS text) IN :acc_ids"
                account_filter_params["acc_ids"] = unique_accounts
        else:
            acct_pred = "CAST(account_id AS text) IN :acc_ids"
            account_filter_params["acc_ids"] = unique_accounts

        stmt = text(
            f"""
            SELECT CAST(account_id AS text) AS src_id,
                   {dst_expr} AS dst_id,
                   COUNT(*) AS total_cnt,
                   SUM(CASE WHEN {ts_col} <= :snapshot_at THEN 1 ELSE 0 END) AS asof_cnt
            FROM transactions
            WHERE {acct_pred}
              AND {base_pred}
            GROUP BY CAST(account_id AS text), {dst_expr};
            """
        ).bindparams(bindparam("acc_ids", expanding=True))

        rows = conn.execute(stmt, account_filter_params).fetchall()
        counts_by_pair: dict[tuple[str, str], tuple[int, int]] = {
            (str(r[0]), str(r[1])): (int(r[2] or 0), int(r[3] or 0)) for r in rows
        }

        for edge in group:
            key = (
                edge["src_type"],
                edge["src_id"],
                edge["dst_type"],
                edge["dst_id"],
                edge["edge_type"],
            )
            reconstructed_counts[key] = counts_by_pair.get(
                (edge["src_id"], edge["dst_id"]), (0, 0)
            )

    resolved_weights: dict[tuple[str, str, str, str, str], float] = {}
    for edge in spanning_edges:
        key = (
            edge["src_type"],
            edge["src_id"],
            edge["dst_type"],
            edge["dst_id"],
            edge["edge_type"],
        )
        total_cnt, asof_cnt = reconstructed_counts.get(key, (0, 0))
        stored_weight = float(edge["weight"])
        if not math.isclose(stored_weight, float(total_cnt), rel_tol=0.0, abs_tol=1e-9):
            raise HistoricalEdgeReconstructionError(
                f"Cannot reconstruct historical weight at {snapshot_at.isoformat()} for edge {key}: "
                f"entity_edges.weight={stored_weight} does not match total matching transaction count={total_cnt} "
                f"(last_seen_at={edge['last_seen_at']})."
            )
        if asof_cnt <= 0:
            raise HistoricalEdgeReconstructionError(
                f"Cannot reconstruct historical weight at {snapshot_at.isoformat()} for edge {key}: "
                f"first_seen_at={edge['first_seen_at']} <= snapshot_at but zero matching transactions exist on or before snapshot_at."
            )
        resolved_weights[key] = float(asof_cnt)

    return resolved_weights


def fetch_entity_edges(
    engine: Engine | str,
    snapshot_at: datetime | None = None,
) -> list[tuple[str, str, str, str, float]]:
    """Fetch entity edges on or before snapshot_at without future edge-weight leakage.

    - When `snapshot_at` is None: returns all current `entity_edges` rows.
    - When `snapshot_at` is specified:
      * Edges with `last_seen_at <= snapshot_at` use their stored `weight`.
      * Edges spanning `snapshot_at` (`first_seen_at <= snapshot_at < last_seen_at`)
        have their historical weight reconstructed from `transactions` (`ts <= snapshot_at`)
        when transaction-backed, or raise `HistoricalEdgeReconstructionError` if they
        contain non-reconstructible post-snapshot weight mutations.
    """
    norm_snapshot_at = normalize_snapshot_time(snapshot_at) if snapshot_at is not None else None

    with _managed_engine(engine) as engine_obj:
        with engine_obj.connect() as conn:
            stmt = text(
                """
                SELECT src_type, src_id, dst_type, dst_id, edge_type, weight, first_seen_at, last_seen_at
                FROM entity_edges
                WHERE (:snapshot_at IS NULL OR first_seen_at <= :snapshot_at)
                ORDER BY src_type, src_id, dst_type, dst_id, edge_type;
                """
            )
            rows = conn.execute(stmt, {"snapshot_at": norm_snapshot_at}).fetchall()
            if not rows:
                return []

            if norm_snapshot_at is None:
                return [
                    (str(r[0]), str(r[1]), str(r[2]), str(r[3]), float(r[5]))
                    for r in rows
                ]

            edge_records: list[dict[str, Any]] = []
            spanning_edges: list[dict[str, Any]] = []
            for r in rows:
                first_seen = _parse_db_timestamp(r[6])
                last_seen = _parse_db_timestamp(r[7])
                rec = {
                    "src_type": str(r[0]),
                    "src_id": str(r[1]),
                    "dst_type": str(r[2]),
                    "dst_id": str(r[3]),
                    "edge_type": str(r[4]),
                    "weight": float(r[5]),
                    "first_seen_at": first_seen,
                    "last_seen_at": last_seen,
                }
                edge_records.append(rec)
                if last_seen > norm_snapshot_at:
                    spanning_edges.append(rec)

            reconstructed: dict[tuple[str, str, str, str, str], float] = {}
            if spanning_edges:
                reconstructed = _reconstruct_spanning_txn_weights(
                    conn, spanning_edges, norm_snapshot_at
                )

            results: list[tuple[str, str, str, str, float]] = []
            for rec in edge_records:
                key = (
                    rec["src_type"],
                    rec["src_id"],
                    rec["dst_type"],
                    rec["dst_id"],
                    rec["edge_type"],
                )
                weight = reconstructed.get(key, rec["weight"])
                results.append((rec["src_type"], rec["src_id"], rec["dst_type"], rec["dst_id"], weight))
            return results


def fetch_fraud_account_ids(
    engine: Engine | str,
    snapshot_at: datetime | None = None,
    *,
    confidence_threshold: float = DEFAULT_FRAUD_CONFIDENCE_THRESHOLD,
) -> set[str]:
    """Fetch temporally eligible confirmed fraud account IDs as of `snapshot_at`.

    Eligibility rules (aligned with `ground_truth_labels` + `m2_0003` + `dataset_builder`):
    1. `is_fraud IS TRUE` and `confidence >= confidence_threshold` (default 0.85).
    2. Confirmed on or before `snapshot_at` (`labeled_at <= snapshot_at`).
    3. Bounded incident start must not be in the future (`incident_start_at IS NULL OR incident_start_at <= snapshot_at`).
    4. Transaction-backed labels (`txn_id IS NOT NULL`) resolve via `JOIN transactions`
       only when the underlying transaction occurred on or before `snapshot_at`
       (`occurred_at <= snapshot_at`) and within any label `[incident_start_at, incident_end_at]` bounds.
    """
    norm_snapshot_at = normalize_snapshot_time(snapshot_at) if snapshot_at is not None else None
    fraud_accounts: set[str] = set()

    with _managed_engine(engine) as engine_obj:
        with engine_obj.connect() as conn:
            inspector = inspect(conn)
            tables = set(inspector.get_table_names())
            if "ground_truth_labels" not in tables:
                return fraud_accounts

            cols_gtl = {c["name"] for c in inspector.get_columns("ground_truth_labels")}
            has_conf = "confidence" in cols_gtl
            has_target = "label_target" in cols_gtl
            has_inc_start = "incident_start_at" in cols_gtl
            has_inc_end = "incident_end_at" in cols_gtl

            base_preds = ["is_fraud IS TRUE", "(:snapshot_at IS NULL OR labeled_at <= :snapshot_at)"]
            if has_conf:
                base_preds.append("confidence >= :confidence_threshold")
            if has_inc_start:
                base_preds.append(
                    "(:snapshot_at IS NULL OR incident_start_at IS NULL OR incident_start_at <= :snapshot_at)"
                )

            acc_preds = list(base_preds) + ["account_id IS NOT NULL", "txn_id IS NULL"]
            if has_target:
                acc_preds.append("label_target = 'account'")

            stmt_direct = text(
                f"""
                SELECT DISTINCT CAST(account_id AS text) AS acc_id
                FROM ground_truth_labels
                WHERE {' AND '.join(acc_preds)};
                """
            )
            params: dict[str, Any] = {
                "snapshot_at": norm_snapshot_at,
                "confidence_threshold": confidence_threshold,
            }
            for r in conn.execute(stmt_direct, params).scalars().all():
                if r:
                    fraud_accounts.add(str(r))

            if "transactions" in tables:
                cols_txn = {c["name"] for c in inspector.get_columns("transactions")}
                pk_col = "txn_id" if "txn_id" in cols_txn else "id"
                ts_col = "occurred_at" if "occurred_at" in cols_txn else "ts"

                txn_preds = [
                    "gtl.is_fraud IS TRUE",
                    "gtl.account_id IS NULL",
                    "gtl.txn_id IS NOT NULL",
                    "t.account_id IS NOT NULL",
                    "(:snapshot_at IS NULL OR gtl.labeled_at <= :snapshot_at)",
                    f"(:snapshot_at IS NULL OR t.{ts_col} <= :snapshot_at)",
                ]
                if has_conf:
                    txn_preds.append("gtl.confidence >= :confidence_threshold")
                if has_target:
                    txn_preds.append("gtl.label_target = 'transaction'")
                if has_inc_start:
                    txn_preds.append(
                        "(:snapshot_at IS NULL OR gtl.incident_start_at IS NULL OR gtl.incident_start_at <= :snapshot_at)"
                    )
                    txn_preds.append(
                        f"(gtl.incident_start_at IS NULL OR t.{ts_col} >= gtl.incident_start_at)"
                    )
                if has_inc_end:
                    txn_preds.append(
                        f"(gtl.incident_end_at IS NULL OR t.{ts_col} <= gtl.incident_end_at)"
                    )

                stmt_txn = text(
                    f"""
                    SELECT DISTINCT CAST(t.account_id AS text) AS acc_id
                    FROM ground_truth_labels gtl
                    JOIN transactions t ON t.{pk_col} = gtl.txn_id
                    WHERE {' AND '.join(txn_preds)};
                    """
                )
                for r in conn.execute(stmt_txn, params).scalars().all():
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

    total_persisted = 0
    with _managed_engine(engine) as engine_obj:
        is_pg = engine_obj.dialect.name == "postgresql"
        features_expr = "CAST(:features AS jsonb)" if is_pg else ":features"
        now_expr = "NOW()" if is_pg else "CURRENT_TIMESTAMP"
        stmt = text(
            f"""
            INSERT INTO graph_feature_snapshots (
                account_id, snapshot_at, pagerank, degree, in_degree, out_degree,
                clustering_coefficient, features, created_at
            ) VALUES (
                :account_id, :snapshot_at, :pagerank, :degree, :in_degree, :out_degree,
                :clustering_coefficient, {features_expr}, {now_expr}
            )
            ON CONFLICT (account_id, snapshot_at)
            DO UPDATE SET
                pagerank = EXCLUDED.pagerank,
                degree = EXCLUDED.degree,
                in_degree = EXCLUDED.in_degree,
                out_degree = EXCLUDED.out_degree,
                clustering_coefficient = EXCLUDED.clustering_coefficient,
                features = EXCLUDED.features,
                created_at = {now_expr};
            """
        )

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
    *,
    progress_out: dict[str, int] | None = None,
) -> int:
    """Publish graph features to Redis hash keys 'gfeat:acct:{account_id}' with TTL.

    Key fields written:
        component_size -> %.17g str (D.GRAPH_FIELDS)
        dist_to_fraud  -> %.17g str (D.GRAPH_FIELDS)
        risk_rank      -> %.17g str (D.GRAPH_FIELDS)
        computed_at    -> ISO-8601 UTC timestamp

    If `progress_out` is provided, updates `progress_out['published']` and
    `progress_out['failed']` after each committed pipeline batch.
    """
    if progress_out is not None:
        progress_out["published"] = 0
        progress_out["failed"] = len(snapshots)

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
                    "computed_at": s.snapshot_at.isoformat(),
                },
            )
            pipe.expire(key, ttl_s)
        pipe.execute()
        total_published += len(chunk)
        if progress_out is not None:
            progress_out["published"] = total_published
            progress_out["failed"] = len(snapshots) - total_published

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
    redis_url: str | None = None,
) -> dict[str, Any]:
    """Execute a single graph feature snapshot cycle.

    Steps:
    1. Extract active accounts, temporally valid entity edges, and eligible fraud seeds
       at the unified `norm_snapshot_at` timestamp.
    2. Build undirected NetworkX entity graph and compute canonical features.
    3. Upsert durable snapshots into PostgreSQL `graph_feature_snapshots`.
    4. Publish feature hashes to Redis `gfeat:acct:{id}` with TTL `D.GRAPH_TTL_S` (unless
       `skip_redis=True`), reporting explicit partial-publication status if Redis fails.
    """
    t0 = time.perf_counter()
    executed_at = datetime.now(timezone.utc)
    norm_snapshot_at = executed_at if snapshot_at is None else normalize_snapshot_time(snapshot_at)
    bucket_at = hourly_bucket_for(norm_snapshot_at)
    logger.info(
        f"Starting graph feature job for snapshot_at={norm_snapshot_at.isoformat()} "
        f"(hourly_bucket={bucket_at.isoformat()})"
    )

    owns_engine = engine is None or isinstance(engine, str)
    if engine is None:
        engine_obj = create_engine(resolve_database_url(), pool_pre_ping=True)
    elif isinstance(engine, str):
        engine_obj = create_engine(normalize_db_url(engine), pool_pre_ping=True)
    else:
        engine_obj = engine

    owns_redis = not skip_redis and redis_client is None
    r_client = redis_client
    if not skip_redis and r_client is None:
        resolved_redis_url = resolve_redis_url(redis_url)
        r_client = redis.Redis.from_url(resolved_redis_url, decode_responses=False)

    try:
        # Phase A: Data Extraction (single consistent as-of timestamp)
        t_extract = time.perf_counter()
        account_ids = fetch_active_account_ids(engine_obj, active_only=active_only)
        edges = fetch_entity_edges(engine_obj, norm_snapshot_at)
        fraud_accounts = fetch_fraud_account_ids(engine_obj, norm_snapshot_at)
        duration_extract = time.perf_counter() - t_extract
        logger.info(
            f"Extracted {len(account_ids)} accounts, {len(edges)} edges, "
            f"{len(fraud_accounts)} fraud accounts in {duration_extract:.3f}s"
        )

        # Phase B: Undirected Graph Construction & Feature Computation
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

        # Phase C: Durable PostgreSQL Snapshot Persistence (Source of Truth)
        t_db = time.perf_counter()
        persisted_count = persist_graph_snapshots(
            engine=engine_obj,
            snapshots=snapshots,
            batch_size=batch_size,
        )
        duration_db = time.perf_counter() - t_db
        logger.info(f"Persisted {persisted_count} snapshots to PostgreSQL in {duration_db:.3f}s")

        # Phase D: Disposable Redis Online Feature Publishing (Non-atomic dual-write; observable partial status)
        published_count = 0
        failed_count = 0
        redis_error_msg: str | None = None
        if skip_redis:
            status = "success"
            publication_status = "skipped_redis"
            redis_status = "skipped"
        elif r_client is None:
            status = "partial_failure"
            publication_status = "postgres_only"
            redis_status = "failed"
            failed_count = len(snapshots)
            redis_error_msg = "Redis client unavailable"
        else:
            progress: dict[str, int] = {"published": 0, "failed": len(snapshots)}
            t_redis = time.perf_counter()
            try:
                if not snapshots and callable(getattr(r_client, "ping", None)):
                    r_client.ping()
                published_count = publish_graph_features_to_redis(
                    redis_client=r_client,
                    snapshots=snapshots,
                    ttl_s=D.GRAPH_TTL_S,
                    batch_size=batch_size,
                    progress_out=progress,
                )
                failed_count = 0
                status = "success"
                publication_status = "complete"
                redis_status = "published"
                duration_redis = time.perf_counter() - t_redis
                logger.info(f"Published {published_count} accounts to Redis in {duration_redis:.3f}s")
            except Exception as exc:
                published_count = progress["published"]
                failed_count = max(0, len(snapshots) - published_count)
                redis_error_msg = f"{type(exc).__name__}: {exc}"
                status = "partial_failure"
                publication_status = "partial_redis" if published_count > 0 else "postgres_only"
                redis_status = "partial" if published_count > 0 else "failed"
                logger.error(
                    f"Redis publication {redis_status} after PostgreSQL commit "
                    f"(persisted={persisted_count}, published={published_count}, failed={failed_count}): "
                    f"{redis_error_msg}"
                )

        total_duration = time.perf_counter() - t0
        summary: dict[str, Any] = {
            "status": status,
            "publication_status": publication_status,
            "postgres_persisted": True,
            "redis_status": redis_status,
            "executed_at": executed_at.isoformat(),
            "snapshot_at": norm_snapshot_at.isoformat(),
            "hourly_bucket": bucket_at.isoformat(),
            "num_accounts": len(account_ids),
            "num_edges": len(edges),
            "num_graph_nodes": G.number_of_nodes(),
            "num_graph_edges": G.number_of_edges(),
            "num_fraud_accounts": len(fraud_accounts),
            "persisted_snapshots": persisted_count,
            "redis_published": published_count,
            "redis_failed": failed_count,
            "redis_error": redis_error_msg,
            "retryable": status != "success",
            "duration_s": total_duration,
        }
        if status == "success":
            logger.info(f"Completed graph feature job successfully in {total_duration:.3f}s")
        else:
            logger.warning(
                f"Graph feature job finished with {status} ({publication_status}) in {total_duration:.3f}s"
            )
        return summary

    finally:
        if owns_redis and r_client is not None:
            close_fn = getattr(r_client, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as exc:
                    logger.warning(f"Error closing Redis client: {exc}")
        if owns_engine and isinstance(engine_obj, Engine):
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
        help="Optional ISO-8601 UTC timestamp for snapshot (default: exact current UTC time).",
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
            summary = run_job(
                engine=args.db_url,
                redis_client=None,
                snapshot_at=args.snapshot_at,
                skip_redis=args.skip_redis,
                batch_size=args.batch_size,
                active_only=active_only,
                redis_url=args.redis_url,
            )
            if summary.get("status") != "success":
                logger.error(
                    f"Graph feature job did not complete cleanly: "
                    f"status={summary.get('status')}, publication_status={summary.get('publication_status')}, "
                    f"redis_error={summary.get('redis_error')}"
                )
                return 1
            return 0
        except Exception as exc:
            logger.error(f"Graph feature job failed: {exc}")
            return 1

    stop_signal_received = False

    def handle_signal(sig: int, frame: Any) -> None:
        nonlocal stop_signal_received
        logger.info(f"Received termination signal ({sig}), shutting down daemon...")
        stop_signal_received = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info(f"Starting hourly graph feature job daemon (interval={args.interval_s}s)...")
    while not stop_signal_received:
        cycle_started = time.monotonic()
        try:
            summary = run_job(
                engine=args.db_url,
                redis_client=None,
                snapshot_at=None,
                skip_redis=args.skip_redis,
                batch_size=args.batch_size,
                active_only=active_only,
                redis_url=args.redis_url,
            )
            if summary.get("status") != "success":
                logger.error(
                    f"Scheduled graph feature pass degraded: "
                    f"status={summary.get('status')}, publication_status={summary.get('publication_status')}, "
                    f"redis_error={summary.get('redis_error')}"
                )
        except Exception as exc:
            logger.error(f"Error during scheduled graph feature cycle: {exc}")

        remaining = max(0.0, float(args.interval_s) - (time.monotonic() - cycle_started))
        sleep_until = time.monotonic() + remaining
        while time.monotonic() < sleep_until and not stop_signal_received:
            time.sleep(0.5)

    logger.info("Graph feature job daemon terminated gracefully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
