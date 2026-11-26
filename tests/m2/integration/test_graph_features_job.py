"""Integration tests for AegisFlow M2 Hourly Graph Feature Job.

Tests:
- PostgreSQL persistence to `graph_feature_snapshots`
- Check constraints and data type integrity
- Idempotent upsert on repeated runs with `ON CONFLICT (account_id, snapshot_at)`
- Redis hash publishing to `gfeat:acct:{account_id}` with TTL `GRAPH_TTL_S`
- Exact parity between Redis publish and `services.ingestion.online_features.read_graph`
- Offline pipeline compatibility via `aegis_ml.features.offline._join_graph_snapshots`
- End-to-end `run_job` execution with clean database teardown
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal
import json
import os
import re
import sys
from unittest.mock import patch
from urllib.parse import urlsplit
import uuid
import fakeredis
from loguru import logger
import pandas as pd
import psycopg
from psycopg.rows import dict_row
import pytest
import redis as redis_lib
from alembic import command
from alembic.config import Config
import sqlalchemy as sa
from sqlalchemy import text

from aegis_ml.features import definitions as D
from aegis_ml.features.graph_features import (
    GraphFeatureSnapshot,
    HistoricalEdgeReconstructionError,
    build_graph_from_edges,
    compute_graph_features,
    fetch_entity_edges,
    fetch_fraud_account_ids,
    normalize_db_url,
    persist_graph_snapshots,
    publish_graph_features_to_redis,
    run_job,
)
from aegis_ml.features.offline import _join_graph_snapshots
from ingestion import config as ingestion_config
import ingestion.online_features as OF
from ingestion.persistence import PgStore
from ingestion.schemas import TransactionIn

DATABASE_URL = os.getenv("DATABASE_URL", ingestion_config.DATABASE_URL)
ENGINE_URL = normalize_db_url(DATABASE_URL)


ADMIN_URL = os.getenv(
    "ADMIN_DATABASE_URL",
    ENGINE_URL.replace("postgresql+psycopg://", "postgresql://").rsplit("/", 1)[0] + "/postgres",
)
ALEMBIC_INI_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../alembic.ini")
)


_DISPOSABLE_DB_PATTERN = re.compile(r"^aegisflow_test_m2_0003_[0-9a-f]{8}$")


def _validate_disposable_db_name(name: str) -> None:
    """Explicit runtime guard ensuring target database is an isolated disposable test database.

    Validates complete expected format: 'aegisflow_test_m2_0003_' followed by exactly
    eight lowercase hexadecimal characters.
    Rejects malformed names, development database names, and any name outside this exact format.
    """
    if not isinstance(name, str) or not _DISPOSABLE_DB_PATTERN.fullmatch(name):
        raise RuntimeError(
            f"Safety check failed: database name {name!r} does not match expected disposable format 'aegisflow_test_m2_0003_[0-9a-f]{{8}}'"
        )
    dev_db_name = DATABASE_URL.rsplit("/", 1)[-1].split("?")[0]
    if name == dev_db_name or name == "aegisflow":
        raise RuntimeError(f"Safety check failed: database name cannot be development database {name!r}")


def _verify_admin_server_safety(dev_url: str, admin_url: str, admin_conn: psycopg.Connection) -> None:
    """Verify that administrative connection targets the exact same PostgreSQL server as DATABASE_URL.

    Checks:
    1. Host & port equivalence (normalizing loopback addresses).
    2. Administrative database is distinct from target development database (e.g. 'postgres' vs 'aegisflow').
    3. Server-side postmaster start time match (ensuring physical server instance identity).
    Fails safely with RuntimeError if verification cannot be established.
    """
    dev_parsed = urlsplit(dev_url.replace("postgresql+psycopg://", "postgresql://"))
    admin_parsed = urlsplit(admin_url.replace("postgresql+psycopg://", "postgresql://"))

    dev_host = (dev_parsed.hostname or "localhost").lower()
    admin_host = (admin_parsed.hostname or "localhost").lower()
    dev_port = dev_parsed.port or 5432
    admin_port = admin_parsed.port or 5432

    loopbacks = {"localhost", "127.0.0.1", "::1"}
    hosts_match = (dev_host in loopbacks and admin_host in loopbacks) or (dev_host == admin_host)
    if not hosts_match or dev_port != admin_port:
        raise RuntimeError(
            f"Safety check failed: ADMIN_DATABASE_URL targets {admin_host}:{admin_port}, "
            f"which does not match DATABASE_URL server {dev_host}:{dev_port}. "
            "Cross-server administrative operations are strictly prohibited."
        )

    dev_db = (dev_parsed.path or "").lstrip("/")
    admin_db = (admin_parsed.path or "").lstrip("/")
    if not admin_db or admin_db == dev_db or admin_db == "aegisflow":
        raise RuntimeError(
            f"Safety check failed: administrative database {admin_db!r} cannot be the target development database {dev_db!r}."
        )
    if admin_db not in ("postgres", "template1"):
        raise RuntimeError(
            f"Safety check failed: administrative database {admin_db!r} is not a recognized admin database ('postgres', 'template1')."
        )

    # Server-side cluster identity check
    try:
        with admin_conn.cursor() as cur:
            cur.execute("SELECT pg_postmaster_start_time();")
            admin_start_time = cur.fetchone()[0]
    except Exception as exc:
        raise RuntimeError(f"Safety check failed: unable to query postmaster start time from admin connection: {exc}")

    try:
        raw_dev_url = dev_url.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(raw_dev_url) as dev_conn:
            with dev_conn.cursor() as cur:
                cur.execute("SELECT pg_postmaster_start_time();")
                dev_start_time = cur.fetchone()[0]
    except Exception as exc:
        raise RuntimeError(f"Safety check failed: unable to probe DATABASE_URL server for cluster verification: {exc}")

    if admin_start_time != dev_start_time:
        raise RuntimeError(
            f"Safety check failed: admin connection cluster start time ({admin_start_time}) "
            f"does not match development database cluster start time ({dev_start_time}). "
            "Connections target distinct PostgreSQL server clusters."
        )


@pytest.fixture(scope="function")
def isolated_graph_test_db():
    """Create a completely isolated, disposable PostgreSQL database for graph feature testing.

    Bootstrapped with scripts/init_db.sql and upgraded to Alembic head (m2_0002).
    Completely isolated from the shared development database.
    Dropped in teardown using DROP DATABASE ... WITH (FORCE).
    Lifecycle guarantees cleanup on any failure after CREATE DATABASE.
    """
    try:
        admin_conn = psycopg.connect(ADMIN_URL, autocommit=True)
    except Exception as exc:
        pytest.skip(f"PostgreSQL admin database unavailable for isolated test database: {exc}")

    # Administrative connection safety verification
    try:
        _verify_admin_server_safety(DATABASE_URL, ADMIN_URL, admin_conn)
    except Exception:
        admin_conn.close()
        raise

    db_id = uuid.uuid4().hex[:8]
    disposable_db = f"aegisflow_test_m2_0003_{db_id}"

    # Explicit runtime guard immediately before CREATE DATABASE
    _validate_disposable_db_name(disposable_db)

    try:
        admin_conn.execute(
            psycopg.sql.SQL("CREATE DATABASE {};").format(
                psycopg.sql.Identifier(disposable_db)
            )
        )
    except Exception as exc:
        admin_conn.close()
        pytest.skip(f"Failed to create disposable database {disposable_db}: {exc}")

    test_conn = None
    try:
        test_db_url = f"{ENGINE_URL.replace('postgresql+psycopg://', 'postgresql://').rsplit('/', 1)[0]}/{disposable_db}"
        alembic_db_url = f"postgresql+psycopg://{test_db_url.split('://', 1)[1]}"

        # Bootstrap with clean M1 schema
        init_sql_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../scripts/init_db.sql")
        )
        with open(init_sql_path) as f:
            m1_sql = f.read()

        with psycopg.connect(test_db_url, autocommit=True) as init_conn:
            init_conn.execute(m1_sql)

        # Upgrade to head (m2_0002)
        cfg = Config(ALEMBIC_INI_PATH)
        cfg.set_main_option("sqlalchemy.url", alembic_db_url)
        command.upgrade(cfg, "head")

        test_conn = psycopg.connect(test_db_url, row_factory=dict_row)
        test_conn.autocommit = False

        class IsolatedDB:
            name = disposable_db
            url = test_db_url
            alembic_url = alembic_db_url
            conn = test_conn

        isolated = IsolatedDB()

        yield isolated

    finally:
        # Step 1: Close test connection first to terminate sessions before dropping database
        if test_conn is not None:
            try:
                test_conn.close()
            except Exception as conn_err:
                logger.warning(f"Error closing test connection to {disposable_db}: {conn_err}")

        primary_exc = sys.exception()

        # Step 2: Validate database name again and drop fixture-created database
        try:
            _validate_disposable_db_name(disposable_db)
            admin_conn.execute(
                psycopg.sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE);").format(
                    psycopg.sql.Identifier(disposable_db)
                )
            )
        except Exception as drop_exc:
            logger.error(f"Failed to drop disposable database {disposable_db}: {drop_exc}")
            if primary_exc is not None:
                raise RuntimeError(
                    f"Cleanup failed: unable to drop disposable database {disposable_db}: {drop_exc}"
                ) from primary_exc
            else:
                raise RuntimeError(
                    f"Cleanup failed: unable to drop disposable database {disposable_db}: {drop_exc}"
                ) from drop_exc
        finally:
            admin_conn.close()


@pytest.fixture
def fake_redis_client():
    """Provide a clean FakeRedis instance for testing."""
    client = fakeredis.FakeRedis()
    client.flushdb()
    yield client
    client.flushdb()


# ============================================================================
# 1. Database Persistence & Idempotency Tests (Isolated Test Database)
# ============================================================================


def test_persist_graph_snapshots_db(isolated_graph_test_db):
    """Verify that graph snapshots are correctly persisted with all column types and check constraints."""
    suffix = uuid.uuid4().hex[:8]
    cust_ref = f"CUST-GF-{suffix}"
    acc_num = f"ACC-GF-{suffix}"
    snap_time = datetime(2026, 11, 4, 14, 0, 0, tzinfo=timezone.utc)

    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute(
            "INSERT INTO customers (external_ref, full_name, country_code) VALUES (%s, 'Graph Test Customer', 'US') RETURNING customer_id;",
            (cust_ref,),
        )
        cust_id = cur.fetchone()["customer_id"]
        cur.execute(
            "INSERT INTO accounts (customer_id, account_number, account_type, currency, opened_on) VALUES (%s, %s, 'current', 'USD', '2026-01-01') RETURNING account_id;",
            (cust_id, acc_num),
        )
        acc_id = cur.fetchone()["account_id"]
        isolated_graph_test_db.conn.commit()

    acc_str = str(acc_id)
    snapshot = GraphFeatureSnapshot(
        account_id=acc_str,
        snapshot_at=snap_time,
        pagerank=0.085,
        degree=4,
        in_degree=2,
        out_degree=2,
        clustering_coefficient=0.33,
        component_size=6.0,
        dist_to_fraud=2.0,
        risk_rank=0.085,
        features={
            "component_size": 6.0,
            "dist_to_fraud": 2.0,
            "risk_rank": 0.085,
        },
    )

    count = persist_graph_snapshots(isolated_graph_test_db.alembic_url, [snapshot])
    assert count == 1

    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute(
            """
            SELECT account_id::text, snapshot_at, pagerank, degree, in_degree, out_degree,
                   clustering_coefficient, features
            FROM graph_feature_snapshots
            WHERE account_id = %s AND snapshot_at = %s;
            """,
            (acc_id, snap_time),
        )
        row = cur.fetchone()

        assert str(row["account_id"]) == acc_str
        assert row["snapshot_at"] == snap_time
        assert row["pagerank"] == pytest.approx(0.085)
        assert row["degree"] == 4
        assert row["in_degree"] == 2
        assert row["out_degree"] == 2
        assert row["clustering_coefficient"] == pytest.approx(0.33)
        assert row["features"]["component_size"] == 6.0
        assert row["features"]["dist_to_fraud"] == 2.0
        assert row["features"]["risk_rank"] == pytest.approx(0.085)


def test_persist_graph_snapshots_idempotency_and_update(isolated_graph_test_db):
    """Verify that repeating snapshot persistence for the same hour updates the row without duplication."""
    suffix = uuid.uuid4().hex[:8]
    cust_ref = f"CUST-IDEM-{suffix}"
    acc_num = f"ACC-IDEM-{suffix}"
    snap_time = datetime(2026, 11, 4, 16, 0, 0, tzinfo=timezone.utc)

    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute(
            "INSERT INTO customers (external_ref, full_name, country_code) VALUES (%s, 'Idempotency Test Customer', 'GB') RETURNING customer_id;",
            (cust_ref,),
        )
        cust_id = cur.fetchone()["customer_id"]
        cur.execute(
            "INSERT INTO accounts (customer_id, account_number, account_type, currency, opened_on) VALUES (%s, %s, 'savings', 'GBP', '2026-01-01') RETURNING account_id;",
            (cust_id, acc_num),
        )
        acc_id = cur.fetchone()["account_id"]
        isolated_graph_test_db.conn.commit()

    acc_str = str(acc_id)
    # Initial snapshot
    s1 = GraphFeatureSnapshot(
        account_id=acc_str,
        snapshot_at=snap_time,
        pagerank=0.10,
        degree=2,
        in_degree=1,
        out_degree=1,
        clustering_coefficient=0.0,
        component_size=3.0,
        dist_to_fraud=-1.0,
        risk_rank=0.10,
        features={"component_size": 3.0, "dist_to_fraud": -1.0, "risk_rank": 0.10},
    )
    persist_graph_snapshots(isolated_graph_test_db.alembic_url, [s1])

    # Updated snapshot for the EXACT SAME hour (e.g. recalculated with new edges)
    s2 = GraphFeatureSnapshot(
        account_id=acc_str,
        snapshot_at=snap_time,
        pagerank=0.25,
        degree=5,
        in_degree=3,
        out_degree=2,
        clustering_coefficient=0.45,
        component_size=8.0,
        dist_to_fraud=1.0,
        risk_rank=0.25,
        features={"component_size": 8.0, "dist_to_fraud": 1.0, "risk_rank": 0.25},
    )
    persist_graph_snapshots(isolated_graph_test_db.alembic_url, [s2])

    # Verify exactly 1 snapshot exists and values were updated
    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute(
            """
            SELECT pagerank, degree, in_degree, out_degree, clustering_coefficient, features
            FROM graph_feature_snapshots
            WHERE account_id = %s AND snapshot_at = %s;
            """,
            (acc_id, snap_time),
        )
        rows = cur.fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert row["pagerank"] == pytest.approx(0.25)
        assert row["degree"] == 5
        assert row["in_degree"] == 3
        assert row["out_degree"] == 2
        assert row["clustering_coefficient"] == pytest.approx(0.45)
        assert row["features"]["component_size"] == 8.0
        assert row["features"]["dist_to_fraud"] == 1.0


# ============================================================================
# 2. Redis Online Feature Publishing & Parity Tests
# ============================================================================


def test_publish_graph_features_to_redis_and_read_graph(fake_redis_client):
    """Verify that Redis hashes are published with correct key, fields, and TTL."""
    acc_id = str(uuid.uuid4())
    snap_time = datetime(2026, 11, 4, 18, 0, 0, tzinfo=timezone.utc)

    snapshot = GraphFeatureSnapshot(
        account_id=acc_id,
        snapshot_at=snap_time,
        pagerank=0.425,
        degree=3,
        in_degree=1,
        out_degree=2,
        clustering_coefficient=0.5,
        component_size=15.0,
        dist_to_fraud=2.0,
        risk_rank=0.425,
        features={"component_size": 15.0, "dist_to_fraud": 2.0, "risk_rank": 0.425},
    )

    published = publish_graph_features_to_redis(
        fake_redis_client,
        [snapshot],
        ttl_s=D.GRAPH_TTL_S,
    )
    assert published == 1

    # Verify key structure and TTL
    redis_key = f"gfeat:acct:{acc_id}"
    ttl = fake_redis_client.ttl(redis_key)
    assert 0 < ttl <= D.GRAPH_TTL_S

    # Verify raw hash fields
    raw = fake_redis_client.hgetall(redis_key)
    assert b"component_size" in raw or "component_size" in raw
    assert b"dist_to_fraud" in raw or "dist_to_fraud" in raw
    assert b"risk_rank" in raw or "risk_rank" in raw
    assert b"computed_at" in raw or "computed_at" in raw

    # Read back via authoritative online_features.read_graph
    features = OF.read_graph(fake_redis_client, acc_id)
    assert features["graph_component_size"] == pytest.approx(15.0)
    assert features["graph_dist_to_fraud"] == pytest.approx(2.0)
    assert features["graph_risk_rank"] == pytest.approx(0.425)


# ============================================================================
# 3. Offline Pipeline Point-in-Time Join Compatibility
# ============================================================================


def test_offline_join_with_computed_graph_snapshots():
    """Verify that compute_graph_features snapshots integrate seamlessly with offline point-in-time join."""
    acc_1 = "acc-join-1"
    acc_2 = "acc-join-2"

    t_snap_1 = datetime(2026, 11, 4, 10, 0, 0, tzinfo=timezone.utc)
    t_snap_2 = datetime(2026, 11, 4, 11, 0, 0, tzinfo=timezone.utc)

    # Hourly snapshot 1 (10:00)
    edges_1 = [
        ("account", acc_1, "device", "dev-1", 1.0),
        ("account", acc_2, "device", "dev-1", 1.0),
    ]
    G1 = build_graph_from_edges(edges_1)
    snaps_1 = compute_graph_features(G1, [acc_1, acc_2], fraud_account_ids=[acc_1], snapshot_at=t_snap_1)

    # Hourly snapshot 2 (11:00) with new fraud and edges
    edges_2 = edges_1 + [("account", "acc-new", "account", acc_2, 2.0)]
    G2 = build_graph_from_edges(edges_2)
    snaps_2 = compute_graph_features(G2, [acc_1, acc_2], fraud_account_ids=[acc_1], snapshot_at=t_snap_2)

    # Build DataFrame matching PostgreSQL table schema
    records = []
    for s in snaps_1 + snaps_2:
        records.append({
            "account_id": s.account_id,
            "snapshot_at": s.snapshot_at,
            "pagerank": s.pagerank,
            "degree": s.degree,
            "in_degree": s.in_degree,
            "out_degree": s.out_degree,
            "clustering_coefficient": s.clustering_coefficient,
            "features": s.features,
        })
    snapshots_df = pd.DataFrame(records)

    # Query transactions:
    # Txn 1 at 10:30 (should get snap_1)
    # Txn 2 at 11:15 (should get snap_2)
    txns = pd.DataFrame({
        "id": ["t1", "t2"],
        "account_id": [acc_2, acc_2],
        "ts": [
            datetime(2026, 11, 4, 10, 30, 0, tzinfo=timezone.utc),
            datetime(2026, 11, 4, 11, 15, 0, tzinfo=timezone.utc),
        ],
    })

    joined = _join_graph_snapshots(txns, snapshots_df)

    # At 10:30, acc_2 component size is 3.0 (acc_1, dev_1, acc_2)
    assert joined["graph_component_size"][0] == 3.0
    assert joined["graph_dist_to_fraud"][0] == 2.0

    # At 11:15, acc_2 component size is 4.0 (acc_1, dev_1, acc_2, acc-new)
    assert joined["graph_component_size"][1] == 4.0
    assert joined["graph_dist_to_fraud"][1] == 2.0


# ============================================================================
# 4. End-to-End run_job Execution
# ============================================================================


def test_end_to_end_run_job_with_database_and_fakeredis(isolated_graph_test_db, fake_redis_client):
    """Verify full run_job execution on isolated DB: DB extraction -> NetworkX -> DB snapshot -> Redis publish."""
    suffix = uuid.uuid4().hex[:8]
    cust_ref = f"CUST-E2E-{suffix}"
    acc_num = f"ACC-E2E-{suffix}"
    snap_time = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)

    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute(
            "INSERT INTO customers (external_ref, full_name, country_code) VALUES (%s, 'E2E Test Customer', 'DE') RETURNING customer_id;",
            (cust_ref,),
        )
        cust_id = cur.fetchone()["customer_id"]
        cur.execute(
            "INSERT INTO accounts (customer_id, account_number, account_type, currency, opened_on) VALUES (%s, %s, 'current', 'EUR', '2026-01-01') RETURNING account_id;",
            (cust_id, acc_num),
        )
        acc_id = cur.fetchone()["account_id"]

        # Insert entity edge for this account
        cur.execute(
            """
            INSERT INTO entity_edges (src_type, src_id, dst_type, dst_id, edge_type, weight, first_seen_at, last_seen_at)
            VALUES ('account', %s, 'device', 'e2e-dev-1', 'uses_device', 1.0, %s, %s)
            ON CONFLICT DO NOTHING;
            """,
            (str(acc_id), snap_time - timedelta(minutes=10), snap_time - timedelta(minutes=10)),
        )
        isolated_graph_test_db.conn.commit()

    # Execute run_job
    metrics = run_job(
        engine=isolated_graph_test_db.alembic_url,
        redis_client=fake_redis_client,
        snapshot_at=snap_time,
        skip_redis=False,
        batch_size=100,
    )

    assert metrics["status"] == "success"
    assert metrics["persisted_snapshots"] >= 1
    assert metrics["redis_published"] >= 1

    # Check DB snapshot exists
    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute(
            """
            SELECT pagerank, degree, features
            FROM graph_feature_snapshots
            WHERE account_id = %s AND snapshot_at = %s;
            """,
            (acc_id, snap_time),
        )
        snap = cur.fetchone()

        assert snap is not None
        assert snap["degree"] >= 1
        assert snap["features"]["component_size"] >= 2.0

    # Check Redis hash exists and has valid values
    redis_feats = OF.read_graph(fake_redis_client, str(acc_id))
    assert redis_feats["graph_component_size"] >= 2.0
    assert redis_feats["graph_dist_to_fraud"] == -1.0


def test_database_failure_during_batch_persistence_rolls_back(isolated_graph_test_db):
    """Verify that a batch persistence failure (e.g. foreign-key violation) rolls back atomically."""
    snap_time = datetime(2026, 11, 4, 13, 0, 0, tzinfo=timezone.utc)
    unknown_aid = str(uuid.uuid4())

    bad_snapshot = GraphFeatureSnapshot(
        account_id=unknown_aid,
        snapshot_at=snap_time,
        pagerank=0.1,
        degree=1,
        in_degree=1,
        out_degree=0,
        clustering_coefficient=0.0,
        component_size=2.0,
        dist_to_fraud=-1.0,
        risk_rank=0.1,
        features={"component_size": 2.0, "dist_to_fraud": -1.0, "risk_rank": 0.1},
    )

    with pytest.raises(Exception):
        persist_graph_snapshots(isolated_graph_test_db.alembic_url, [bad_snapshot])

    # Verify no orphan snapshot was persisted
    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM graph_feature_snapshots WHERE account_id = %s;",
            (unknown_aid,),
        )
        cnt = cur.fetchone()["count"]
        assert cnt == 0


def test_run_job_with_skip_redis(isolated_graph_test_db):
    """Verify that run_job with skip_redis=True completes successfully without requiring Redis."""
    snap_time = datetime(2026, 11, 4, 14, 0, 0, tzinfo=timezone.utc)
    metrics = run_job(
        engine=isolated_graph_test_db.alembic_url,
        redis_client=None,
        snapshot_at=snap_time,
        skip_redis=True,
    )
    assert metrics["status"] == "success"
    assert metrics["redis_published"] == 0


def test_distinct_edge_types_aggregation_in_db_and_graph(isolated_graph_test_db):
    """Verify that distinct database relationship types between the same endpoints are aggregated correctly.

    Verifies:
    1. Database level: PostgreSQL entity_edges table allows distinct edge_type records between
       the same (src_type, src_id, dst_type, dst_id) endpoints under uq_entity_edges_identity.
    2. Extraction level: fetch_entity_edges() preserves both rows.
    3. Graph construction: build_graph_from_edges() deliberately collapses the distinct edge types
       into a single directed edge in the simple DiGraph with additive weight W(u, v) = sum(w).
    4. Ordering stability: query uses deterministic ORDER BY to ensure bit-exact float summation.
    5. Feature computation: PageRank and degree metrics correctly reflect the collapsed cumulative weight.
    """
    snap_time = datetime(2026, 11, 4, 15, 0, 0, tzinfo=timezone.utc)
    t_edge = snap_time - timedelta(hours=1)

    with isolated_graph_test_db.conn.cursor() as cur:
        # Create customer and two accounts
        cur.execute(
            "INSERT INTO customers (external_ref, full_name, country_code) VALUES ('CUST-EDGETYPE', 'Edge Test Cust', 'US') RETURNING customer_id;"
        )
        cust_id = cur.fetchone()["customer_id"]
        cur.execute(
            """
            INSERT INTO accounts (customer_id, account_number, account_type, currency, opened_on, status)
            VALUES
                (%s, 'ACC-EDGE-SRC', 'current', 'USD', '2026-01-01', 'active'),
                (%s, 'ACC-EDGE-DST', 'savings', 'USD', '2026-01-01', 'active')
            RETURNING account_number, account_id;
            """,
            (cust_id, cust_id),
        )
        rows = cur.fetchall()
        acc_map = {r["account_number"]: str(r["account_id"]) for r in rows}
        acc_src = acc_map["ACC-EDGE-SRC"]
        acc_dst = acc_map["ACC-EDGE-DST"]

        # Insert TWO distinct relationship types between the exact same source and destination
        cur.execute(
            """
            INSERT INTO entity_edges (src_type, src_id, dst_type, dst_id, edge_type, weight, first_seen_at, last_seen_at)
            VALUES
                ('account', %s, 'account', %s, 'transfers_to', 2.5, %s, %s),
                ('account', %s, 'account', %s, 'shares_card_with', 3.5, %s, %s);
            """,
            (acc_src, acc_dst, t_edge, t_edge, acc_src, acc_dst, t_edge, t_edge),
        )
        isolated_graph_test_db.conn.commit()

    # Step 1: Extraction preserves BOTH database rows
    from aegis_ml.features.graph_features import fetch_entity_edges
    edges = fetch_entity_edges(isolated_graph_test_db.alembic_url, snap_time)
    assert len(edges) == 2, f"Expected 2 distinct edge records extracted, got {len(edges)}"

    # Confirm extracted tuples have expected endpoints and weights
    weights = [e[4] for e in edges if e[1] == acc_src and e[3] == acc_dst]
    assert sorted(weights) == [2.5, 3.5]

    # Step 2: Graph construction collapses distinct relationship types into 1 undirected edge with summed weight
    G = build_graph_from_edges(edges)
    src_node = f"acct:{acc_src}"
    dst_node = f"acct:{acc_dst}"

    assert not G.is_directed()
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1, "Graph must collapse distinct relationship types into 1 undirected edge"
    assert G.has_edge(src_node, dst_node)
    assert G.has_edge(dst_node, src_node)
    assert G[src_node][dst_node]["weight"] == pytest.approx(6.0), "Edge weight must be additive (2.5 + 3.5 = 6.0)"

    # Step 3: Feature computation operates seamlessly on the undirected collapsed graph
    snapshots = compute_graph_features(G, [acc_src, acc_dst], snapshot_at=snap_time)
    assert len(snapshots) == 2
    snap_src = next(s for s in snapshots if s.account_id == acc_src)
    snap_dst = next(s for s in snapshots if s.account_id == acc_dst)

    assert snap_src.degree == 1
    assert snap_src.in_degree == 1
    assert snap_src.out_degree == 1
    assert snap_src.risk_rank == 0.0

    assert snap_dst.degree == 1
    assert snap_dst.in_degree == 1
    assert snap_dst.out_degree == 1
    assert snap_dst.risk_rank == 0.0


# ============================================================================
# 5. Task 3: Non-Empty End-to-End Verification on Isolated Database
# ============================================================================


def test_task3_non_empty_snapshot_generation_isolated_db(isolated_graph_test_db, fake_redis_client):
    """Execute complete run_job flow against isolated database with realistic multi-account fixtures.

    Verifies all 8 acceptance criteria:
    1. Target active accounts are extracted (closed accounts excluded).
    2. Graph contains expected typed nodes and edges.
    3. Canonical graph features are computed.
    4. PostgreSQL snapshots are persisted.
    5. Feature JSONB contains all 3 canonical fields.
    6. Redis receives expected account hashes when enabled with GRAPH_TTL_S.
    7. Repeated execution remains idempotent.
    8. Accounts without graph relationships receive correct defaults.
    """
    snap_time = datetime(2026, 11, 4, 15, 0, 0, tzinfo=timezone.utc)
    t_edge = snap_time - timedelta(hours=1)

    # 1. Seed isolated test database with realistic minimal fixtures
    with isolated_graph_test_db.conn.cursor() as cur:
        # Customer
        cur.execute(
            "INSERT INTO customers (external_ref, full_name, country_code) VALUES ('CUST-T3-001', 'Task 3 Customer', 'US') RETURNING customer_id;"
        )
        cust_id = cur.fetchone()["customer_id"]

        # 4 active accounts + 1 closed account
        cur.execute(
            """
            INSERT INTO accounts (customer_id, account_number, account_type, currency, opened_on, status)
            VALUES
                (%s, 'ACC-T3-FRAUD', 'current', 'USD', '2026-01-01', 'active'),
                (%s, 'ACC-T3-MULE', 'current', 'USD', '2026-01-01', 'active'),
                (%s, 'ACC-T3-VICTIM', 'savings', 'USD', '2026-01-01', 'active'),
                (%s, 'ACC-T3-ISOLATED', 'savings', 'USD', '2026-01-01', 'active'),
                (%s, 'ACC-T3-CLOSED', 'current', 'USD', '2026-01-01', 'closed')
            RETURNING account_number, account_id;
            """,
            (cust_id, cust_id, cust_id, cust_id, cust_id),
        )
        acc_rows = cur.fetchall()
        acc_map = {r["account_number"]: str(r["account_id"]) for r in acc_rows}

        acc_fraud = acc_map["ACC-T3-FRAUD"]
        acc_mule = acc_map["ACC-T3-MULE"]
        acc_victim = acc_map["ACC-T3-VICTIM"]
        acc_isolated = acc_map["ACC-T3-ISOLATED"]
        acc_closed = acc_map["ACC-T3-CLOSED"]

        # Known fraud label for acc_fraud
        cur.execute(
            """
            INSERT INTO ground_truth_labels (account_id, label_target, is_fraud, fraud_type, confidence, source, labeled_at)
            VALUES (%s, 'account', true, 'synthetic_identity', 1.0, 'test_fixture', %s);
            """,
            (acc_fraud, t_edge),
        )

        # Entity relationships in entity_edges
        cur.execute(
            """
            INSERT INTO entity_edges (src_type, src_id, dst_type, dst_id, edge_type, weight, first_seen_at, last_seen_at)
            VALUES
                ('account', %s, 'device', 'dev-shared-1', 'uses_device', 1.0, %s, %s),
                ('account', %s, 'device', 'dev-shared-1', 'uses_device', 1.0, %s, %s),
                ('account', %s, 'account', %s, 'transfers_to', 2.0, %s, %s),
                ('account', %s, 'merchant', 'merch-amazon-1', 'pays_merchant', 1.5, %s, %s),
                ('account', %s, 'ip', '198.51.100.42', 'uses_ip', 1.0, %s, %s);
            """,
            (
                acc_fraud, t_edge, t_edge,
                acc_mule, t_edge, t_edge,
                acc_victim, acc_mule, t_edge, t_edge,
                acc_victim, t_edge, t_edge,
                acc_victim, t_edge, t_edge,
            ),
        )
        isolated_graph_test_db.conn.commit()

    # 2. Execute run_job() flow
    metrics = run_job(
        engine=isolated_graph_test_db.alembic_url,
        redis_client=fake_redis_client,
        snapshot_at=snap_time,
        skip_redis=False,
        active_only=True,
    )

    # Verification 1: All active target accounts extracted (4 active, 1 closed excluded)
    assert metrics["status"] == "success"
    assert metrics["num_accounts"] == 4
    assert metrics["num_edges"] == 5
    assert metrics["num_fraud_accounts"] == 1
    assert metrics["persisted_snapshots"] == 4
    assert metrics["redis_published"] == 4

    # Verification 2: Graph nodes & edges
    assert metrics["num_graph_nodes"] == 6  # 3 accounts in edges + 1 device + 1 merchant + 1 ip
    assert metrics["num_graph_edges"] == 5

    # Verification 3, 4, 5: Canonical features in PostgreSQL snapshots
    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute(
            """
            SELECT account_id::text, pagerank, degree, in_degree, out_degree,
                   clustering_coefficient, features
            FROM graph_feature_snapshots
            WHERE snapshot_at = %s;
            """,
            (snap_time,),
        )
        snaps = {r["account_id"]: r for r in cur.fetchall()}
        assert len(snaps) == 4
        assert acc_closed not in snaps  # closed account excluded

        # Check acc_fraud
        sf = snaps[acc_fraud]
        assert sf["features"]["dist_to_fraud"] == 0.0
        assert sf["features"]["component_size"] == 6.0
        assert sf["pagerank"] > 0.0
        assert sf["features"]["risk_rank"] == sf["pagerank"]
        assert sf["degree"] == 1
        assert sf["out_degree"] == 1
        assert sf["in_degree"] == 1

        # Check acc_mule (shares device with fraud -> dist = 2.0)
        sm = snaps[acc_mule]
        assert sm["features"]["dist_to_fraud"] == 2.0
        assert sm["features"]["component_size"] == 6.0
        assert sm["pagerank"] > 0.0
        assert sm["degree"] == 2
        assert sm["in_degree"] == 2
        assert sm["out_degree"] == 2

        # Check acc_victim (transfers to mule -> dist = 3.0)
        sv = snaps[acc_victim]
        assert sv["features"]["dist_to_fraud"] == 3.0
        assert sv["features"]["component_size"] == 6.0
        assert sv["pagerank"] > 0.0
        assert sv["degree"] == 3  # transfers_to, pays_merchant, and uses_ip
        assert sv["out_degree"] == 3
        assert sv["in_degree"] == 3

        # Verification 8: Isolated account receives canonical defaults
        si = snaps[acc_isolated]
        assert si["features"]["dist_to_fraud"] == -1.0
        assert si["features"]["component_size"] == 1.0
        assert si["features"]["risk_rank"] == 0.0
        assert si["pagerank"] == 0.0
        assert si["degree"] == 0
        assert si["in_degree"] == 0
        assert si["out_degree"] == 0
        assert si["clustering_coefficient"] == 0.0

    # Verification 6: Redis online feature store hashes and TTL
    for aid in (acc_fraud, acc_mule, acc_victim, acc_isolated):
        redis_feats = OF.read_graph(fake_redis_client, aid)
        assert "graph_component_size" in redis_feats
        assert "graph_dist_to_fraud" in redis_feats
        assert "graph_risk_rank" in redis_feats

        # Check TTL
        ttl = fake_redis_client.ttl(f"gfeat:acct:{aid}")
        assert 0 < ttl <= D.GRAPH_TTL_S

    # Check exact Redis values
    rf_fraud = OF.read_graph(fake_redis_client, acc_fraud)
    assert rf_fraud["graph_dist_to_fraud"] == 0.0
    assert rf_fraud["graph_component_size"] == 6.0

    rf_mule = OF.read_graph(fake_redis_client, acc_mule)
    assert rf_mule["graph_dist_to_fraud"] == 2.0
    assert rf_mule["graph_component_size"] == 6.0

    rf_victim = OF.read_graph(fake_redis_client, acc_victim)
    assert rf_victim["graph_dist_to_fraud"] == 3.0
    assert rf_victim["graph_component_size"] == 6.0

    rf_iso = OF.read_graph(fake_redis_client, acc_isolated)
    assert rf_iso["graph_dist_to_fraud"] == -1.0
    assert rf_iso["graph_component_size"] == 1.0
    assert rf_iso["graph_risk_rank"] == 0.0

    # Verification 7: Repeated execution remains idempotent
    metrics_repeat = run_job(
        engine=isolated_graph_test_db.alembic_url,
        redis_client=fake_redis_client,
        snapshot_at=snap_time,
        skip_redis=False,
        active_only=True,
    )
    assert metrics_repeat["status"] == "success"
    assert metrics_repeat["persisted_snapshots"] == 4

    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM graph_feature_snapshots WHERE snapshot_at = %s;",
            (snap_time,),
        )
        total_after_repeat = cur.fetchone()["count"]
        assert total_after_repeat == 4  # Still exactly 4, zero duplicates


# ============================================================================
# 6. Safety Guard & Validation Tests
# ============================================================================


def test_validate_disposable_db_name_format_validation():
    """Verify that _validate_disposable_db_name strictly requires 'aegisflow_test_m2_0003_' + 8 lowercase hex chars."""
    # Valid names
    _validate_disposable_db_name("aegisflow_test_m2_0003_1a2b3c4d")
    _validate_disposable_db_name("aegisflow_test_m2_0003_00000000")
    _validate_disposable_db_name("aegisflow_test_m2_0003_ffffffff")

    # Invalid names
    invalid_names = [
        "aegisflow",                                # Production / dev database
        "aegisflow_test_m2_0003_1A2B3C4D",          # Uppercase hex rejected
        "aegisflow_test_m2_0003_1a2b3c4",           # 7 chars (too short)
        "aegisflow_test_m2_0003_1a2b3c4de",         # 9 chars (too long)
        "aegisflow_test_m2_0003_1a2b3c4g",          # Non-hex character 'g'
        "aegisflow_test_m2_0002_1a2b3c4d",          # Wrong milestone/step prefix
        "aegisflow_test_m2_0003_",                  # Suffix missing
        "",                                         # Empty string
        "postgres",                                 # System database
    ]
    for bad_name in invalid_names:
        with pytest.raises(RuntimeError, match="Safety check failed"):
            _validate_disposable_db_name(bad_name)


def test_admin_database_safety_guards():
    """Verify that _verify_admin_server_safety prevents cross-server operations and targets an admin DB."""
    from unittest.mock import MagicMock

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = ["2026-11-04 00:00:00+00"]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    # 1. Host mismatch rejected
    with pytest.raises(RuntimeError, match="does not match DATABASE_URL server"):
        _verify_admin_server_safety(
            "postgresql://user:pass@localhost:5432/aegisflow",
            "postgresql://user:pass@remotehost:5432/postgres",
            mock_conn,
        )

    # 2. Port mismatch rejected
    with pytest.raises(RuntimeError, match="does not match DATABASE_URL server"):
        _verify_admin_server_safety(
            "postgresql://user:pass@localhost:5432/aegisflow",
            "postgresql://user:pass@localhost:5433/postgres",
            mock_conn,
        )

    # 3. Admin DB targeting dev DB 'aegisflow' rejected
    with pytest.raises(RuntimeError, match="cannot be the target development database"):
        _verify_admin_server_safety(
            "postgresql://user:pass@localhost:5432/aegisflow",
            "postgresql://user:pass@localhost:5432/aegisflow",
            mock_conn,
        )

    # 4. Admin DB targeting non-admin DB rejected
    with pytest.raises(RuntimeError, match="not a recognized admin database"):
        _verify_admin_server_safety(
            "postgresql://user:pass@localhost:5432/aegisflow",
            "postgresql://user:pass@localhost:5432/somedb",
            mock_conn,
        )


# ============================================================================
# 7. Step 2.9 Temporal, Undirected & Publication Consistency Regression Tests
# ============================================================================


def test_historical_edge_weight_reconstruction_no_future_leakage_isolated_db(
    isolated_graph_test_db, fake_redis_client
):
    """Build T1 snapshot, add post-T1 PgStore edge contributions, recompute T1 -> unchanged;
    later snapshot T2 sees new weight; unbacked non-transaction spanning edge fails closed."""
    t_before = datetime(2026, 11, 4, 10, 15, 0, tzinfo=timezone.utc)
    t_snap_1 = datetime(2026, 11, 4, 11, 0, 0, tzinfo=timezone.utc)
    t_after = datetime(2026, 11, 4, 11, 20, 0, tzinfo=timezone.utc)
    t_snap_2 = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)

    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute(
            "INSERT INTO customers (external_ref, full_name, country_code) VALUES ('CUST-HIST-EDGE', 'Hist Edge Cust', 'IN') RETURNING customer_id;"
        )
        cust_id = cur.fetchone()["customer_id"]
        cur.execute(
            """
            INSERT INTO accounts (customer_id, account_number, account_type, currency, opened_on, status)
            VALUES
                (%s, 'ACC-HE-FRAUD', 'current', 'INR', '2026-01-01', 'active'),
                (%s, 'ACC-HE-B', 'savings', 'INR', '2026-01-01', 'active'),
                (%s, 'ACC-HE-C', 'savings', 'INR', '2026-01-01', 'active')
            RETURNING account_number, account_id;
            """,
            (cust_id, cust_id, cust_id),
        )
        acc_map = {r["account_number"]: str(r["account_id"]) for r in cur.fetchall()}
        acc_fraud = acc_map["ACC-HE-FRAUD"]
        acc_b = acc_map["ACC-HE-B"]
        acc_c = acc_map["ACC-HE-C"]

        cur.execute(
            """
            INSERT INTO ground_truth_labels (account_id, label_target, is_fraud, fraud_type, confidence, source, labeled_at)
            VALUES (%s, 'account', true, 'money_mule', 1.0, 'test_fixture', %s);
            """,
            (acc_fraud, t_before),
        )
        isolated_graph_test_db.conn.commit()

    store = PgStore(isolated_graph_test_db.alembic_url)
    shared_dev = uuid.uuid4()

    # T0 (before T1): each account has weight 1.0 on shared_dev
    for aid in (acc_fraud, acc_b, acc_c):
        store.store(
            TransactionIn(
                id=uuid.uuid4(),
                ts=t_before,
                account_id=uuid.UUID(aid),
                device_id=shared_dev,
                amount=Decimal("100.00"),
                currency="INR",
                channel="card",
            )
        )

    # Initial historical snapshot at T1
    run_job(
        engine=isolated_graph_test_db.alembic_url,
        redis_client=fake_redis_client,
        snapshot_at=t_snap_1,
        skip_redis=True,
    )
    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute(
            "SELECT account_id::text, pagerank, degree, features FROM graph_feature_snapshots WHERE snapshot_at = %s;",
            (t_snap_1,),
        )
        initial_t1 = {r["account_id"]: r for r in cur.fetchall()}

    # Add 5 post-T1 transactions linking acc_b <-> shared_dev (entity_edges.weight jumps 1.0 -> 6.0)
    for idx in range(5):
        store.store(
            TransactionIn(
                id=uuid.uuid4(),
                ts=t_after + timedelta(minutes=idx),
                account_id=uuid.UUID(acc_b),
                device_id=shared_dev,
                amount=Decimal("250.00"),
                currency="INR",
                channel="card",
            )
        )

    # Recompute T1: reconstructed weight must stay 1.0 so features are identical
    run_job(
        engine=isolated_graph_test_db.alembic_url,
        redis_client=fake_redis_client,
        snapshot_at=t_snap_1,
        skip_redis=True,
    )
    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute(
            "SELECT account_id::text, pagerank, degree, features FROM graph_feature_snapshots WHERE snapshot_at = %s;",
            (t_snap_1,),
        )
        recomputed_t1 = {r["account_id"]: r for r in cur.fetchall()}

    for aid in (acc_fraud, acc_b, acc_c):
        assert recomputed_t1[aid]["pagerank"] == pytest.approx(initial_t1[aid]["pagerank"])
        assert recomputed_t1[aid]["degree"] == initial_t1[aid]["degree"]
        assert recomputed_t1[aid]["features"] == initial_t1[aid]["features"]

    # Later snapshot T2 sees the higher post-T1 edge weight on acc_b -> higher personalised rank than acc_c
    run_job(
        engine=isolated_graph_test_db.alembic_url,
        redis_client=fake_redis_client,
        snapshot_at=t_snap_2,
        skip_redis=False,
    )
    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute(
            "SELECT account_id::text, pagerank, features FROM graph_feature_snapshots WHERE snapshot_at = %s;",
            (t_snap_2,),
        )
        snap_t2 = {r["account_id"]: r for r in cur.fetchall()}

    assert snap_t2[acc_b]["pagerank"] > recomputed_t1[acc_b]["pagerank"]
    assert snap_t2[acc_b]["pagerank"] > snap_t2[acc_c]["pagerank"]

    # Adding an unbacked non-transaction spanning edge must raise HistoricalEdgeReconstructionError
    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO entity_edges
                (src_type, src_id, dst_type, dst_id, edge_type, weight, first_seen_at, last_seen_at)
            VALUES
                ('account', %s, 'account', %s, 'shares_card_with', 3.0, %s, %s);
            """,
            (acc_b, acc_c, t_before, t_after),
        )
        isolated_graph_test_db.conn.commit()

    with pytest.raises(HistoricalEdgeReconstructionError):
        fetch_entity_edges(isolated_graph_test_db.alembic_url, snapshot_at=t_snap_1)


def test_temporal_fraud_label_eligibility_no_future_leakage_isolated_db(
    isolated_graph_test_db, fake_redis_client
):
    """Verify future labels, future txns on txn-backed labels, future incidents, and subthreshold labels never leak."""
    t_early = datetime(2026, 11, 4, 9, 0, 0, tzinfo=timezone.utc)
    t_inc_end = datetime(2026, 11, 4, 10, 0, 0, tzinfo=timezone.utc)
    t_conf_before = datetime(2026, 11, 4, 10, 30, 0, tzinfo=timezone.utc)
    t_snap_before_future = datetime(2026, 11, 4, 11, 0, 0, tzinfo=timezone.utc)
    t_after = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)
    t_snap_after_future = datetime(2026, 11, 4, 13, 0, 0, tzinfo=timezone.utc)

    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute(
            "INSERT INTO customers (external_ref, full_name, country_code) VALUES ('CUST-TEMP-LBL', 'Temp Label Cust', 'IN') RETURNING customer_id;"
        )
        cust_id = cur.fetchone()["customer_id"]
        cur.execute(
            """
            INSERT INTO accounts (customer_id, account_number, account_type, currency, opened_on, status)
            VALUES
                (%s, 'ACC-TL-ELIG-BOUNDED', 'current', 'INR', '2026-01-01', 'active'),
                (%s, 'ACC-TL-LATE-CONF', 'current', 'INR', '2026-01-01', 'active'),
                (%s, 'ACC-TL-FUTURE-TXN', 'current', 'INR', '2026-01-01', 'active'),
                (%s, 'ACC-TL-FUTURE-INC', 'current', 'INR', '2026-01-01', 'active'),
                (%s, 'ACC-TL-LOW-CONF', 'current', 'INR', '2026-01-01', 'active'),
                (%s, 'ACC-TL-PEER', 'savings', 'INR', '2026-01-01', 'active')
            RETURNING account_number, account_id;
            """,
            (cust_id,) * 6,
        )
        acc_map = {r["account_number"]: str(r["account_id"]) for r in cur.fetchall()}

        # Peer shares device with ACC-TL-FUTURE-TXN and ACC-TL-LATE-CONF (NOT with ACC-TL-ELIG-BOUNDED at first)
        cur.execute(
            """
            INSERT INTO entity_edges
                (src_type, src_id, dst_type, dst_id, edge_type, weight, first_seen_at, last_seen_at)
            VALUES
                ('account', %s, 'device', 'dev-late', 'uses_device', 1.0, %s, %s),
                ('account', %s, 'device', 'dev-late', 'uses_device', 1.0, %s, %s),
                ('account', %s, 'device', 'dev-ftxn', 'uses_device', 1.0, %s, %s);
            """,
            (
                acc_map["ACC-TL-LATE-CONF"], t_early, t_early,
                acc_map["ACC-TL-PEER"], t_early, t_early,
                acc_map["ACC-TL-FUTURE-TXN"], t_early, t_early,
            ),
        )

        # Future transaction occurring at t_after (> t_snap_before_future)
        future_txn_id = uuid.uuid4()
        cur.execute(
            """
            INSERT INTO transactions (txn_id, external_ref, occurred_at, account_id, amount, currency, channel)
            VALUES (%s, 'EXT-TL-FUT', %s, %s, 5000.00, 'INR', 'card');
            """,
            (future_txn_id, t_after, acc_map["ACC-TL-FUTURE-TXN"]),
        )

        # Seed ground truth labels
        cur.execute(
            """
            INSERT INTO ground_truth_labels
                (account_id, txn_id, label_target, is_fraud, fraud_type, confidence, source, labeled_at, incident_start_at, incident_end_at)
            VALUES
                -- 1. Confirmed bounded completed incident before t_snap_before_future -> ELIGIBLE at 11:00
                (%s, NULL, 'account', true, 'money_mule', 0.95, 'test_fixture', %s, %s, %s),
                -- 2. Confirmed AFTER t_snap_before_future -> NOT eligible at 11:00, eligible at 13:00
                (%s, NULL, 'account', true, 'money_mule', 0.95, 'test_fixture', %s, %s, %s),
                -- 3. Txn-backed fraud with labeled_at <= 11:00 YET transaction occurred_at = 12:00 (> 11:00) -> BLOCKED at 11:00!
                (NULL, %s, 'transaction', true, 'card_not_present', 0.99, 'test_fixture', %s, NULL, NULL),
                -- 4. Bounded incident starting at 12:00 -> NOT eligible at 11:00
                (%s, NULL, 'account', true, 'account_takeover', 0.95, 'test_fixture', %s, %s, %s),
                -- 5. Sub-threshold confidence (0.45) -> NEVER eligible
                (%s, NULL, 'account', true, 'money_mule', 0.45, 'test_fixture', %s, NULL, NULL);
            """,
            (
                acc_map["ACC-TL-ELIG-BOUNDED"], t_conf_before, t_early, t_inc_end,
                acc_map["ACC-TL-LATE-CONF"], t_after, t_early, t_inc_end,
                future_txn_id, t_conf_before,
                acc_map["ACC-TL-FUTURE-INC"], t_after, t_after, t_after,
                acc_map["ACC-TL-LOW-CONF"], t_conf_before,
            ),
        )
        isolated_graph_test_db.conn.commit()

    # At T1 (11:00): ONLY ACC-TL-ELIG-BOUNDED is eligible
    fraud_t1 = fetch_fraud_account_ids(isolated_graph_test_db.alembic_url, snapshot_at=t_snap_before_future)
    assert fraud_t1 == {acc_map["ACC-TL-ELIG-BOUNDED"]}

    # Run job at T1 -> peer connected ONLY to future-fraud ACC-TL-LATE-CONF must get dist=-1 and exact risk_rank=0.0
    run_job(
        engine=isolated_graph_test_db.alembic_url,
        redis_client=fake_redis_client,
        snapshot_at=t_snap_before_future,
        skip_redis=False,
    )
    peer_feats_t1 = OF.read_graph(fake_redis_client, acc_map["ACC-TL-PEER"])
    assert peer_feats_t1["graph_dist_to_fraud"] == -1.0
    assert peer_feats_t1["graph_risk_rank"] == 0.0

    # At T0 (before any confirmation): zero fraud seeds -> exact 0.0 risk ranks everywhere
    assert fetch_fraud_account_ids(isolated_graph_test_db.alembic_url, snapshot_at=t_early) == set()

    # At T2 (13:00): late-confirmed, future-txn, and future-incident become eligible
    fraud_t2 = fetch_fraud_account_ids(isolated_graph_test_db.alembic_url, snapshot_at=t_snap_after_future)
    assert fraud_t2 == {
        acc_map["ACC-TL-ELIG-BOUNDED"],
        acc_map["ACC-TL-LATE-CONF"],
        acc_map["ACC-TL-FUTURE-TXN"],
        acc_map["ACC-TL-FUTURE-INC"],
    }


def test_mid_hour_execution_and_redis_partial_failure_retry_isolated_db(
    isolated_graph_test_db, fake_redis_client
):
    """Verify mid-hour snapshots keep exact timestamps and partial Redis failures are observable & retryable."""
    ist = timezone(timedelta(hours=5, minutes=30))
    snap_mid_ist = datetime(2026, 11, 4, 21, 5, 42, tzinfo=ist)  # == 15:35:42Z
    expected_utc = datetime(2026, 11, 4, 15, 35, 42, tzinfo=timezone.utc)

    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute(
            "INSERT INTO customers (external_ref, full_name, country_code) VALUES ('CUST-MIDHR', 'MidHour Cust', 'IN') RETURNING customer_id;"
        )
        cust_id = cur.fetchone()["customer_id"]
        cur.execute(
            """
            INSERT INTO accounts (customer_id, account_number, account_type, currency, opened_on, status)
            VALUES
                (%s, 'ACC-MH-1', 'current', 'INR', '2026-01-01', 'active'),
                (%s, 'ACC-MH-2', 'savings', 'INR', '2026-01-01', 'active')
            RETURNING account_number, account_id;
            """,
            (cust_id, cust_id),
        )
        acc_map = {r["account_number"]: str(r["account_id"]) for r in cur.fetchall()}
        isolated_graph_test_db.conn.commit()

    # Force batch_size=1 so chunk 1 commits and chunk 2 raises
    real_pipeline = fake_redis_client.pipeline
    pipe_calls = 0

    class FailSecondBatchPipe:
        def __init__(self, inner, fail: bool):
            self._inner = inner
            self._fail = fail

        def hset(self, *args, **kwargs):
            return self._inner.hset(*args, **kwargs)

        def expire(self, *args, **kwargs):
            return self._inner.expire(*args, **kwargs)

        def execute(self):
            if self._fail:
                raise redis_lib.ConnectionError("Simulated Redis outage on second chunk")
            return self._inner.execute()

    def flaky_pipe(*args, **kwargs):
        nonlocal pipe_calls
        pipe_calls += 1
        return FailSecondBatchPipe(real_pipeline(*args, **kwargs), fail=pipe_calls >= 2)

    with patch.object(fake_redis_client, "pipeline", side_effect=flaky_pipe):
        partial_summary = run_job(
            engine=isolated_graph_test_db.alembic_url,
            redis_client=fake_redis_client,
            snapshot_at=snap_mid_ist,
            batch_size=1,
        )

    assert partial_summary["status"] == "partial_failure"
    assert partial_summary["publication_status"] == "partial_redis"
    assert partial_summary["postgres_persisted"] is True
    assert partial_summary["persisted_snapshots"] == 2
    assert partial_summary["redis_published"] == 1
    assert partial_summary["redis_failed"] == 1
    assert partial_summary["snapshot_at"] == expected_utc.isoformat()
    assert partial_summary["hourly_bucket"] == "2026-11-04T15:00:00+00:00"

    # Durable DB still has exact mid-hour snapshots
    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute("SELECT snapshot_at FROM graph_feature_snapshots;")
        db_snaps = cur.fetchall()
        assert len(db_snaps) == 2
        assert all(row["snapshot_at"] == expected_utc for row in db_snaps)

    # Safe retry succeeds with zero duplicates in PostgreSQL and publishes both keys with 7200s TTL
    retry_summary = run_job(
        engine=isolated_graph_test_db.alembic_url,
        redis_client=fake_redis_client,
        snapshot_at=snap_mid_ist,
        batch_size=1,
    )
    assert retry_summary["status"] == "success"
    assert retry_summary["publication_status"] == "complete"
    assert retry_summary["redis_published"] == 2
    assert retry_summary["redis_failed"] == 0

    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute("SELECT count(*) AS cnt FROM graph_feature_snapshots WHERE snapshot_at = %s;", (expected_utc,))
        assert cur.fetchone()["cnt"] == 2

    for aid in acc_map.values():
        key = f"gfeat:acct:{aid}"
        assert 0 < fake_redis_client.ttl(key) <= D.GRAPH_TTL_S
        raw = fake_redis_client.hgetall(key)
        assert raw[b"computed_at"] == expected_utc.isoformat().encode("utf-8")


def test_directed_edge_reversal_invariance_isolated_db(isolated_graph_test_db, fake_redis_client):
    """Reversing edge directions in PostgreSQL produces identical undirected graph features."""
    t_edge = datetime(2026, 11, 4, 14, 0, 0, tzinfo=timezone.utc)
    t_snap = datetime(2026, 11, 4, 15, 0, 0, tzinfo=timezone.utc)

    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute(
            "INSERT INTO customers (external_ref, full_name, country_code) VALUES ('CUST-REV', 'Rev Cust', 'IN') RETURNING customer_id;"
        )
        cust_id = cur.fetchone()["customer_id"]
        cur.execute(
            """
            INSERT INTO accounts (customer_id, account_number, account_type, currency, opened_on, status)
            VALUES
                (%s, 'ACC-REV-F', 'current', 'INR', '2026-01-01', 'active'),
                (%s, 'ACC-REV-1', 'savings', 'INR', '2026-01-01', 'active'),
                (%s, 'ACC-REV-2', 'savings', 'INR', '2026-01-01', 'active')
            RETURNING account_number, account_id;
            """,
            (cust_id, cust_id, cust_id),
        )
        acc_map = {r["account_number"]: str(r["account_id"]) for r in cur.fetchall()}
        cur.execute(
            """
            INSERT INTO ground_truth_labels (account_id, label_target, is_fraud, fraud_type, confidence, labeled_at)
            VALUES (%s, 'account', true, 'money_mule', 1.0, %s);
            """,
            (acc_map["ACC-REV-F"], t_edge),
        )
        cur.execute(
            """
            INSERT INTO entity_edges (src_type, src_id, dst_type, dst_id, edge_type, weight, first_seen_at, last_seen_at)
            VALUES
                ('account', %s, 'device', 'dev-rev', 'uses_device', 2.0, %s, %s),
                ('account', %s, 'account', %s, 'transfers_to', 3.0, %s, %s);
            """,
            (
                acc_map["ACC-REV-F"], t_edge, t_edge,
                acc_map["ACC-REV-1"], acc_map["ACC-REV-F"], t_edge, t_edge,
            ),
        )
        isolated_graph_test_db.conn.commit()

    run_job(
        engine=isolated_graph_test_db.alembic_url,
        redis_client=fake_redis_client,
        snapshot_at=t_snap,
    )
    feats_a = OF.read_graph(fake_redis_client, acc_map["ACC-REV-1"])

    # Flip src/dst of both edges and recompute
    with isolated_graph_test_db.conn.cursor() as cur:
        cur.execute("DELETE FROM entity_edges;")
        cur.execute(
            """
            INSERT INTO entity_edges (src_type, src_id, dst_type, dst_id, edge_type, weight, first_seen_at, last_seen_at)
            VALUES
                ('device', 'dev-rev', 'account', %s, 'uses_device', 2.0, %s, %s),
                ('account', %s, 'account', %s, 'transfers_to', 3.0, %s, %s);
            """,
            (
                acc_map["ACC-REV-F"], t_edge, t_edge,
                acc_map["ACC-REV-F"], acc_map["ACC-REV-1"], t_edge, t_edge,
            ),
        )
        isolated_graph_test_db.conn.commit()

    run_job(
        engine=isolated_graph_test_db.alembic_url,
        redis_client=fake_redis_client,
        snapshot_at=t_snap,
    )
    feats_b = OF.read_graph(fake_redis_client, acc_map["ACC-REV-1"])
    assert feats_a == pytest.approx(feats_b)
