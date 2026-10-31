"""Database migration tests for M2 feature store schema."""

import os
from datetime import datetime, timezone
import pytest
import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://aegisflow:aegisflow@localhost:5432/aegisflow"
)
PG_URL = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")


@pytest.fixture(scope="module")
def db_conn():
    try:
        conn = psycopg.connect(PG_URL, row_factory=dict_row)
        conn.autocommit = False
        yield conn
        conn.close()
    except Exception as e:
        pytest.skip(f"PostgreSQL database unavailable: {e}")


def test_migration_tables_exist(db_conn):
    """Verify that M2 tables exist in the database."""
    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN ('entity_edges', 'graph_feature_snapshots', 'ground_truth_labels');
            """
        )
        tables = {row["table_name"] for row in cur.fetchall()}
        assert "entity_edges" in tables
        assert "graph_feature_snapshots" in tables
        assert "ground_truth_labels" in tables


def test_entity_edges_uniqueness_and_upsert(db_conn):
    """Verify entity_edges unique constraint and upsert capability."""
    with db_conn.cursor() as cur:
        # Clean test records
        cur.execute(
            """
            DELETE FROM entity_edges
            WHERE src_id = 'test-acc-1' AND dst_id = 'test-dev-1';
            """
        )
        # 1. Insert initial edge and commit so it persists
        cur.execute(
            """
            INSERT INTO entity_edges (src_type, src_id, dst_type, dst_id, edge_type, weight)
            VALUES ('account', 'test-acc-1', 'device', 'test-dev-1', 'used_device', 1.0)
            RETURNING edge_id;
            """
        )
        edge_id = cur.fetchone()["edge_id"]
        assert edge_id is not None
        db_conn.commit()

    # 2. Attempt duplicate insert without conflict clause -> should raise UniqueViolation
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO entity_edges (src_type, src_id, dst_type, dst_id, edge_type, weight)
                VALUES ('account', 'test-acc-1', 'device', 'test-dev-1', 'used_device', 2.0);
                """
            )
    db_conn.rollback()

    # 3. Test online upsert support via ON CONFLICT DO UPDATE
    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO entity_edges (src_type, src_id, dst_type, dst_id, edge_type, weight)
            VALUES ('account', 'test-acc-1', 'device', 'test-dev-1', 'used_device', 1.0)
            ON CONFLICT (src_type, src_id, dst_type, dst_id, edge_type)
            DO UPDATE SET weight = entity_edges.weight + EXCLUDED.weight,
                          last_seen_at = NOW()
            RETURNING weight;
            """
        )
        updated_weight = cur.fetchone()["weight"]
        assert updated_weight == 2.0

        # Cleanup
        cur.execute(
            """
            DELETE FROM entity_edges
            WHERE src_id = 'test-acc-1' AND dst_id = 'test-dev-1';
            """
        )
        db_conn.commit()


def test_graph_feature_snapshots_uniqueness(db_conn):
    """Verify graph_feature_snapshots account and timestamp uniqueness."""
    with db_conn.cursor() as cur:
        # Create a test customer and account if none exist
        cur.execute(
            """
            INSERT INTO customers (external_ref, full_name, country_code)
            VALUES ('CUST-M2-TEST', 'Test M2 Migration Customer', 'IN')
            ON CONFLICT (external_ref) DO UPDATE SET full_name = EXCLUDED.full_name
            RETURNING customer_id;
            """
        )
        cust_id = cur.fetchone()["customer_id"]

        cur.execute(
            """
            INSERT INTO accounts (customer_id, account_number, account_type, currency, opened_on)
            VALUES (%s, 'ACC-M2-TEST-001', 'current', 'INR', '2026-01-01')
            ON CONFLICT (account_number) DO UPDATE SET currency = EXCLUDED.currency
            RETURNING account_id;
            """,
            (cust_id,)
        )
        account_id = cur.fetchone()["account_id"]
        now = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)

        # Cleanup test snapshot
        cur.execute(
            "DELETE FROM graph_feature_snapshots WHERE account_id = %s AND snapshot_at = %s;",
            (account_id, now)
        )
        # Insert snapshot
        cur.execute(
            """
            INSERT INTO graph_feature_snapshots
                (account_id, snapshot_at, pagerank, degree, in_degree, out_degree, clustering_coefficient, features)
            VALUES (%s, %s, 0.05, 10, 6, 4, 0.12, '{"hub_score": 0.8}'::jsonb);
            """,
            (account_id, now)
        )
        db_conn.commit()

    # Duplicate insert on (account_id, snapshot_at) must fail
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO graph_feature_snapshots
                    (account_id, snapshot_at, pagerank, degree)
                VALUES (%s, %s, 0.10, 15);
                """,
                (account_id, now)
            )
    db_conn.rollback()

    # Cleanup
    with db_conn.cursor() as cur:
        cur.execute(
            "DELETE FROM graph_feature_snapshots WHERE account_id = %s AND snapshot_at = %s;",
            (account_id, now)
        )
        cur.execute("DELETE FROM accounts WHERE account_id = %s;", (account_id,))
        cur.execute("DELETE FROM customers WHERE customer_id = %s;", (cust_id,))
        db_conn.commit()
