"""Database migration integration and safety tests for M2 feature store schema."""

import os
from datetime import datetime, timezone, timedelta
import pytest
import psycopg
from psycopg.rows import dict_row
from alembic.config import Config
from alembic import command
import sqlalchemy as sa
from db.migrations.versions.m2_0001_feature_store import consolidate_entity_edge_duplicates, MANAGED_COMMENT

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://aegisflow:aegisflow@localhost:5432/aegisflow"
)
PG_URL = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
ADMIN_URL = PG_URL.rsplit("/", 1)[0] + "/postgres"


@pytest.fixture(scope="function")
def db_conn():
    try:
        conn = psycopg.connect(PG_URL, row_factory=dict_row)
        conn.autocommit = False
        yield conn
        conn.close()
    except Exception as e:
        pytest.skip(f"PostgreSQL database unavailable: {e}")


def test_expected_tables_and_columns(db_conn):
    """Verify that M2 tables, columns, and types exist as defined."""
    with db_conn.cursor() as cur:
        # Check table presence
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

        # Verify entity_edges columns
        cur.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'entity_edges';
            """
        )
        cols_ee = {row["column_name"]: row for row in cur.fetchall()}
        assert cols_ee["edge_id"]["is_nullable"] == "NO"
        assert cols_ee["src_type"]["data_type"] == "character varying"
        assert cols_ee["src_id"]["data_type"] == "character varying"
        assert cols_ee["dst_type"]["data_type"] == "character varying"
        assert cols_ee["dst_id"]["data_type"] == "character varying"
        assert cols_ee["edge_type"]["data_type"] == "character varying"
        assert cols_ee["weight"]["data_type"] == "double precision"
        assert cols_ee["first_seen_at"]["data_type"] == "timestamp with time zone"
        assert cols_ee["last_seen_at"]["data_type"] == "timestamp with time zone"
        assert cols_ee["properties"]["data_type"] == "jsonb"

        # Verify graph_feature_snapshots columns
        cur.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'graph_feature_snapshots';
            """
        )
        cols_gfs = {row["column_name"]: row for row in cur.fetchall()}
        assert cols_gfs["snapshot_id"]["is_nullable"] == "NO"
        assert cols_gfs["account_id"]["is_nullable"] == "NO"
        assert cols_gfs["snapshot_at"]["data_type"] == "timestamp with time zone"
        assert cols_gfs["pagerank"]["data_type"] == "double precision"
        assert cols_gfs["degree"]["data_type"] == "integer"
        assert cols_gfs["in_degree"]["data_type"] == "integer"
        assert cols_gfs["out_degree"]["data_type"] == "integer"
        assert cols_gfs["clustering_coefficient"]["data_type"] == "double precision"
        assert cols_gfs["features"]["data_type"] == "jsonb"

        # Verify ground_truth_labels columns
        cur.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'ground_truth_labels';
            """
        )
        cols_gtl = {row["column_name"]: row for row in cur.fetchall()}
        assert cols_gtl["label_id"]["is_nullable"] == "NO"
        assert cols_gtl["label_target"]["data_type"] == "character varying"
        assert cols_gtl["is_fraud"]["data_type"] == "boolean"
        assert cols_gtl["confidence"]["data_type"] == "double precision"
        assert cols_gtl["source"]["data_type"] == "character varying"
        assert cols_gtl["labeled_at"]["data_type"] == "timestamp with time zone"


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
            VALUES ('account', 'test-acc-1', 'device', 'test-dev-1', 'used_device', 1.5)
            ON CONFLICT (src_type, src_id, dst_type, dst_id, edge_type)
            DO UPDATE SET weight = entity_edges.weight + EXCLUDED.weight,
                          last_seen_at = NOW()
            RETURNING weight;
            """
        )
        updated_weight = cur.fetchone()["weight"]
        assert updated_weight == 2.5

        # Cleanup
        cur.execute(
            """
            DELETE FROM entity_edges
            WHERE src_id = 'test-acc-1' AND dst_id = 'test-dev-1';
            """
        )
        db_conn.commit()


def test_entity_edges_check_constraints(db_conn):
    """Verify entity_edges domain check constraints."""
    # 1. Weight must be positive (> 0)
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO entity_edges (src_type, src_id, dst_type, dst_id, edge_type, weight)
                VALUES ('account', 'ck-acc-1', 'device', 'ck-dev-1', 'used_device', 0.0);
                """
            )
    db_conn.rollback()

    # 2. Empty strings rejected
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO entity_edges (src_type, src_id, dst_type, dst_id, edge_type, weight)
                VALUES ('', 'ck-acc-1', 'device', 'ck-dev-1', 'used_device', 1.0);
                """
            )
    db_conn.rollback()

    # 3. Chronology: first_seen_at <= last_seen_at
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO entity_edges (src_type, src_id, dst_type, dst_id, edge_type, weight, first_seen_at, last_seen_at)
                VALUES ('account', 'ck-acc-1', 'device', 'ck-dev-1', 'used_device', 1.0,
                        NOW() + INTERVAL '1 hour', NOW());
                """
            )
    db_conn.rollback()


def test_floating_point_integrity_and_overflow(db_conn):
    """Native PostgreSQL tests verifying rejection of NaN, +Infinity, -Infinity and overflow behavior."""
    with db_conn.cursor() as cur:
        # Create prerequisite customer and account for snapshots and labels
        cur.execute(
            """
            INSERT INTO customers (external_ref, full_name, country_code)
            VALUES ('CUST-FP-TEST', 'Float Point Test Customer', 'IN')
            ON CONFLICT (external_ref) DO UPDATE SET full_name = EXCLUDED.full_name
            RETURNING customer_id;
            """
        )
        cust_id = cur.fetchone()["customer_id"]

        cur.execute(
            """
            INSERT INTO accounts (customer_id, account_number, account_type, currency, opened_on)
            VALUES (%s, 'ACC-FP-TEST-001', 'current', 'INR', '2026-01-01')
            ON CONFLICT (account_number) DO UPDATE SET currency = EXCLUDED.currency
            RETURNING account_id;
            """,
            (cust_id,)
        )
        acc_id = cur.fetchone()["account_id"]

        cur.execute(
            """
            INSERT INTO transactions (external_ref, account_id, channel, direction, amount, occurred_at)
            VALUES ('TXN-FP-TEST-001', %s, 'wire', 'debit', 100.00, NOW())
            ON CONFLICT (external_ref) DO UPDATE SET amount = EXCLUDED.amount
            RETURNING txn_id;
            """,
            (acc_id,)
        )
        txn_id = cur.fetchone()["txn_id"]
        db_conn.commit()

    # --- 1. entity_edges.weight tests ---
    # Valid finite boundary values
    for valid_w in (1e-10, 1.0, 1000.5, 1e300):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO entity_edges (src_type, src_id, dst_type, dst_id, edge_type, weight)
                VALUES ('account', 'fp-src-ok', 'device', 'fp-dst-ok', 'edge-ok', %s)
                RETURNING weight;
                """,
                (valid_w,)
            )
            assert cur.fetchone()["weight"] == pytest.approx(valid_w)
            cur.execute("DELETE FROM entity_edges WHERE src_id = 'fp-src-ok';")
            db_conn.commit()

    # Rejected non-finite and negative/zero weights
    for invalid_w in ("'NaN'::double precision", "'Infinity'::double precision", "'-Infinity'::double precision", "0.0", "-1.0"):
        with pytest.raises(psycopg.errors.CheckViolation):
            with db_conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO entity_edges (src_type, src_id, dst_type, dst_id, edge_type, weight) "
                    f"VALUES ('account', 'fp-src', 'device', 'fp-dst', 'edge-bad', {invalid_w});"
                )
        db_conn.rollback()

    # --- 2. graph_feature_snapshots.pagerank tests ---
    now = datetime(2026, 9, 15, 10, 0, tzinfo=timezone.utc)
    # Valid finite boundary values (including zero)
    for valid_pr in (0.0, 1e-10, 0.05, 1.0, 1e300):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO graph_feature_snapshots (account_id, snapshot_at, pagerank)
                VALUES (%s, %s, %s)
                RETURNING pagerank;
                """,
                (acc_id, now, valid_pr)
            )
            assert cur.fetchone()["pagerank"] == pytest.approx(valid_pr)
            cur.execute("DELETE FROM graph_feature_snapshots WHERE account_id = %s;", (acc_id,))
            db_conn.commit()

    for invalid_pr in ("'NaN'::double precision", "'Infinity'::double precision", "'-Infinity'::double precision", "-0.001"):
        with pytest.raises(psycopg.errors.CheckViolation):
            with db_conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO graph_feature_snapshots (account_id, snapshot_at, pagerank) "
                    f"VALUES (%s, %s, {invalid_pr});",
                    (acc_id, now)
                )
        db_conn.rollback()

    # --- 3. graph_feature_snapshots.clustering_coefficient tests ---
    # Valid finite boundary values: [0.0, 1.0]
    for valid_cc in (0.0, 0.25, 0.75, 1.0):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO graph_feature_snapshots (account_id, snapshot_at, clustering_coefficient)
                VALUES (%s, %s, %s)
                RETURNING clustering_coefficient;
                """,
                (acc_id, now, valid_cc)
            )
            assert cur.fetchone()["clustering_coefficient"] == pytest.approx(valid_cc)
            cur.execute("DELETE FROM graph_feature_snapshots WHERE account_id = %s;", (acc_id,))
            db_conn.commit()

    for invalid_cc in ("'NaN'::double precision", "'Infinity'::double precision", "'-Infinity'::double precision", "-0.01", "1.01"):
        with pytest.raises(psycopg.errors.CheckViolation):
            with db_conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO graph_feature_snapshots (account_id, snapshot_at, clustering_coefficient) "
                    f"VALUES (%s, %s, {invalid_cc});",
                    (acc_id, now)
                )
        db_conn.rollback()

    # --- 4. ground_truth_labels.confidence tests ---
    # Valid finite boundary values: [0.0, 1.0]
    for valid_conf in (0.0, 0.5, 1.0):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ground_truth_labels (txn_id, label_target, is_fraud, confidence)
                VALUES (%s, 'transaction', true, %s)
                RETURNING confidence;
                """,
                (txn_id, valid_conf)
            )
            assert cur.fetchone()["confidence"] == pytest.approx(valid_conf)
            cur.execute("DELETE FROM ground_truth_labels WHERE txn_id = %s;", (txn_id,))
            db_conn.commit()

    for invalid_conf in ("'NaN'::double precision", "'Infinity'::double precision", "'-Infinity'::double precision", "-0.1", "1.1"):
        with pytest.raises(psycopg.errors.CheckViolation):
            with db_conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO ground_truth_labels (txn_id, label_target, is_fraud, confidence) "
                    f"VALUES (%s, 'transaction', true, {invalid_conf});",
                    (txn_id,)
                )
        db_conn.rollback()

    # --- 5. Weight aggregation overflow test ---
    with pytest.raises(psycopg.errors.NumericValueOutOfRange):
        with db_conn.cursor() as cur:
            cur.execute("SELECT ('1.79e308'::double precision + '1.79e308'::double precision);")
    db_conn.rollback()

    # Cleanup prerequisite data
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM transactions WHERE txn_id = %s;", (txn_id,))
        cur.execute("DELETE FROM accounts WHERE account_id = %s;", (acc_id,))
        cur.execute("DELETE FROM customers WHERE customer_id = %s;", (cust_id,))
        db_conn.commit()


def test_duplicate_consolidation_preserves_data():
    """Verify consolidate_entity_edge_duplicates preserves earliest first_seen, latest last_seen, summed weight, and merged properties."""
    engine = sa.create_engine(DATABASE_URL.replace("postgresql://", "postgresql+psycopg://"))
    with engine.begin() as conn:
        # Drop unique constraint temporarily to insert duplicates for consolidation testing
        conn.execute(sa.text("ALTER TABLE entity_edges DROP CONSTRAINT IF EXISTS uq_entity_edges_identity;"))
        conn.execute(sa.text("DELETE FROM entity_edges WHERE src_id = 'dedup-src' AND dst_id = 'dedup-dst';"))

        t1 = datetime(2026, 1, 10, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        t3 = datetime(2026, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

        # Insert duplicate occurrences with varying weights, timestamps, and properties
        conn.execute(sa.text("""
            INSERT INTO entity_edges (src_type, src_id, dst_type, dst_id, edge_type, weight, first_seen_at, last_seen_at, properties)
            VALUES
                ('account', 'dedup-src', 'device', 'dedup-dst', 'shared_hw', 1.5, :t1, :t2, '{"ip": "10.0.0.1", "channel": "mobile"}'::jsonb),
                ('account', 'dedup-src', 'device', 'dedup-dst', 'shared_hw', 3.5, :t2, :t3, '{"ip": "10.0.0.2", "app_ver": "2.4"}'::jsonb);
        """), {"t1": t1, "t2": t2, "t3": t3})

        # Run consolidation function
        consolidate_entity_edge_duplicates(conn)

        # Restore unique constraint
        conn.execute(sa.text("""
            ALTER TABLE entity_edges ADD CONSTRAINT uq_entity_edges_identity
            UNIQUE (src_type, src_id, dst_type, dst_id, edge_type);
        """))

        # Verify results
        res = conn.execute(sa.text("""
            SELECT weight, first_seen_at, last_seen_at, properties
            FROM entity_edges
            WHERE src_id = 'dedup-src' AND dst_id = 'dedup-dst';
        """)).mappings().fetchall()

        assert len(res) == 1
        survivor = res[0]
        assert survivor["weight"] == 5.0
        assert survivor["first_seen_at"] == t1
        assert survivor["last_seen_at"] == t3
        props = survivor["properties"]
        assert props["channel"] == "mobile"
        assert props["app_ver"] == "2.4"
        assert props["ip"] == ["10.0.0.1", "10.0.0.2"]

        # Cleanup
        conn.execute(sa.text("DELETE FROM entity_edges WHERE src_id = 'dedup-src' AND dst_id = 'dedup-dst';"))
    engine.dispose()


def test_graph_snapshots_uniqueness_and_cascade(db_conn):
    """Verify graph_feature_snapshots uniqueness, check constraints, and cascade delete."""
    with db_conn.cursor() as cur:
        # Create test customer and account
        cur.execute(
            """
            INSERT INTO customers (external_ref, full_name, country_code)
            VALUES ('CUST-M2-CASCADE', 'Cascade Test Customer', 'IN')
            ON CONFLICT (external_ref) DO UPDATE SET full_name = EXCLUDED.full_name
            RETURNING customer_id;
            """
        )
        cust_id = cur.fetchone()["customer_id"]

        cur.execute(
            """
            INSERT INTO accounts (customer_id, account_number, account_type, currency, opened_on)
            VALUES (%s, 'ACC-M2-CASCADE-001', 'current', 'INR', '2026-01-01')
            ON CONFLICT (account_number) DO UPDATE SET currency = EXCLUDED.currency
            RETURNING account_id;
            """,
            (cust_id,)
        )
        account_id = cur.fetchone()["account_id"]
        now = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)

        # Cleanup test snapshot
        cur.execute(
            "DELETE FROM graph_feature_snapshots WHERE account_id = %s;",
            (account_id,)
        )
        # 1. Insert snapshot
        cur.execute(
            """
            INSERT INTO graph_feature_snapshots
                (account_id, snapshot_at, pagerank, degree, in_degree, out_degree, clustering_coefficient, features)
            VALUES (%s, %s, 0.05, 10, 6, 4, 0.12, '{"hub_score": 0.8}'::jsonb);
            """,
            (account_id, now)
        )
        db_conn.commit()

    # 2. Duplicate insert on (account_id, snapshot_at) must fail
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

    # 3. Check constraints
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO graph_feature_snapshots
                    (account_id, snapshot_at, pagerank, degree)
                VALUES (%s, %s, -0.01, 5);
                """,
                (account_id, now + timedelta(hours=1))
            )
    db_conn.rollback()

    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO graph_feature_snapshots
                    (account_id, snapshot_at, clustering_coefficient)
                VALUES (%s, %s, 1.5);
                """,
                (account_id, now + timedelta(hours=1))
            )
    db_conn.rollback()

    # 4. Cascade delete: deleting account must cascade-delete snapshot
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM accounts WHERE account_id = %s;", (account_id,))
        cur.execute("SELECT count(*) FROM graph_feature_snapshots WHERE account_id = %s;", (account_id,))
        cnt = cur.fetchone()["count"]
        assert cnt == 0
        cur.execute("DELETE FROM customers WHERE customer_id = %s;", (cust_id,))
        db_conn.commit()


def test_ground_truth_labels_constraints_and_bounds(db_conn):
    """Verify ground_truth_labels domain constraints, confidence bounds, and target model."""
    with db_conn.cursor() as cur:
        # Create test customer, account, transaction
        cur.execute(
            """
            INSERT INTO customers (external_ref, full_name, country_code)
            VALUES ('CUST-M2-LABELS', 'Labels Test Customer', 'IN')
            ON CONFLICT (external_ref) DO UPDATE SET full_name = EXCLUDED.full_name
            RETURNING customer_id;
            """
        )
        cust_id = cur.fetchone()["customer_id"]

        cur.execute(
            """
            INSERT INTO accounts (customer_id, account_number, account_type, currency, opened_on)
            VALUES (%s, 'ACC-M2-LABELS-001', 'current', 'INR', '2026-01-01')
            ON CONFLICT (account_number) DO UPDATE SET currency = EXCLUDED.currency
            RETURNING account_id;
            """,
            (cust_id,)
        )
        account_id = cur.fetchone()["account_id"]

        cur.execute(
            """
            INSERT INTO transactions (external_ref, account_id, channel, direction, amount, occurred_at)
            VALUES ('TXN-M2-LABELS-001', %s, 'wire', 'debit', 50000.00, NOW())
            ON CONFLICT (external_ref) DO UPDATE SET amount = EXCLUDED.amount
            RETURNING txn_id;
            """,
            (account_id,)
        )
        txn_id = cur.fetchone()["txn_id"]

        # Clean prior test labels
        cur.execute("DELETE FROM ground_truth_labels WHERE txn_id = %s OR account_id = %s;", (txn_id, account_id))
        db_conn.commit()

    # 1. Valid transaction label
    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ground_truth_labels (txn_id, label_target, is_fraud, fraud_type, confidence, source)
            VALUES (%s, 'transaction', true, 'money_mule', 0.95, 'analyst_review')
            RETURNING label_id;
            """,
            (txn_id,)
        )
        assert cur.fetchone()["label_id"] is not None
        db_conn.commit()

    # 2. Duplicate label for same transaction must fail (uq_ground_truth_txn)
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ground_truth_labels (txn_id, label_target, is_fraud, fraud_type, confidence, source)
                VALUES (%s, 'transaction', true, 'account_takeover', 0.80, 'tipoff');
                """,
                (txn_id,)
            )
    db_conn.rollback()

    # 3. Confidence range checks: must be in [0.0, 1.0]
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ground_truth_labels (account_id, label_target, is_fraud, confidence)
                VALUES (%s, 'account', true, 1.2);
                """,
                (account_id,)
            )
    db_conn.rollback()

    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ground_truth_labels (account_id, label_target, is_fraud, confidence)
                VALUES (%s, 'account', true, -0.1);
                """,
                (account_id,)
            )
    db_conn.rollback()

    # 4. Target validity: target must be 'transaction' or 'account'
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ground_truth_labels (account_id, label_target, is_fraud)
                VALUES (%s, 'device', true);
                """,
                (account_id,)
            )
    db_conn.rollback()

    # 5. Target presence check: transaction target must have txn_id and NULL account_id
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ground_truth_labels (label_target, is_fraud, confidence)
                VALUES ('transaction', true, 0.9);
                """
            )
    db_conn.rollback()

    # Ambiguous target: transaction target with both txn_id and account_id must fail
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ground_truth_labels (txn_id, account_id, label_target, is_fraud, confidence)
                VALUES (%s, %s, 'transaction', true, 0.9);
                """,
                (txn_id, account_id)
            )
    db_conn.rollback()

    # 6. Account-level label: account target must have account_id and NULL txn_id
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ground_truth_labels (label_target, is_fraud, confidence)
                VALUES ('account', true, 0.9);
                """
            )
    db_conn.rollback()

    # Ambiguous target: account target with both account_id and txn_id must fail
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ground_truth_labels (txn_id, account_id, label_target, is_fraud, confidence)
                VALUES (%s, %s, 'account', true, 0.9);
                """,
                (txn_id, account_id)
            )
    db_conn.rollback()

    # 7. Valid account label and uniqueness per (account_id, fraud_type)
    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ground_truth_labels (account_id, label_target, is_fraud, fraud_type, confidence)
            VALUES (%s, 'account', true, 'mule_ring', 0.90);
            """,
            (account_id,)
        )
        db_conn.commit()

    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ground_truth_labels (account_id, label_target, is_fraud, fraud_type, confidence)
                VALUES (%s, 'account', true, 'mule_ring', 0.70);
                """,
                (account_id,)
            )
    db_conn.rollback()

    # Cleanup
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM ground_truth_labels WHERE txn_id = %s OR account_id = %s;", (txn_id, account_id))
        cur.execute("DELETE FROM transactions WHERE txn_id = %s;", (txn_id,))
        cur.execute("DELETE FROM accounts WHERE account_id = %s;", (account_id,))
        cur.execute("DELETE FROM customers WHERE customer_id = %s;", (cust_id,))
        db_conn.commit()


def test_compatible_preexisting_table_adoption():
    """Dedicated 9-step test verifying unmanaged compatible table adoption and safe downgrade."""
    admin_conn = psycopg.connect(ADMIN_URL, autocommit=True)
    disposable_db = "aegisflow_test_compat_adopt"
    admin_conn.cursor().execute(f"DROP DATABASE IF EXISTS {disposable_db};")
    admin_conn.cursor().execute(f"CREATE DATABASE {disposable_db};")

    test_db_url = f"{PG_URL.rsplit('/', 1)[0]}/{disposable_db}"
    alembic_db_url = f"postgresql+psycopg://{test_db_url.split('://', 1)[1]}"

    # Step 2: Apply the M1 SQL bootstrap
    with open("scripts/init_db.sql") as f:
        m1_sql = f.read()

    conn = psycopg.connect(test_db_url, autocommit=True)
    cur = conn.cursor()
    cur.execute(m1_sql)

    # Step 3: Create an unmanaged but compatible entity_edges table (no managed comment)
    cur.execute("""
        CREATE TABLE entity_edges (
            edge_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            src_type VARCHAR(32) NOT NULL,
            src_id VARCHAR(80) NOT NULL,
            dst_type VARCHAR(32) NOT NULL,
            dst_id VARCHAR(80) NOT NULL,
            edge_type VARCHAR(32) NOT NULL,
            weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
            first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            properties JSONB NOT NULL DEFAULT '{}'::jsonb
        );
    """)

    # Step 4: Insert duplicate logical edges with different weights, timestamps, and properties
    t1 = datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 2, 2, 12, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 2, 3, 14, 0, tzinfo=timezone.utc)
    cur.execute(
        """
        INSERT INTO entity_edges (src_type, src_id, dst_type, dst_id, edge_type, weight, first_seen_at, last_seen_at, properties)
        VALUES
            ('account', 'acc-compat-1', 'device', 'dev-compat-1', 'used_device', 1.25, %s, %s, '{"ip": "10.0.0.1", "device_model": "iPhone"}'::jsonb),
            ('account', 'acc-compat-1', 'device', 'dev-compat-1', 'used_device', 2.75, %s, %s, '{"ip": "10.0.0.2", "os": "iOS 18"}'::jsonb);
        """,
        (t1, t2, t2, t3)
    )
    conn.close()

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", alembic_db_url)

    # Step 5: Execute alembic upgrade head
    command.upgrade(cfg, "head")

    # Step 6: Verify deterministic consolidation and unique constraint creation
    conn = psycopg.connect(test_db_url, autocommit=True)
    cur = conn.cursor()
    cur.execute("SELECT weight, first_seen_at, last_seen_at, properties FROM entity_edges WHERE src_id = 'acc-compat-1';")
    rows = cur.fetchall()
    assert len(rows) == 1
    consolidated = rows[0]
    assert consolidated[0] == pytest.approx(4.0)
    assert consolidated[1] == t1
    assert consolidated[2] == t3
    assert consolidated[3]["device_model"] == "iPhone"
    assert consolidated[3]["os"] == "iOS 18"
    assert consolidated[3]["ip"] == ["10.0.0.1", "10.0.0.2"]

    # Verify unique constraint exists
    cur.execute("""
        SELECT conname FROM pg_constraint
        WHERE conrelid = 'entity_edges'::regclass AND conname = 'uq_entity_edges_identity';
    """)
    assert cur.fetchone() is not None

    # Step 7: Execute alembic downgrade base
    command.downgrade(cfg, "base")

    # Step 8: Verify the unmanaged table and its data remain intact
    cur.execute("SELECT count(*), sum(weight) FROM entity_edges;")
    cnt, total_w = cur.fetchone()
    assert cnt == 1
    assert total_w == pytest.approx(4.0)

    # Step 9: Verify migration-added constraints are correctly reverted
    cur.execute("""
        SELECT conname FROM pg_constraint
        WHERE conrelid = 'entity_edges'::regclass AND conname = 'uq_entity_edges_identity';
    """)
    assert cur.fetchone() is None
    conn.close()

    # Cleanup disposable database
    admin_conn.cursor().execute(f"DROP DATABASE {disposable_db};")
    admin_conn.close()


def test_ownership_marker_safety():
    """Verify exact ownership marker matching, protection of unrelated comments, and independent target table downgrade."""
    admin_conn = psycopg.connect(ADMIN_URL, autocommit=True)
    disposable_db = "aegisflow_test_ownership"
    admin_conn.cursor().execute(f"DROP DATABASE IF EXISTS {disposable_db};")
    admin_conn.cursor().execute(f"CREATE DATABASE {disposable_db};")

    test_db_url = f"{PG_URL.rsplit('/', 1)[0]}/{disposable_db}"
    alembic_db_url = f"postgresql+psycopg://{test_db_url.split('://', 1)[1]}"

    # Seed M1 bootstrap
    with open("scripts/init_db.sql") as f:
        m1_sql = f.read()

    conn = psycopg.connect(test_db_url, autocommit=True)
    cur = conn.cursor()
    cur.execute(m1_sql)

    # Create entity_edges with an unrelated comment
    cur.execute("""
        CREATE TABLE entity_edges (
            edge_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            src_type VARCHAR(32) NOT NULL,
            src_id VARCHAR(80) NOT NULL,
            dst_type VARCHAR(32) NOT NULL,
            dst_id VARCHAR(80) NOT NULL,
            edge_type VARCHAR(32) NOT NULL,
            weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
            first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            properties JSONB NOT NULL DEFAULT '{}'::jsonb
        );
        COMMENT ON TABLE entity_edges IS 'unrelated:legacy_system=v0_staging';
        INSERT INTO entity_edges (src_type, src_id, dst_type, dst_id, edge_type, weight)
        VALUES ('account', 'unrelated-src', 'device', 'unrelated-dst', 'used_device', 1.0);
    """)
    conn.close()

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", alembic_db_url)

    # Upgrade: adopts entity_edges and creates graph_feature_snapshots & ground_truth_labels
    command.upgrade(cfg, "head")

    conn = psycopg.connect(test_db_url, autocommit=True)
    cur = conn.cursor()

    # Confirm entity_edges still has its original unrelated comment
    cur.execute("SELECT obj_description(c.oid, 'pg_class') FROM pg_class c WHERE c.relname = 'entity_edges';")
    assert cur.fetchone()[0] == "unrelated:legacy_system=v0_staging"

    # Confirm migration-created tables have exact MANAGED_COMMENT
    cur.execute("SELECT obj_description(c.oid, 'pg_class') FROM pg_class c WHERE c.relname = 'graph_feature_snapshots';")
    assert cur.fetchone()[0] == MANAGED_COMMENT
    cur.execute("SELECT obj_description(c.oid, 'pg_class') FROM pg_class c WHERE c.relname = 'ground_truth_labels';")
    assert cur.fetchone()[0] == MANAGED_COMMENT

    # Downgrade: must independently drop managed tables, but preserve entity_edges
    command.downgrade(cfg, "base")

    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name IN ('entity_edges', 'graph_feature_snapshots', 'ground_truth_labels');
    """)
    remaining_tables = {r[0] for r in cur.fetchall()}
    assert "entity_edges" in remaining_tables
    assert "graph_feature_snapshots" not in remaining_tables
    assert "ground_truth_labels" not in remaining_tables

    # Confirm entity_edges data and comment remain intact
    cur.execute("SELECT count(*) FROM entity_edges WHERE src_id = 'unrelated-src';")
    assert cur.fetchone()[0] == 1
    cur.execute("SELECT obj_description(c.oid, 'pg_class') FROM pg_class c WHERE c.relname = 'entity_edges';")
    assert cur.fetchone()[0] == "unrelated:legacy_system=v0_staging"

    conn.close()

    # Cleanup disposable database
    admin_conn.cursor().execute(f"DROP DATABASE {disposable_db};")
    admin_conn.close()


def test_sentinel_preexisting_table_safety():
    """Verify that migration aborts on incompatible pre-existing table and protects sentinel data across downgrade."""
    admin_conn = psycopg.connect(ADMIN_URL, autocommit=True)
    disposable_db = "aegisflow_test_sentinel"
    admin_conn.cursor().execute(f"DROP DATABASE IF EXISTS {disposable_db};")
    admin_conn.cursor().execute(f"CREATE DATABASE {disposable_db};")

    test_db_url = f"{PG_URL.rsplit('/', 1)[0]}/{disposable_db}"
    alembic_db_url = f"postgresql+psycopg://{test_db_url.split('://', 1)[1]}"

    # Incompatible table
    conn = psycopg.connect(test_db_url, autocommit=True)
    cur = conn.cursor()
    cur.execute("CREATE TABLE entity_edges (id int PRIMARY KEY, sentinel text);")
    cur.execute("INSERT INTO entity_edges (id, sentinel) VALUES (1, 'sentinel_do_not_drop');")
    conn.close()

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", alembic_db_url)

    # Attempt migration upgrade -> must fail precondition check
    upgrade_failed = False
    try:
        command.upgrade(cfg, "head")
    except Exception as e:
        upgrade_failed = True
        assert "Precondition failed: table 'entity_edges' already exists with incompatible schema" in str(e)

    assert upgrade_failed, "Expected RuntimeError on incompatible pre-existing table"

    # Confirm sentinel row remains untouched
    conn = psycopg.connect(test_db_url, autocommit=True)
    cur = conn.cursor()
    cur.execute("SELECT sentinel FROM entity_edges WHERE id = 1;")
    assert cur.fetchone()[0] == "sentinel_do_not_drop"

    # Attempt downgrade -> sentinel table must still remain untouched
    try:
        command.downgrade(cfg, "base")
    except Exception:
        pass

    cur.execute("SELECT sentinel FROM entity_edges WHERE id = 1;")
    assert cur.fetchone()[0] == "sentinel_do_not_drop"
    conn.close()

    # Cleanup disposable database
    admin_conn.cursor().execute(f"DROP DATABASE {disposable_db};")
    admin_conn.close()


def test_fresh_m1_bootstrap_and_migration_lifecycle():
    """Verify clean M1 bootstrap -> upgrade head -> downgrade base -> upgrade head lifecycle."""
    admin_conn = psycopg.connect(ADMIN_URL, autocommit=True)
    disposable_db = "aegisflow_test_lifecycle"
    admin_conn.cursor().execute(f"DROP DATABASE IF EXISTS {disposable_db};")
    admin_conn.cursor().execute(f"CREATE DATABASE {disposable_db};")

    test_db_url = f"{PG_URL.rsplit('/', 1)[0]}/{disposable_db}"
    alembic_db_url = f"postgresql+psycopg://{test_db_url.split('://', 1)[1]}"

    # Step 1: Bootstrap clean DB with scripts/init_db.sql
    with open("scripts/init_db.sql") as f:
        m1_sql = f.read()

    conn = psycopg.connect(test_db_url, autocommit=True)
    cur = conn.cursor()
    cur.execute(m1_sql)

    # Step 2: Verify exactly the 14 M1 tables exist
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;")
    m1_tables = {r[0] for r in cur.fetchall()}
    assert len(m1_tables) == 14
    assert "entity_edges" not in m1_tables
    assert "graph_feature_snapshots" not in m1_tables
    assert "ground_truth_labels" not in m1_tables

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", alembic_db_url)

    # Step 3: Upgrade to head
    command.upgrade(cfg, "head")

    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;")
    upgraded_tables = {r[0] for r in cur.fetchall()}
    assert "entity_edges" in upgraded_tables
    assert "graph_feature_snapshots" in upgraded_tables
    assert "ground_truth_labels" in upgraded_tables
    assert "alembic_version" in upgraded_tables
    assert len(upgraded_tables) == 18

    # Verify table comments
    for tbl in ('entity_edges', 'graph_feature_snapshots', 'ground_truth_labels'):
        cur.execute("SELECT obj_description(c.oid, 'pg_class') FROM pg_class c WHERE c.relname = %s;", (tbl,))
        assert cur.fetchone()[0] == MANAGED_COMMENT

    # Step 4: Downgrade to base
    command.downgrade(cfg, "base")

    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;")
    downgraded_tables = {r[0] for r in cur.fetchall()}
    assert "entity_edges" not in downgraded_tables
    assert "graph_feature_snapshots" not in downgraded_tables
    assert "ground_truth_labels" not in downgraded_tables
    # All 14 M1 tables must be completely intact
    for tbl in m1_tables:
        assert tbl in downgraded_tables

    # Step 5: Re-upgrade to head
    command.upgrade(cfg, "head")

    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;")
    reupgraded_tables = {r[0] for r in cur.fetchall()}
    assert "entity_edges" in reupgraded_tables
    assert "graph_feature_snapshots" in reupgraded_tables
    assert "ground_truth_labels" in reupgraded_tables
    assert len(reupgraded_tables) == 18

    conn.close()

    # Cleanup disposable database
    admin_conn.cursor().execute(f"DROP DATABASE {disposable_db};")
    admin_conn.close()
