"""Integration and safety tests for Alembic migration m2_0003 (Ground Truth Incidents).

Tests run against an isolated disposable PostgreSQL database.
Never modifies, downgrades, or locks the shared development database.
"""

import os
from datetime import datetime, timezone, timedelta
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.errors import CheckViolation, UniqueViolation
import pytest
from alembic import command
from alembic.config import Config
import sqlalchemy as sa

from ingestion import config


DATABASE_URL = os.getenv("DATABASE_URL", config.DATABASE_URL)
PG_URL = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
ADMIN_URL = os.getenv("ADMIN_DATABASE_URL", PG_URL.rsplit("/", 1)[0] + "/postgres")
ALEMBIC_INI_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../alembic.ini")
)


def _validate_disposable_db_name(name: str) -> None:
    """Explicit runtime guard ensuring target database is an isolated disposable test database."""
    if not isinstance(name, str) or not name.startswith("aegisflow_test_m2_0003_"):
        raise RuntimeError(
            f"Safety check failed: database name {name!r} does not have required prefix 'aegisflow_test_m2_0003_'"
        )
    if name == "aegisflow":
        raise RuntimeError("Safety check failed: database name cannot be development database 'aegisflow'")


@pytest.fixture(scope="function")
def isolated_test_db():
    """Create a completely isolated, disposable PostgreSQL database for migration tests.

    Bootstrapped with scripts/init_db.sql and upgraded through Alembic migrations.
    Dropped in teardown using DROP DATABASE ... WITH (FORCE).
    """
    try:
        admin_conn = psycopg.connect(ADMIN_URL, autocommit=True)
    except Exception as exc:
        pytest.skip(f"PostgreSQL admin database unavailable for isolated test database: {exc}")

    db_id = uuid4().hex[:8]
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

    test_db_url = f"{PG_URL.rsplit('/', 1)[0]}/{disposable_db}"
    alembic_db_url = f"postgresql+psycopg://{test_db_url.split('://', 1)[1]}"

    # 1. Bootstrap with clean M1 schema
    init_sql_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../scripts/init_db.sql")
    )
    with open(init_sql_path) as f:
        m1_sql = f.read()

    with psycopg.connect(test_db_url, autocommit=True) as init_conn:
        init_conn.execute(m1_sql)

    # 2. Upgrade to m2_0002 first
    cfg = Config(ALEMBIC_INI_PATH)
    cfg.set_main_option("sqlalchemy.url", alembic_db_url)
    command.upgrade(cfg, "m2_0002")

    test_conn = psycopg.connect(test_db_url, row_factory=dict_row)
    test_conn.autocommit = False

    class IsolatedDB:
        name = disposable_db
        url = test_db_url
        alembic_url = alembic_db_url
        conn = test_conn
        alembic_cfg = cfg

    isolated = IsolatedDB()

    try:
        yield isolated
    finally:
        try:
            test_conn.close()
        except Exception:
            pass
        try:
            _validate_disposable_db_name(disposable_db)
            admin_conn.execute(
                psycopg.sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE);").format(
                    psycopg.sql.Identifier(disposable_db)
                )
            )
        except Exception:
            pass
        admin_conn.close()


def _seed_account(cur, cust_id=None, acc_id=None):
    if cust_id is None:
        cust_id = uuid4()
    if acc_id is None:
        acc_id = uuid4()
    cur.execute(
        "INSERT INTO customers (customer_id, external_ref, full_name, country_code) VALUES (%s, %s, 'Test User', 'IN')",
        (cust_id, f"C-{cust_id.hex[:10]}")
    )
    cur.execute(
        "INSERT INTO accounts (account_id, customer_id, account_number, account_type, opened_on) VALUES (%s, %s, %s, 'savings', CURRENT_DATE)",
        (acc_id, cust_id, f"A-{acc_id.hex[:10]}")
    )
    return cust_id, acc_id


def _seed_transaction(cur, acc_id, txn_id=None, occurred_at=None):
    if txn_id is None:
        txn_id = uuid4()
    if occurred_at is None:
        occurred_at = datetime.now(timezone.utc)
    cur.execute("""
        INSERT INTO transactions (txn_id, external_ref, occurred_at, account_id, amount, currency, channel)
        VALUES (%s, %s, %s, %s, 100.00, 'INR', 'card')
    """, (txn_id, f"TXN-{txn_id.hex[:10]}", occurred_at, acc_id))
    return txn_id


def test_m2_0003_upgrade_and_columns(isolated_test_db):
    """Verify upgrade to m2_0003 succeeds and incident boundary columns exist with proper types."""
    cfg = isolated_test_db.alembic_cfg
    command.upgrade(cfg, "m2_0003")

    conn = isolated_test_db.conn
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'ground_truth_labels'
              AND column_name IN ('incident_start_at', 'incident_end_at')
            ORDER BY column_name;
        """)
        cols = {row["column_name"]: row for row in cur.fetchall()}
        conn.commit()

    assert "incident_start_at" in cols
    assert cols["incident_start_at"]["data_type"] == "timestamp with time zone"
    assert cols["incident_start_at"]["is_nullable"] == "YES"

    assert "incident_end_at" in cols
    assert cols["incident_end_at"]["data_type"] == "timestamp with time zone"
    assert cols["incident_end_at"]["is_nullable"] == "YES"


def test_incident_boundaries_both_null_accepted(isolated_test_db):
    """Verify timeless labels with both incident boundaries NULL are accepted."""
    cfg = isolated_test_db.alembic_cfg
    command.upgrade(cfg, "m2_0003")

    conn = isolated_test_db.conn
    with conn.cursor() as cur:
        _, acc_id = _seed_account(cur)
        cur.execute("""
            INSERT INTO ground_truth_labels (
                account_id, label_target, is_fraud, fraud_type, confidence,
                incident_start_at, incident_end_at
            ) VALUES (%s, 'account', true, 'bust_out', 1.0, NULL, NULL)
            RETURNING label_id;
        """, (acc_id,))
        row = cur.fetchone()
        conn.commit()

    assert row is not None
    assert row["label_id"] is not None


def test_incident_boundaries_valid_chronology_accepted(isolated_test_db):
    """Verify bounded incident with valid chronology (start <= end <= labeled_at) is accepted."""
    cfg = isolated_test_db.alembic_cfg
    command.upgrade(cfg, "m2_0003")

    conn = isolated_test_db.conn
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=10)
    end = now - timedelta(days=5)

    with conn.cursor() as cur:
        _, acc_id = _seed_account(cur)
        cur.execute("""
            INSERT INTO ground_truth_labels (
                account_id, label_target, is_fraud, fraud_type, confidence,
                incident_start_at, incident_end_at, labeled_at
            ) VALUES (%s, 'account', true, 'money_mule', 0.95, %s, %s, %s)
            RETURNING label_id, incident_start_at, incident_end_at;
        """, (acc_id, start, end, now))
        row = cur.fetchone()
        conn.commit()

    assert row is not None
    assert row["incident_start_at"] == start
    assert row["incident_end_at"] == end


def test_incident_boundary_single_populated_rejected(isolated_test_db):
    """Verify that specifying only one boundary is rejected by paired boundary check constraint."""
    cfg = isolated_test_db.alembic_cfg
    command.upgrade(cfg, "m2_0003")

    conn = isolated_test_db.conn
    now = datetime.now(timezone.utc)

    # 1. start populated, end NULL -> rejected
    with conn.cursor() as cur:
        _, acc_id = _seed_account(cur)
        with pytest.raises(CheckViolation) as exc_info:
            cur.execute("""
                INSERT INTO ground_truth_labels (
                    account_id, label_target, is_fraud, fraud_type, confidence,
                    incident_start_at, incident_end_at
                ) VALUES (%s, 'account', true, 'account_takeover', 1.0, %s, NULL)
            """, (acc_id, now))
        conn.rollback()
        assert "ck_ground_truth_incident_paired_boundaries" in str(exc_info.value)

    # 2. start NULL, end populated -> rejected
    with conn.cursor() as cur:
        _, acc_id = _seed_account(cur)
        with pytest.raises(CheckViolation) as exc_info:
            cur.execute("""
                INSERT INTO ground_truth_labels (
                    account_id, label_target, is_fraud, fraud_type, confidence,
                    incident_start_at, incident_end_at
                ) VALUES (%s, 'account', true, 'account_takeover', 1.0, NULL, %s)
            """, (acc_id, now))
        conn.rollback()
        assert "ck_ground_truth_incident_paired_boundaries" in str(exc_info.value)


def test_reversed_incident_chronology_rejected(isolated_test_db):
    """Verify reversed incident chronology (start > end) and future discovery (start > labeled_at) are rejected."""
    cfg = isolated_test_db.alembic_cfg
    command.upgrade(cfg, "m2_0003")

    conn = isolated_test_db.conn
    now = datetime.now(timezone.utc)
    t_early = now - timedelta(days=10)
    t_late = now - timedelta(days=2)

    # 1. start > end -> rejected by ck_ground_truth_incident_chronology
    with conn.cursor() as cur:
        _, acc_id = _seed_account(cur)
        with pytest.raises(CheckViolation) as exc_info:
            cur.execute("""
                INSERT INTO ground_truth_labels (
                    account_id, label_target, is_fraud, fraud_type, confidence,
                    incident_start_at, incident_end_at, labeled_at
                ) VALUES (%s, 'account', true, 'velocity_spike', 0.9, %s, %s, %s)
            """, (acc_id, t_late, t_early, now))
        conn.rollback()
        assert "ck_ground_truth_incident_chronology" in str(exc_info.value)

    # 2. start > labeled_at -> rejected by ck_ground_truth_incident_discovery
    with conn.cursor() as cur:
        _, acc_id = _seed_account(cur)
        t_future_start = now + timedelta(days=1)
        t_future_end = now + timedelta(days=2)
        with pytest.raises(CheckViolation) as exc_info:
            cur.execute("""
                INSERT INTO ground_truth_labels (
                    account_id, label_target, is_fraud, fraud_type, confidence,
                    incident_start_at, incident_end_at, labeled_at
                ) VALUES (%s, 'account', true, 'velocity_spike', 0.9, %s, %s, %s)
            """, (acc_id, t_future_start, t_future_end, now))
        conn.rollback()
        assert "ck_ground_truth_incident_discovery" in str(exc_info.value)


def test_transaction_level_uniqueness_preserved(isolated_test_db):
    """Verify existing transaction-level uniqueness (uq_ground_truth_txn) remains enforced."""
    cfg = isolated_test_db.alembic_cfg
    command.upgrade(cfg, "m2_0003")

    conn = isolated_test_db.conn
    with conn.cursor() as cur:
        _, acc_id = _seed_account(cur)
        txn_id = _seed_transaction(cur, acc_id)

        # First label on txn_id succeeds
        cur.execute("""
            INSERT INTO ground_truth_labels (txn_id, label_target, is_fraud, fraud_type, confidence)
            VALUES (%s, 'transaction', true, 'card_not_present', 1.0)
        """, (txn_id,))

        # Duplicate label on same txn_id must violate uq_ground_truth_txn
        with pytest.raises(UniqueViolation) as exc_info:
            cur.execute("""
                INSERT INTO ground_truth_labels (txn_id, label_target, is_fraud, fraud_type, confidence)
                VALUES (%s, 'transaction', false, 'legitimate_clean', 1.0)
            """, (txn_id,))
        conn.rollback()
        assert "uq_ground_truth_txn" in str(exc_info.value)


def test_timeless_account_label_uniqueness(isolated_test_db):
    """Verify uq_ground_truth_account_timeless prevents duplicate timeless labels of same typology."""
    cfg = isolated_test_db.alembic_cfg
    command.upgrade(cfg, "m2_0003")

    conn = isolated_test_db.conn
    with conn.cursor() as cur:
        _, acc_id = _seed_account(cur)

        # 1. Insert timeless first_party_synthetic label
        cur.execute("""
            INSERT INTO ground_truth_labels (
                account_id, label_target, is_fraud, fraud_type, confidence, incident_start_at, incident_end_at
            ) VALUES (%s, 'account', true, 'first_party_synthetic', 0.95, NULL, NULL)
        """, (acc_id,))
        conn.commit()

        # 2. Duplicate timeless label with same typology -> UniqueViolation
        with pytest.raises(UniqueViolation) as exc_info:
            cur.execute("""
                INSERT INTO ground_truth_labels (
                    account_id, label_target, is_fraud, fraud_type, confidence, incident_start_at, incident_end_at
                ) VALUES (%s, 'account', true, 'first_party_synthetic', 0.99, NULL, NULL)
            """, (acc_id,))
        conn.rollback()
        assert "uq_ground_truth_account_timeless" in str(exc_info.value)

        # 3. Different typology on same account succeeds
        cur.execute("""
            INSERT INTO ground_truth_labels (
                account_id, label_target, is_fraud, fraud_type, confidence, incident_start_at, incident_end_at
            ) VALUES (%s, 'account', true, 'bust_out', 0.90, NULL, NULL)
        """, (acc_id,))
        conn.commit()


def test_bounded_incident_uniqueness(isolated_test_db):
    """Verify uq_ground_truth_account_incident prevents duplicate incidents with identical start time."""
    cfg = isolated_test_db.alembic_cfg
    command.upgrade(cfg, "m2_0003")

    conn = isolated_test_db.conn
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=20)
    end = now - timedelta(days=10)

    with conn.cursor() as cur:
        _, acc_id = _seed_account(cur)

        # 1. Insert bounded incident
        cur.execute("""
            INSERT INTO ground_truth_labels (
                account_id, label_target, is_fraud, fraud_type, confidence,
                incident_start_at, incident_end_at, labeled_at
            ) VALUES (%s, 'account', true, 'account_takeover', 0.95, %s, %s, %s)
        """, (acc_id, start, end, now))

        # 2. Duplicate incident with exact same account_id, fraud_type, and incident_start_at -> UniqueViolation
        with pytest.raises(UniqueViolation) as exc_info:
            cur.execute("""
                INSERT INTO ground_truth_labels (
                    account_id, label_target, is_fraud, fraud_type, confidence,
                    incident_start_at, incident_end_at, labeled_at
                ) VALUES (%s, 'account', true, 'account_takeover', 0.99, %s, %s, %s)
            """, (acc_id, start, end + timedelta(days=2), now))
        conn.rollback()
        assert "uq_ground_truth_account_incident" in str(exc_info.value)


def test_different_incident_periods_same_account_and_typology(isolated_test_db):
    """Verify multiple distinct incidents of the same typology on an account are supported."""
    cfg = isolated_test_db.alembic_cfg
    command.upgrade(cfg, "m2_0003")

    conn = isolated_test_db.conn
    now = datetime.now(timezone.utc)
    inc1_start = now - timedelta(days=60)
    inc1_end = now - timedelta(days=50)

    inc2_start = now - timedelta(days=30)
    inc2_end = now - timedelta(days=20)

    with conn.cursor() as cur:
        _, acc_id = _seed_account(cur)

        # Incident 1 (ATO in January)
        cur.execute("""
            INSERT INTO ground_truth_labels (
                account_id, label_target, is_fraud, fraud_type, confidence,
                incident_start_at, incident_end_at, labeled_at
            ) VALUES (%s, 'account', true, 'account_takeover', 0.95, %s, %s, %s)
            RETURNING label_id;
        """, (acc_id, inc1_start, inc1_end, now))
        row1 = cur.fetchone()

        # Incident 2 (ATO in March) - same account, same typology, different start -> SUCCEEDS
        cur.execute("""
            INSERT INTO ground_truth_labels (
                account_id, label_target, is_fraud, fraud_type, confidence,
                incident_start_at, incident_end_at, labeled_at
            ) VALUES (%s, 'account', true, 'account_takeover', 0.90, %s, %s, %s)
            RETURNING label_id;
        """, (acc_id, inc2_start, inc2_end, now))
        row2 = cur.fetchone()
        conn.commit()

    assert row1 is not None and row2 is not None
    assert row1["label_id"] != row2["label_id"]


def test_required_indexes_definitions(isolated_test_db):
    """Verify all required indexes exist with expected uniqueness and partial predicates."""
    cfg = isolated_test_db.alembic_cfg
    command.upgrade(cfg, "m2_0003")

    conn = isolated_test_db.conn
    with conn.cursor() as cur:
        cur.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'ground_truth_labels';
        """)
        indexes = {row["indexname"]: row["indexdef"] for row in cur.fetchall()}
        conn.commit()

    # Legacy index must be removed
    assert "uq_ground_truth_account" not in indexes

    # Timeless unique index
    assert "uq_ground_truth_account_timeless" in indexes
    timeless_def = indexes["uq_ground_truth_account_timeless"]
    assert "UNIQUE INDEX" in timeless_def
    assert "incident_start_at IS NULL" in timeless_def

    # Incident unique index
    assert "uq_ground_truth_account_incident" in indexes
    incident_def = indexes["uq_ground_truth_account_incident"]
    assert "UNIQUE INDEX" in incident_def
    assert "incident_start_at IS NOT NULL" in incident_def

    # Incident window lookup index
    assert "idx_ground_truth_incident_window" in indexes
    window_def = indexes["idx_ground_truth_incident_window"]
    assert "UNIQUE" not in window_def
    assert "account_id" in window_def and "incident_start_at" in window_def and "incident_end_at" in window_def

    # Existing indexes preserved
    assert "uq_ground_truth_txn" in indexes
    assert "idx_ground_truth_account" in indexes
    assert "idx_ground_truth_labeled_at" in indexes
    assert "idx_ground_truth_fraud" in indexes


def test_existing_rows_preserved_across_migration(isolated_test_db):
    """Verify pre-existing labels from m2_0002 are preserved with NULL boundaries."""
    conn = isolated_test_db.conn
    cfg = isolated_test_db.alembic_cfg

    # Insert data at m2_0002
    with conn.cursor() as cur:
        _, acc_id = _seed_account(cur)
        txn_id = _seed_transaction(cur, acc_id)

        cur.execute("""
            INSERT INTO ground_truth_labels (
                txn_id, label_target, is_fraud, fraud_type, confidence, source, notes
            ) VALUES (%s, 'transaction', true, 'card_not_present', 0.95, 'analyst', 'Pre-existing txn label')
            RETURNING label_id;
        """, (txn_id,))
        txn_lbl_id = cur.fetchone()["label_id"]

        cur.execute("""
            INSERT INTO ground_truth_labels (
                account_id, label_target, is_fraud, fraud_type, confidence, source, notes
            ) VALUES (%s, 'account', true, 'first_party_synthetic', 0.88, 'seed', 'Pre-existing acc label')
            RETURNING label_id;
        """, (acc_id,))
        acc_lbl_id = cur.fetchone()["label_id"]
        conn.commit()

    # Perform upgrade to m2_0003
    command.upgrade(cfg, "m2_0003")

    # Verify rows exist and boundary columns are NULL
    with conn.cursor() as cur:
        cur.execute("""
            SELECT label_id, incident_start_at, incident_end_at, notes
            FROM ground_truth_labels
            WHERE label_id IN (%s, %s);
        """, (txn_lbl_id, acc_lbl_id))
        rows = {row["label_id"]: row for row in cur.fetchall()}
        conn.commit()

    assert txn_lbl_id in rows
    assert rows[txn_lbl_id]["incident_start_at"] is None
    assert rows[txn_lbl_id]["incident_end_at"] is None
    assert rows[txn_lbl_id]["notes"] == "Pre-existing txn label"

    assert acc_lbl_id in rows
    assert rows[acc_lbl_id]["incident_start_at"] is None
    assert rows[acc_lbl_id]["incident_end_at"] is None
    assert rows[acc_lbl_id]["notes"] == "Pre-existing acc label"


def test_downgrade_guard_blocks_duplicate_incidents(isolated_test_db):
    """Verify downgrade guard aborts if multiple incidents exist for the same account and typology."""
    cfg = isolated_test_db.alembic_cfg
    command.upgrade(cfg, "m2_0003")

    conn = isolated_test_db.conn
    now = datetime.now(timezone.utc)

    with conn.cursor() as cur:
        _, acc_id = _seed_account(cur)
        # Seed 2 distinct incidents of same typology on same account
        cur.execute("""
            INSERT INTO ground_truth_labels (
                account_id, label_target, is_fraud, fraud_type, confidence,
                incident_start_at, incident_end_at, labeled_at
            ) VALUES (%s, 'account', true, 'bust_out', 1.0, %s, %s, %s);
        """, (acc_id, now - timedelta(days=20), now - timedelta(days=15), now))

        cur.execute("""
            INSERT INTO ground_truth_labels (
                account_id, label_target, is_fraud, fraud_type, confidence,
                incident_start_at, incident_end_at, labeled_at
            ) VALUES (%s, 'account', true, 'bust_out', 1.0, %s, %s, %s);
        """, (acc_id, now - timedelta(days=10), now - timedelta(days=5), now))
        conn.commit()

    # Attempting downgrade to m2_0002 must abort due to duplicate incident guard
    with pytest.raises(Exception) as exc_info:
        command.downgrade(cfg, "m2_0002")
    assert "multiple incidents of the same fraud typology" in str(exc_info.value)

    # Confirm revision is still m2_0003
    engine = sa.create_engine(isolated_test_db.alembic_url)
    with engine.connect() as check_conn:
        version = check_conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar()
    engine.dispose()
    assert version == "m2_0003"


def test_clean_downgrade_reverts_schema_and_restores_index(isolated_test_db):
    """Verify clean downgrade succeeds when no multi-incident duplicates exist and restores m2_0002."""
    cfg = isolated_test_db.alembic_cfg
    command.upgrade(cfg, "m2_0003")

    conn = isolated_test_db.conn
    # Downgrade to m2_0002
    command.downgrade(cfg, "m2_0002")

    # Confirm revision is m2_0002
    engine = sa.create_engine(isolated_test_db.alembic_url)
    with engine.connect() as check_conn:
        version = check_conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar()
    engine.dispose()
    assert version == "m2_0002"

    # Verify columns are dropped and legacy index is restored
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'ground_truth_labels'
              AND column_name IN ('incident_start_at', 'incident_end_at');
        """)
        assert cur.fetchall() == []

        cur.execute("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'ground_truth_labels';
        """)
        indexes = {row["indexname"] for row in cur.fetchall()}
        assert "uq_ground_truth_account" in indexes
        assert "uq_ground_truth_account_timeless" not in indexes
        assert "uq_ground_truth_account_incident" not in indexes
        assert "idx_ground_truth_incident_window" not in indexes
        conn.commit()
