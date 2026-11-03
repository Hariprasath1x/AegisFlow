"""Integration and safety tests for Alembic migration m2_0002 (Transaction Schema Alignment)."""

import os
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
import pytest
from alembic import command
from alembic.config import Config
import sqlalchemy as sa
from sqlalchemy import create_engine, text

from ingestion import config


DATABASE_URL = os.getenv("DATABASE_URL", config.DATABASE_URL)
PG_URL = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
ADMIN_URL = os.getenv("ADMIN_DATABASE_URL", PG_URL.rsplit("/", 1)[0] + "/postgres")
ALEMBIC_INI_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../alembic.ini")
)


@pytest.fixture(scope="function")
def db_conn():
    try:
        conn = psycopg.connect(PG_URL, row_factory=dict_row)
        conn.autocommit = False
        yield conn
        conn.rollback()
        conn.close()
    except Exception as e:
        pytest.skip(f"PostgreSQL database unavailable: {e}")


def _validate_disposable_db_name(name: str) -> None:
    """Explicit runtime guard ensuring target database is an isolated disposable test database.

    Raises RuntimeError if name does not match expected disposable prefix or equals 'aegisflow'.
    """
    if not isinstance(name, str) or not name.startswith("aegisflow_test_m2_0002_"):
        raise RuntimeError(
            f"Safety check failed: database name {name!r} does not have required prefix 'aegisflow_test_m2_0002_'"
        )
    if name == "aegisflow":
        raise RuntimeError("Safety check failed: database name cannot be development database 'aegisflow'")


@pytest.fixture(scope="function")
def isolated_test_db():
    """Create a completely isolated, disposable PostgreSQL database for destructive migration tests.

    Bootstrapped with scripts/init_db.sql and upgraded to Alembic head (m2_0002).
    Completely isolated from the shared development database.
    Dropped in teardown using DROP DATABASE ... WITH (FORCE).
    """
    try:
        admin_conn = psycopg.connect(ADMIN_URL, autocommit=True)
    except Exception as exc:
        pytest.skip(f"PostgreSQL admin database unavailable for isolated test database: {exc}")

    db_id = uuid4().hex[:8]
    disposable_db = f"aegisflow_test_m2_0002_{db_id}"

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

    # Connect to disposable DB for test usage
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
            # Explicit runtime guard immediately before DROP DATABASE
            _validate_disposable_db_name(disposable_db)
            admin_conn.execute(
                psycopg.sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE);").format(
                    psycopg.sql.Identifier(disposable_db)
                )
            )
        except Exception:
            pass
        admin_conn.close()


def get_alembic_config():
    cfg = Config(ALEMBIC_INI_PATH)
    cfg.set_main_option("sqlalchemy.url", DATABASE_URL)
    return cfg


def test_m2_0002_columns_and_types(db_conn):
    """Verify that destination_account_id, merchant_id, ip_id exist with expected types and nullability."""
    with db_conn.cursor() as cur:
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'transactions'
              AND column_name IN ('destination_account_id', 'merchant_id', 'ip_id', 'direction', 'occurred_at');
        """)
        cols = {row["column_name"]: row for row in cur.fetchall()}

        assert "destination_account_id" in cols
        assert cols["destination_account_id"]["data_type"] == "uuid"
        assert cols["destination_account_id"]["is_nullable"] == "YES"

        assert "merchant_id" in cols
        assert cols["merchant_id"]["data_type"] == "uuid"
        assert cols["merchant_id"]["is_nullable"] == "YES"

        assert "ip_id" in cols
        assert cols["ip_id"]["data_type"] == "uuid"
        assert cols["ip_id"]["is_nullable"] == "YES"

        # Direction must now be nullable
        assert cols["direction"]["is_nullable"] == "YES"

        # occurred_at must be timestamp with time zone (TIMESTAMPTZ)
        assert cols["occurred_at"]["data_type"] == "timestamp with time zone"
        assert cols["occurred_at"]["is_nullable"] == "NO"


def test_m2_0002_foreign_keys_and_constraints(db_conn):
    """Verify foreign keys and check constraints on transactions table."""
    with db_conn.cursor() as cur:
        cur.execute("""
            SELECT conname, contype, pg_get_constraintdef(oid) as def
            FROM pg_constraint
            WHERE conrelid = 'transactions'::regclass;
        """)
        constraints = {row["conname"]: row for row in cur.fetchall()}

        # Foreign keys
        assert "fk_transactions_destination_account" in constraints
        assert "REFERENCES accounts(account_id)" in constraints["fk_transactions_destination_account"]["def"]

        # Beneficiary FK preserved
        assert "transactions_counterparty_id_fkey" in constraints
        assert "REFERENCES counterparties(counterparty_id)" in constraints["transactions_counterparty_id_fkey"]["def"]

        # Channel check constraint
        assert "ck_transactions_channel" in constraints
        ch_def = constraints["ck_transactions_channel"]["def"]
        for expected_channel in ["card", "wire", "upi", "ach", "atm", "transfer", "wallet"]:
            assert expected_channel in ch_def

        # Direction check constraint allows NULL or debit/credit
        assert "ck_transactions_direction" in constraints
        dir_def = constraints["ck_transactions_direction"]["def"]
        assert "direction IS NULL" in dir_def
        assert "debit" in dir_def and "credit" in dir_def

        # Transfer target constraint
        assert "ck_transactions_transfer_target" in constraints
        tt_def = constraints["ck_transactions_transfer_target"]["def"]
        assert "transfer" in tt_def
        assert "destination_account_id IS NOT NULL" in tt_def


def test_m2_0002_partial_indexes(db_conn):
    """Verify partial indexes exist on destination_account_id, merchant_id, and ip_id."""
    with db_conn.cursor() as cur:
        cur.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'transactions';
        """)
        indexes = {row["indexname"]: row["indexdef"] for row in cur.fetchall()}

        assert "idx_txn_dest_account" in indexes
        assert "WHERE (destination_account_id IS NOT NULL)" in indexes["idx_txn_dest_account"]

        assert "idx_txn_merchant_id" in indexes
        assert "WHERE (merchant_id IS NOT NULL)" in indexes["idx_txn_merchant_id"]

        assert "idx_txn_ip_id" in indexes
        assert "WHERE (ip_id IS NOT NULL)" in indexes["idx_txn_ip_id"]


def test_transfer_target_constraint_enforcement(db_conn):
    """Verify ck_transactions_transfer_target enforces destination rules."""
    cust_id = uuid4()
    acc_1 = uuid4()
    acc_2 = uuid4()

    with db_conn.cursor() as cur:
        # Seed customer and accounts
        cur.execute(
            "INSERT INTO customers (customer_id, external_ref, full_name, country_code) VALUES (%s, %s, 'C', 'IN')",
            (cust_id, f"C-{cust_id.hex[:10]}")
        )
        cur.execute(
            "INSERT INTO accounts (account_id, customer_id, account_number, account_type, opened_on) VALUES (%s, %s, %s, 'savings', CURRENT_DATE)",
            (acc_1, cust_id, f"A1-{acc_1.hex[:10]}")
        )
        cur.execute(
            "INSERT INTO accounts (account_id, customer_id, account_number, account_type, opened_on) VALUES (%s, %s, %s, 'savings', CURRENT_DATE)",
            (acc_2, cust_id, f"A2-{acc_2.hex[:10]}")
        )

        # 1. Valid transfer: channel='transfer', destination_account_id set, counterparty_id NULL -> succeeds
        cur.execute("""
            INSERT INTO transactions (
                txn_id, occurred_at, account_id, destination_account_id, counterparty_id,
                amount, currency, channel, external_ref
            ) VALUES (%s, now(), %s, %s, NULL, 50, 'INR', 'transfer', %s)
        """, (uuid4(), acc_1, acc_2, f"REF-{uuid4().hex[:10]}"))

        # 2. Invalid transfer: channel='transfer' with NULL destination -> violates constraint
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute("""
                INSERT INTO transactions (
                    txn_id, occurred_at, account_id, destination_account_id, counterparty_id,
                    amount, currency, channel, external_ref
                ) VALUES (%s, now(), %s, NULL, NULL, 50, 'INR', 'transfer', %s)
            """, (uuid4(), acc_1, f"REF-{uuid4().hex[:10]}"))
        db_conn.rollback()

        # 3. Invalid non-transfer: channel='card' with destination_account_id set -> violates constraint
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute("""
                INSERT INTO transactions (
                    txn_id, occurred_at, account_id, destination_account_id, counterparty_id,
                    amount, currency, channel, external_ref
                ) VALUES (%s, now(), %s, %s, NULL, 50, 'INR', 'card', %s)
            """, (uuid4(), acc_1, acc_2, f"REF-{uuid4().hex[:10]}"))
        db_conn.rollback()


def test_downgrade_guard_d2_destination_account_id(isolated_test_db):
    """D2: Actual Alembic downgrade is blocked when any row has destination_account_id IS NOT NULL.

    Runs entirely within an isolated, disposable test database.
    Never modifies, locks, or drops constraints on the shared development database.

    - Exercises the real command.downgrade() path against the live migration.
    - Confirms RuntimeError is raised with 'destination_account_id' in the message.
    - Confirms database remains at m2_0002 after the failed downgrade.
    - Confirms the transaction row is unchanged after the blocked downgrade.
    - Confirms all M2 schema objects (column, FK, partial index, check constraint) remain intact.
    - Cleans up its fixture and restores/validates the check constraint before disposal.
    """
    cfg = isolated_test_db.alembic_cfg
    db_conn = isolated_test_db.conn
    test_db_url = isolated_test_db.url

    cust_id = uuid4()
    acc_id = uuid4()
    acc_dest = uuid4()
    txn_id = uuid4()

    # Seed customer/account fixture data in the isolated database
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO customers (customer_id, external_ref, full_name, country_code) VALUES (%s, %s, 'D2 Cust', 'IN')",
            (cust_id, f"D2-{cust_id.hex[:10]}")
        )
        cur.execute(
            "INSERT INTO accounts (account_id, customer_id, account_number, account_type, opened_on) VALUES (%s, %s, %s, 'savings', CURRENT_DATE)",
            (acc_id, cust_id, f"D2A-{acc_id.hex[:10]}")
        )
        cur.execute(
            "INSERT INTO accounts (account_id, customer_id, account_number, account_type, opened_on) VALUES (%s, %s, %s, 'savings', CURRENT_DATE)",
            (acc_dest, cust_id, f"D2B-{acc_dest.hex[:10]}")
        )
        db_conn.commit()

    # Dedicated autocommit connection to the isolated test database for DDL
    ddl_conn = psycopg.connect(test_db_url, autocommit=True)
    ddl_conn.execute("SET lock_timeout = '5s'")
    try:
        # Ensure db_conn has no open transaction before DDL
        db_conn.rollback()

        # Temporarily drop the transfer-target check constraint in the isolated database
        ddl_conn.execute(
            "ALTER TABLE transactions DROP CONSTRAINT IF EXISTS ck_transactions_transfer_target"
        )

        # Insert D2-triggering row: channel='card' passes D1, destination_account_id IS NOT NULL triggers D2
        with db_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO transactions (
                    txn_id, occurred_at, account_id, destination_account_id,
                    amount, currency, channel, external_ref, direction
                ) VALUES (%s, now(), %s, %s, 100, 'INR', 'card', %s, 'debit')
            """, (txn_id, acc_id, acc_dest, f"D2REF-{txn_id.hex[:10]}"))
            db_conn.commit()

        # Restore the constraint as NOT VALID in the isolated database
        ddl_conn.execute("""
            ALTER TABLE transactions ADD CONSTRAINT ck_transactions_transfer_target CHECK (
                (channel = 'transfer' AND destination_account_id IS NOT NULL AND counterparty_id IS NULL)
                OR
                (channel != 'transfer' AND destination_account_id IS NULL)
            ) NOT VALID
        """)

        # Confirm destination_account_id is populated in the row
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT destination_account_id, amount FROM transactions WHERE txn_id = %s", (txn_id,)
            )
            row = cur.fetchone()
            db_conn.commit()
        assert row is not None
        assert row["destination_account_id"] == acc_dest
        assert row["amount"] == Decimal("100")

        # Invoke the actual Alembic downgrade — must be blocked by D2 guard
        with pytest.raises(Exception) as exc_info:
            command.downgrade(cfg, "m2_0001")
        assert "destination_account_id" in str(exc_info.value)

        # Confirm isolated database is still at m2_0002 (downgrade was aborted)
        engine = sa.create_engine(isolated_test_db.alembic_url)
        with engine.connect() as conn:
            version = conn.execute(sa.text(
                "SELECT version_num FROM alembic_version"
            )).scalar()
        engine.dispose()
        assert version == "m2_0002", f"Expected m2_0002 but found {version!r}"

        # Confirm all M2 schema objects remain intact in the isolated database
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'transactions'
                  AND column_name = 'destination_account_id';
            """)
            assert cur.fetchone() is not None, "destination_account_id column was dropped"

            cur.execute("""
                SELECT conname FROM pg_constraint
                WHERE conrelid = 'transactions'::regclass
                  AND conname = 'fk_transactions_destination_account';
            """)
            assert cur.fetchone() is not None, "fk_transactions_destination_account was dropped"

            cur.execute("""
                SELECT indexname FROM pg_indexes
                WHERE tablename = 'transactions' AND indexname = 'idx_txn_dest_account';
            """)
            assert cur.fetchone() is not None, "idx_txn_dest_account was dropped"

            cur.execute("""
                SELECT conname FROM pg_constraint
                WHERE conrelid = 'transactions'::regclass
                  AND conname = 'ck_transactions_transfer_target';
            """)
            assert cur.fetchone() is not None, "ck_transactions_transfer_target was dropped"
            db_conn.commit()

        # Confirm transaction row is unchanged after the blocked downgrade
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT destination_account_id, amount FROM transactions WHERE txn_id = %s", (txn_id,)
            )
            row_after = cur.fetchone()
            db_conn.commit()
        assert row_after is not None
        assert row_after["destination_account_id"] == acc_dest
        assert row_after["amount"] == Decimal("100")

    finally:
        # Ensure db_conn has no open transaction before DDL
        db_conn.rollback()

        # Clean up test row first in the isolated database
        with psycopg.connect(test_db_url, autocommit=True) as cleanup_conn:
            with cleanup_conn.cursor() as cur:
                cur.execute("DELETE FROM transactions WHERE txn_id = %s", (txn_id,))
                cur.execute("DELETE FROM accounts WHERE customer_id = %s", (cust_id,))
                cur.execute("DELETE FROM customers WHERE customer_id = %s", (cust_id,))

        # Restore and validate canonical constraint in the isolated database
        ddl_conn.execute(
            "ALTER TABLE transactions DROP CONSTRAINT IF EXISTS ck_transactions_transfer_target"
        )
        ddl_conn.execute("""
            ALTER TABLE transactions ADD CONSTRAINT ck_transactions_transfer_target CHECK (
                (channel = 'transfer' AND destination_account_id IS NOT NULL AND counterparty_id IS NULL)
                OR
                (channel != 'transfer' AND destination_account_id IS NULL)
            )
        """)
        # Verify it is validated
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT convalidated FROM pg_constraint
                WHERE conrelid = 'transactions'::regclass
                  AND conname = 'ck_transactions_transfer_target';
            """)
            validated = cur.fetchone()
            assert validated is not None and validated["convalidated"] is True
            db_conn.commit()
        ddl_conn.close()


def test_downgrade_guards_sequential_enforcement(isolated_test_db):
    """Verify downgrade guards D1-D6 sequentially refuse downgrade on incompatible data and allow clean downgrade.

    Runs entirely within an isolated, disposable test database.
    Never modifies, downgrades, or locks the shared development database.
    """
    cfg = isolated_test_db.alembic_cfg
    db_conn = isolated_test_db.conn

    cust_id = uuid4()
    acc_id = uuid4()
    acc_2 = uuid4()

    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO customers (customer_id, external_ref, full_name, country_code) VALUES (%s, %s, 'Guard Cust', 'IN')",
            (cust_id, f"GC-{cust_id.hex[:10]}")
        )
        cur.execute(
            "INSERT INTO accounts (account_id, customer_id, account_number, account_type, opened_on) VALUES (%s, %s, %s, 'savings', CURRENT_DATE)",
            (acc_id, cust_id, f"GA-{acc_id.hex[:10]}")
        )
        cur.execute(
            "INSERT INTO accounts (account_id, customer_id, account_number, account_type, opened_on) VALUES (%s, %s, %s, 'savings', CURRENT_DATE)",
            (acc_2, cust_id, f"GA2-{acc_2.hex[:10]}")
        )
        db_conn.commit()

    try:
        # D1: Row with M2 channel 'transfer'
        txn_d1 = uuid4()
        with db_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO transactions (
                    txn_id, occurred_at, account_id, destination_account_id, amount,
                    currency, channel, external_ref, direction
                ) VALUES (%s, now(), %s, %s, 10, 'INR', 'transfer', %s, 'debit')
            """, (txn_d1, acc_id, acc_2, f"REF-{txn_d1.hex[:10]}"))
            db_conn.commit()

        with pytest.raises(Exception) as exc_info:
            command.downgrade(cfg, "m2_0001")
        assert "M2-only channels" in str(exc_info.value)

        # Cleanup D1 row
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM transactions WHERE txn_id = %s", (txn_d1,))
            db_conn.commit()

        # D3: Row with populated merchant_id
        txn_d3 = uuid4()
        with db_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO transactions (
                    txn_id, occurred_at, account_id, merchant_id, amount,
                    currency, channel, external_ref, direction
                ) VALUES (%s, now(), %s, %s, 10, 'INR', 'card', %s, 'debit')
            """, (txn_d3, acc_id, uuid4(), f"REF-{txn_d3.hex[:10]}"))
            db_conn.commit()

        with pytest.raises(Exception) as exc_info:
            command.downgrade(cfg, "m2_0001")
        assert "merchant_id" in str(exc_info.value)

        # Cleanup D3 row
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM transactions WHERE txn_id = %s", (txn_d3,))
            db_conn.commit()

        # D4: Row with populated ip_id
        txn_d4 = uuid4()
        with db_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO transactions (
                    txn_id, occurred_at, account_id, ip_id, amount,
                    currency, channel, external_ref, direction
                ) VALUES (%s, now(), %s, %s, 10, 'INR', 'card', %s, 'debit')
            """, (txn_d4, acc_id, uuid4(), f"REF-{txn_d4.hex[:10]}"))
            db_conn.commit()

        with pytest.raises(Exception) as exc_info:
            command.downgrade(cfg, "m2_0001")
        assert "ip_id" in str(exc_info.value)

        # Cleanup D4 row
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM transactions WHERE txn_id = %s", (txn_d4,))
            db_conn.commit()

        # D5: Row with NULL direction
        txn_d5 = uuid4()
        with db_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO transactions (
                    txn_id, occurred_at, account_id, amount,
                    currency, channel, external_ref, direction
                ) VALUES (%s, now(), %s, 10, 'INR', 'card', %s, NULL)
            """, (txn_d5, acc_id, f"REF-{txn_d5.hex[:10]}"))
            db_conn.commit()

        with pytest.raises(Exception) as exc_info:
            command.downgrade(cfg, "m2_0001")
        assert "NULL direction" in str(exc_info.value)

        # Cleanup D5 row
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM transactions WHERE txn_id = %s", (txn_d5,))
            db_conn.commit()

        # D6: Clean database allows downgrade and re-upgrade
        command.downgrade(cfg, "m2_0001")

        # Verify columns removed in m2_0001 in the isolated database
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'transactions'
                  AND column_name IN ('destination_account_id', 'merchant_id', 'ip_id');
            """)
            assert len(cur.fetchall()) == 0

        # Re-upgrade to m2_0002 in the isolated database
        command.upgrade(cfg, "m2_0002")

    finally:
        # Ensure we always clean up test fixture data in the isolated database
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM transactions WHERE account_id IN (%s, %s)", (acc_id, acc_2))
            cur.execute("DELETE FROM accounts WHERE customer_id = %s", (cust_id,))
            cur.execute("DELETE FROM customers WHERE customer_id = %s", (cust_id,))
            db_conn.commit()
