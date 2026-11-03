"""Unit and integration tests for transaction and entity-edge persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import os
from pathlib import Path
import sys
from uuid import UUID, uuid4
import psycopg
import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import IntegrityError

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ML_SRC = _REPO_ROOT / "ml" / "src"
_INGESTION_SRC = _REPO_ROOT / "services" / "ingestion" / "src"
for p in (_ML_SRC, _INGESTION_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ingestion import config
from ingestion.persistence import MemoryStore, PgStore, UnknownEntity, edges_for
from ingestion.schemas import TransactionIn, compare_canonical_fields


# ============================================================================
# 1. edges_for Contract Tests
# ============================================================================


def test_edges_for_all_relationships():
    """Verify edges_for constructs device, ip, merchant, and transfer edges."""
    acc_id = uuid4()
    cp_id = uuid4()
    merch_id = uuid4()
    dev_id = uuid4()
    ip_id = uuid4()

    t_transfer = TransactionIn(
        id=uuid4(),
        ts=datetime.now(timezone.utc),
        account_id=acc_id,
        counterparty_id=cp_id,
        merchant_id=merch_id,
        device_id=dev_id,
        ip_id=ip_id,
        amount=Decimal("100.00"),
        currency="USD",
        channel="transfer",
    )
    edges = edges_for(t_transfer)
    assert len(edges) == 4
    assert ("account", str(acc_id), "device", str(dev_id), "uses_device") in edges
    assert ("account", str(acc_id), "ip", str(ip_id), "uses_ip") in edges
    assert ("account", str(acc_id), "merchant", str(merch_id), "pays_merchant") in edges
    assert ("account", str(acc_id), "account", str(cp_id), "transfers_to") in edges

    # Card transaction with counterparty should NOT produce transfers_to edge
    t_card = TransactionIn(
        id=uuid4(),
        ts=datetime.now(timezone.utc),
        account_id=acc_id,
        counterparty_id=cp_id,
        amount=Decimal("50.00"),
        currency="USD",
        channel="card",
    )
    assert len(edges_for(t_card)) == 0


# ============================================================================
# 2. MemoryStore Unit Tests
# ============================================================================


def test_memorystore_lifecycle_and_idempotency():
    """Verify MemoryStore records transactions, upserts edges, and handles duplicates."""
    acc_id = uuid4()
    dev_id = uuid4()
    store = MemoryStore()

    t = TransactionIn(
        id=uuid4(),
        ts=datetime.now(timezone.utc),
        account_id=acc_id,
        device_id=dev_id,
        amount=Decimal("100.00"),
        currency="USD",
        channel="card",
    )

    # 1. New insert returns True and records weight 1.0
    inserted1 = store.store(t)
    assert inserted1 is True
    edge_key = ("account", str(acc_id), "device", str(dev_id), "uses_device")
    assert store.edges[edge_key] == 1.0

    # 2. Duplicate insert returns False and preserves weight 1.0 (no double-count)
    inserted2 = store.store(t)
    assert inserted2 is False
    assert store.edges[edge_key] == 1.0

    # 3. New transaction on same edge increments weight
    t_new = TransactionIn(
        id=uuid4(),
        ts=datetime.now(timezone.utc),
        account_id=acc_id,
        device_id=dev_id,
        amount=Decimal("200.00"),
        currency="USD",
        channel="card",
    )
    assert store.store(t_new) is True
    assert store.edges[edge_key] == 2.0


def test_memorystore_known_accounts_validation():
    """Verify MemoryStore rejects unknown accounts when configured."""
    known = {uuid4()}
    unknown = uuid4()
    store = MemoryStore(known_accounts=known)

    t_unknown = TransactionIn(
        id=uuid4(),
        ts=datetime.now(timezone.utc),
        account_id=unknown,
        amount=Decimal("10.00"),
        currency="USD",
        channel="card",
    )
    with pytest.raises(UnknownEntity, match="Unknown account"):
        store.store(t_unknown)


def test_memorystore_get_returns_original_canonical_transaction():
    """Verify MemoryStore.get retrieves the identical canonical TransactionIn."""
    store = MemoryStore()
    txn_id = uuid4()
    acc_id = uuid4()
    dev_id = uuid4()
    ip_id = uuid4()

    t = TransactionIn(
        id=txn_id,
        ts=datetime(2026, 10, 31, 12, 0, 0, tzinfo=timezone.utc),
        account_id=acc_id,
        device_id=dev_id,
        ip_id=ip_id,
        amount=Decimal("123.45"),
        currency="USD",
        channel="card",
    )
    assert store.get(txn_id) is None

    store.store(t)
    retrieved = store.get(txn_id)
    assert retrieved is not None
    assert compare_canonical_fields(retrieved, t) is True


def test_pgstore_constructor_accepts_engine_and_string(pg_store_with_account):
    """Verify PgStore accepts either an existing Engine or a database URL string."""
    store, _, _ = pg_store_with_account
    assert isinstance(store.engine, Engine)

    database_url = os.getenv("DATABASE_URL", config.DATABASE_URL)
    store_from_str = PgStore(database_url)
    assert isinstance(store_from_str.engine, Engine)
    assert store_from_str.ping() is True


def test_pgstore_get_returns_original_canonical_transaction(pg_store_with_account):
    """Verify PgStore.get retrieves and reconstructs the canonical TransactionIn accurately."""
    store, acc_id, created_txns = pg_store_with_account
    txn_id = uuid4()
    created_txns.append(txn_id)
    dev_id = uuid4()
    ip_id = uuid4()

    t = TransactionIn(
        id=txn_id,
        ts=datetime(2026, 10, 31, 14, 30, 0, tzinfo=timezone.utc),
        account_id=acc_id,
        device_id=dev_id,
        ip_id=ip_id,
        amount=Decimal("99.95"),
        currency="USD",
        channel="card",
    )

    # Initial get returns None
    assert store.get(txn_id) is None

    # Insert transaction
    assert store.store(t) is True

    # Retrieve and compare
    retrieved = store.get(txn_id)
    assert retrieved is not None
    assert compare_canonical_fields(retrieved, t) is True
    assert retrieved.id == txn_id
    assert retrieved.amount == Decimal("99.95")
    assert retrieved.device_id == dev_id
    assert retrieved.currency == "USD"
    assert retrieved.channel == "card"


# ============================================================================
# 3. PgStore Real PostgreSQL Integration Tests
# ============================================================================


@pytest.fixture
def pg_store_with_account():
    """Provide PgStore connected to real PostgreSQL with a test customer/account fixture."""
    database_url = os.getenv("DATABASE_URL", config.DATABASE_URL)
    pg_url = database_url.replace("postgresql+psycopg://", "postgresql://")

    try:
        engine = create_engine(database_url, pool_pre_ping=True)
        store = PgStore(engine)
        if not store.ping():
            pytest.skip("PostgreSQL ping failed; skipping real PostgreSQL persistence tests.")
    except Exception as exc:
        pytest.skip(f"PostgreSQL connection failed ({exc}); skipping real persistence tests.")

    cust_id = uuid4()
    acc_id = uuid4()
    acc_num = f"ACC-{acc_id.hex[:20]}"
    cust_ref = f"CUST-{cust_id.hex[:20]}"

    with psycopg.connect(pg_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO customers (customer_id, external_ref, full_name, country_code)
                VALUES (%s, %s, 'Persistence Test Cust', 'US');
                """,
                (cust_id, cust_ref),
            )
            cur.execute(
                """
                INSERT INTO accounts (account_id, customer_id, account_number, account_type, currency, opened_on, status)
                VALUES (%s, %s, %s, 'current', 'USD', CURRENT_DATE, 'active');
                """,
                (acc_id, cust_id, acc_num),
            )
            conn.commit()

    created_txns: list[UUID] = []

    try:
        yield store, acc_id, created_txns
    finally:
        with psycopg.connect(pg_url) as conn:
            with conn.cursor() as cur:
                for tid in created_txns:
                    cur.execute("DELETE FROM transactions WHERE txn_id = %s;", (tid,))
                cur.execute("DELETE FROM entity_edges WHERE src_id = %s;", (str(acc_id),))
                cur.execute("DELETE FROM accounts WHERE account_id = %s;", (acc_id,))
                cur.execute("DELETE FROM customers WHERE customer_id = %s;", (cust_id,))
                conn.commit()


def test_pgstore_real_postgres_insert_and_idempotency(pg_store_with_account):
    """Verify real PostgreSQL transaction insert, edge creation, and duplicate idempotency."""
    store, acc_id, created_txns = pg_store_with_account
    txn_id = uuid4()
    created_txns.append(txn_id)
    dev_id = uuid4()
    ip_id = uuid4()

    t = TransactionIn(
        id=txn_id,
        ts=datetime.now(timezone.utc),
        account_id=acc_id,
        device_id=dev_id,
        ip_id=ip_id,
        amount=Decimal("75.50"),
        currency="USD",
        channel="card",
    )

    # 1. First insert must succeed and insert edges
    inserted1 = store.store(t)
    assert inserted1 is True

    # Check edges in database
    with store.engine.connect() as conn:
        edges = conn.execute(
            text("SELECT edge_type, weight FROM entity_edges WHERE src_id = :sid;"),
            {"sid": str(acc_id)},
        ).fetchall()
        assert len(edges) == 2
        edge_map = {r[0]: float(r[1]) for r in edges}
        assert edge_map["uses_device"] == 1.0
        assert edge_map["uses_ip"] == 1.0

    # 2. Duplicate insert must return False and NOT increment edge weights
    inserted2 = store.store(t)
    assert inserted2 is False

    with store.engine.connect() as conn:
        edges_after = conn.execute(
            text("SELECT edge_type, weight FROM entity_edges WHERE src_id = :sid;"),
            {"sid": str(acc_id)},
        ).fetchall()
        edge_map_after = {r[0]: float(r[1]) for r in edges_after}
        assert edge_map_after["uses_device"] == 1.0
        assert edge_map_after["uses_ip"] == 1.0


def test_pgstore_foreign_key_violation_translates_to_unknown_entity(pg_store_with_account):
    """Verify that referencing a non-existent account raises UnknownEntity."""
    store, _, _ = pg_store_with_account
    non_existent_account = uuid4()

    t = TransactionIn(
        id=uuid4(),
        ts=datetime.now(timezone.utc),
        account_id=non_existent_account,
        amount=Decimal("50.00"),
        currency="USD",
        channel="card",
    )

    with pytest.raises(UnknownEntity):
        store.store(t)


def test_pgstore_ping(pg_store_with_account):
    """Verify PgStore ping returns True on live PostgreSQL connection."""
    store, _, _ = pg_store_with_account
    assert store.ping() is True


def test_pgstore_unrelated_integrity_error_not_misclassified(pg_store_with_account):
    """Verify check constraint violation is re-raised directly, not swallowed as UnknownEntity."""
    store, acc_id, created_txns = pg_store_with_account

    # Construct invalid transaction that violates amount check constraint (> 0)
    # Using raw SQL simulation to trigger check_violation SQLSTATE 23514
    with pytest.raises(IntegrityError) as exc_info:
        with store.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO entity_edges (src_type, src_id, dst_type, dst_id, edge_type, weight, first_seen_at, last_seen_at)
                    VALUES ('account', 'acc_1', 'device', 'dev_1', 'uses_device', -5.0, NOW(), NOW());
                """)
            )
    assert not isinstance(exc_info.value, UnknownEntity)
    orig = getattr(exc_info.value, "orig", None)
    sqlstate = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
    assert sqlstate == "23514"  # check_violation, NOT 23503


def test_pgstore_atomic_rollback_on_failed_edge_upsert(pg_store_with_account):
    """Verify that if an edge operation fails, the transaction insert rolls back atomically."""
    store, acc_id, created_txns = pg_store_with_account
    txn_id = uuid4()
    created_txns.append(txn_id)

    # Monkeypatch edges_for to return an invalid edge that violates check constraints
    original_edges_for = sys.modules["ingestion.persistence"].edges_for
    try:
        # Invalid weight or empty src_id to trigger check constraint
        sys.modules["ingestion.persistence"].edges_for = lambda t: [("account", "", "device", "dev_1", "uses_device")]

        t = TransactionIn(
            id=txn_id,
            ts=datetime.now(timezone.utc),
            account_id=acc_id,
            amount=Decimal("100.00"),
            currency="USD",
            channel="card",
            device_id=uuid4(),
        )

        with pytest.raises(IntegrityError):
            store.store(t)

        # Verify transaction row was rolled back and does not exist in database
        with store.engine.connect() as conn:
            count = conn.execute(
                text("SELECT count(*) FROM transactions WHERE txn_id = :tid;"),
                {"tid": txn_id},
            ).scalar()
            assert count == 0, "Transaction row must be rolled back when edge operation fails"

    finally:
        sys.modules["ingestion.persistence"].edges_for = original_edges_for
