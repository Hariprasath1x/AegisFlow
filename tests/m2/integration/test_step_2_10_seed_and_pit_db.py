"""Isolated disposable-PostgreSQL integration tests for Step 2.10 seeding & PIT export.

Runs exclusively against fresh disposable databases prefixed with
`aegisflow_test_m2_210_` and NEVER connects to or modifies the shared `aegisflow` DB.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit
from uuid import uuid4

from alembic import command
from alembic.config import Config
import pandas as pd
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
import pytest
import sqlalchemy as sa

from aegis_ml.features import definitions as D
from aegis_ml.features.graph_features import (
    fetch_fraud_account_ids,
    normalize_db_url,
    run_job,
)
from aegis_ml.features.offline import (
    FORBIDDEN_PIT_TARGET_COLUMNS,
    PIT_TRACE_COLUMNS,
    export_features_pit,
    main as offline_main,
)
from aegis_ml.scenarios.constants import OBSERVATION_CUTOFF
from db.seed.seed_known_fraud import (
    CANONICAL_MANIFEST_HASH,
    EXPECTED_COUNTS,
    derive_known_fraud_cutoff,
    load_canonical_manifest,
    main as seed_main,
    seed_manifest,
)
from ingestion import config as ingestion_config

ROOT = Path(__file__).resolve().parents[3]
ALEMBIC_INI_PATH = str(ROOT / "alembic.ini")
DATABASE_URL = os.getenv("DATABASE_URL", ingestion_config.DATABASE_URL)
ENGINE_URL = normalize_db_url(DATABASE_URL)
PG_URL = ENGINE_URL.replace("postgresql+psycopg://", "postgresql://", 1)
ADMIN_URL = os.getenv("ADMIN_DATABASE_URL", PG_URL.rsplit("/", 1)[0] + "/postgres")

_DISPOSABLE_DB_PATTERN = re.compile(r"^aegisflow_test_m2_210_[0-9a-f]{8}$")


def _validate_disposable_db_name(name: str) -> None:
    if not isinstance(name, str) or not _DISPOSABLE_DB_PATTERN.fullmatch(name):
        raise RuntimeError(
            f"Safety guard rejected non-disposable DB name: {name!r}"
        )
    dev_db_name = DATABASE_URL.rsplit("/", 1)[-1].split("?")[0]
    if name == dev_db_name or name == "aegisflow":
        raise RuntimeError(f"Safety guard prevented touching development DB: {name!r}")


def _verify_admin_server_safety(dev_url: str, admin_url: str, admin_conn: psycopg.Connection) -> None:
    parsed_dev = urlsplit(dev_url)
    parsed_admin = urlsplit(admin_url)
    if (parsed_dev.hostname or "localhost") != (parsed_admin.hostname or "localhost"):
        raise RuntimeError("ADMIN_DATABASE_URL host mismatches DATABASE_URL host")
    if (parsed_dev.port or 5432) != (parsed_admin.port or 5432):
        raise RuntimeError("ADMIN_DATABASE_URL port mismatches DATABASE_URL port")
    cur_db = admin_conn.execute("SELECT current_database();").fetchone()[0]
    if cur_db == "aegisflow":
        raise RuntimeError("Admin connection must not target development database 'aegisflow'")


@pytest.fixture(scope="function")
def isolated_step210_db():
    """Provision a clean, isolated PostgreSQL DB upgraded to m2_0003 and drop it on teardown."""
    try:
        admin_conn = psycopg.connect(ADMIN_URL, autocommit=True)
    except Exception as exc:
        pytest.skip(f"PostgreSQL server unavailable for isolated Step 2.10 test DB: {exc}")

    _verify_admin_server_safety(ENGINE_URL, ADMIN_URL, admin_conn)
    disposable_db = f"aegisflow_test_m2_210_{uuid4().hex[:8]}"
    _validate_disposable_db_name(disposable_db)

    try:
        admin_conn.execute(sql.SQL("CREATE DATABASE {};").format(sql.Identifier(disposable_db)))
    except Exception as exc:
        admin_conn.close()
        pytest.skip(f"Unable to create disposable test DB {disposable_db}: {exc}")

    test_pg_url = f"{PG_URL.rsplit('/', 1)[0]}/{disposable_db}"
    test_sa_url = f"postgresql+psycopg://{test_pg_url.split('://', 1)[1]}"

    try:
        init_sql = (ROOT / "scripts/init_db.sql").read_text(encoding="utf-8")
        with psycopg.connect(test_pg_url, autocommit=True) as init_conn:
            init_conn.execute(init_sql)

        cfg = Config(ALEMBIC_INI_PATH)
        cfg.set_main_option("sqlalchemy.url", test_sa_url)
        command.upgrade(cfg, "head")
    except Exception:
        admin_conn.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE);").format(sql.Identifier(disposable_db))
        )
        admin_conn.close()
        raise

    class _IsolatedDB:
        name = disposable_db
        pg_url = test_pg_url
        sa_url = test_sa_url

    db_obj = _IsolatedDB()
    try:
        yield db_obj
    finally:
        try:
            _validate_disposable_db_name(disposable_db)
            admin_conn.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE);").format(
                    sql.Identifier(disposable_db)
                )
            )
        finally:
            admin_conn.close()


def _insert_m1_demo_record(pg_url: str) -> dict[str, str]:
    """Insert a representative M1 customer/account/txn that seeding must never overwrite."""
    cid = uuid4()
    aid = uuid4()
    tid = uuid4()
    with psycopg.connect(pg_url) as conn:
        conn.execute(
            """
            INSERT INTO customers (customer_id, external_ref, full_name, country_code, kyc_status, risk_rating)
            VALUES (%s, 'AF-M1-DEMO-KEEP-001', 'M1 Preserved Customer', 'IN', 'verified', 'low');
            """,
            (cid,),
        )
        conn.execute(
            """
            INSERT INTO accounts (account_id, customer_id, account_number, account_type, currency, opened_on, status)
            VALUES (%s, %s, 'AF-M1-DEMO-ACCT-001', 'savings', 'INR', '2025-01-01', 'active');
            """,
            (aid, cid),
        )
        conn.execute(
            """
            INSERT INTO transactions (
                txn_id, external_ref, account_id, channel, direction, amount, currency,
                device_id, ip_address, occurred_at
            ) VALUES (
                %s, 'AF-M1-DEMO-TXN-001', %s, 'card', 'debit', 250.00, 'INR',
                'DEV-M1-KEEP', '203.0.113.42', '2025-02-01T12:00:00+00:00'
            );
            """,
            (tid, aid),
        )
        conn.commit()
    return {"customer_id": str(cid), "account_id": str(aid), "txn_id": str(tid)}


def test_fresh_seeding_idempotency_and_m1_preservation(isolated_step210_db) -> None:
    """Verify fresh seeding, exact counts/hash reconciliation, M1 preservation, and idempotent re-run."""
    m1_ids = _insert_m1_demo_record(isolated_step210_db.pg_url)

    first = seed_manifest(isolated_step210_db.pg_url)
    assert first["inserted"] == {
        "customers": EXPECTED_COUNTS["customers"],
        "accounts": EXPECTED_COUNTS["accounts"],
        "transactions": EXPECTED_COUNTS["transactions"],
        "edges_upserted": EXPECTED_COUNTS["edges"],
        "labels": EXPECTED_COUNTS["labels"],
    }
    assert first["reconciled"]["manifest_hash"] == CANONICAL_MANIFEST_HASH
    assert first["reconciled"]["total_edge_weight"] == 455
    assert first["pre_t0"]["cutoff"] == "2025-06-14T00:00:00Z"
    assert first["pre_t0"]["pre_t0_account_count"] == 6

    # Second run via CLI must be strict no-op without duplicating rows or inflating edge weights
    rc_repeat = seed_main(["--db-url", isolated_step210_db.pg_url])
    assert rc_repeat == 0
    rc_verify = seed_main(["--db-url", isolated_step210_db.pg_url, "--verify-only"])
    assert rc_verify == 0

    with psycopg.connect(isolated_step210_db.pg_url) as conn:
        assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 31
        assert conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == 31
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 406
        assert conn.execute("SELECT COUNT(*) FROM ground_truth_labels").fetchone()[0] == 102
        assert conn.execute("SELECT COUNT(*) FROM entity_edges").fetchone()[0] == 44
        assert conn.execute("SELECT SUM(weight) FROM entity_edges").fetchone()[0] == 455.0
        # M1 demo row remains intact
        assert (
            conn.execute(
                "SELECT external_ref FROM transactions WHERE txn_id = %s",
                (m1_ids["txn_id"],),
            ).fetchone()[0]
            == "AF-M1-DEMO-TXN-001"
        )


def test_pre_t0_fraud_eligibility_and_step29_graph_compatibility(isolated_step210_db) -> None:
    """Verify Step 2.9 fraud eligibility and historical edge reconstruction on seeded manifest."""
    seed_manifest(isolated_step210_db.pg_url)
    eng = sa.create_engine(isolated_step210_db.sa_url)
    try:
        # Before mule_ring confirmation (2025-06-13T23:59:59Z) -> 0 eligible fraud accounts
        pre_conf = fetch_fraud_account_ids(
            eng, snapshot_at=datetime(2025, 6, 10, 0, 0, 0, tzinfo=timezone.utc)
        )
        assert pre_conf == set()

        # At T0 (2025-06-14T00:00:00Z) -> exactly the 6 mule_ring accounts
        t0 = derive_known_fraud_cutoff(load_canonical_manifest())
        at_t0 = fetch_fraud_account_ids(eng, snapshot_at=t0)
        assert len(at_t0) == 6

        # Run Step 2.9 graph job historically at T0: tests exact transaction-backed weight reconstruction
        summary_t0 = run_job(engine=eng, skip_redis=True, snapshot_at=t0)
        assert summary_t0["status"] == "success"
        assert summary_t0["num_fraud_accounts"] == 6
        assert summary_t0["persisted_snapshots"] == 30

        # At full observation cutoff -> all 22 confirmed fraud accounts (zero legitimate_clean)
        at_obs = fetch_fraud_account_ids(eng, snapshot_at=OBSERVATION_CUTOFF)
        assert len(at_obs) == 22
        assert at_t0.issubset(at_obs)
    finally:
        eng.dispose()


def test_seeded_db_pit_parquet_export_and_nullable_ip_paths(isolated_step210_db, tmp_path: Path) -> None:
    """Verify PostgreSQL PIT export preserves 19 canonical features, snapshots, as_of filtering, and IPs."""
    seed_manifest(isolated_step210_db.pg_url)
    eng = sa.create_engine(isolated_step210_db.sa_url)
    try:
        t0 = datetime(2025, 6, 14, 0, 0, 0, tzinfo=timezone.utc)
        t_nov = datetime(2025, 11, 1, 0, 0, 0, tzinfo=timezone.utc)
        t_future_snap = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

        # Compute two historical snapshots plus one future snapshot beyond all transactions
        run_job(engine=eng, skip_redis=True, snapshot_at=t0)
        run_job(engine=eng, skip_redis=True, snapshot_at=t_nov)
        run_job(engine=eng, skip_redis=True, snapshot_at=t_future_snap)

        # Also add 2 extra transactions exercising INET ip_address AND UUID ip_id in PostgreSQL
        ip_uuid = uuid4()
        cust_id = uuid4()
        acct_id = uuid4()
        with psycopg.connect(isolated_step210_db.pg_url) as conn:
            conn.execute(
                "INSERT INTO customers (customer_id, external_ref, full_name, country_code) VALUES (%s, 'EXT-IP-C', 'IP Tester', 'IN')",
                (cust_id,),
            )
            conn.execute(
                "INSERT INTO accounts (account_id, customer_id, account_number, account_type, opened_on) VALUES (%s, %s, 'EXT-IP-ACC', 'card', '2025-01-01')",
                (acct_id, cust_id),
            )
            conn.execute(
                """
                INSERT INTO transactions (txn_id, external_ref, account_id, channel, direction, amount, currency, ip_id, ip_address, occurred_at)
                VALUES
                    (%s, 'EXT-IP-1', %s, 'card', 'debit', 10.00, 'INR', %s, NULL, '2025-12-01T10:00:00+00:00'),
                    (%s, 'EXT-IP-2', %s, 'card', 'debit', 20.00, 'INR', NULL, '203.0.113.99', '2025-12-01T10:05:00+00:00'),
                    (%s, 'EXT-IP-3', %s, 'card', 'debit', 30.00, 'INR', NULL, NULL, '2025-12-01T10:10:00+00:00');
                """,
                (uuid4(), acct_id, ip_uuid, uuid4(), acct_id, uuid4(), acct_id),
            )
            conn.commit()

        full_parquet = tmp_path / "full_features_pit.parquet"
        rc_full = offline_main(["--out", str(full_parquet), "--db-url", isolated_step210_db.sa_url])
        assert rc_full == 0

        full_df = pd.read_parquet(full_parquet)
        assert len(full_df) == 408
        assert list(full_df.columns) == list(PIT_TRACE_COLUMNS) + list(D.FEATURE_LIST)
        assert set(FORBIDDEN_PIT_TARGET_COLUMNS).isdisjoint(full_df.columns)

        # Verify INET host formatting (no /32 suffix) and UUID ip_id coalescing
        ext_rows = full_df[full_df["account_id"] == str(acct_id)].sort_values("ts").reset_index(drop=True)
        assert ext_rows["distinct_ips_24h"].tolist() == [1.0, 2.0, 2.0]
        assert ext_rows["ip_id"].tolist() == [str(ip_uuid), "203.0.113.99", None]

        # Future snapshot 2026-06-01 must NEVER be joined to any historical transaction
        non_null_snaps = full_df["feature_snapshot_at"].dropna()
        assert (non_null_snaps < pd.Timestamp(t_future_snap)).all()
        assert (non_null_snaps <= full_df.loc[non_null_snaps.index, "ts"]).all()

        # Verify as-of T0 filtering drops post-T0 transactions and post-T0 snapshots
        t0_parquet = tmp_path / "t0_features_pit.parquet"
        summary_t0 = export_features_pit(t0_parquet, engine=eng, as_of=t0)
        assert summary_t0["rows"] == 187
        t0_df = pd.read_parquet(t0_parquet)
        assert (t0_df["ts"] <= pd.Timestamp(t0)).all()
        t0_snaps = t0_df["feature_snapshot_at"].dropna()
        assert (t0_snaps <= pd.Timestamp(t0)).all()
    finally:
        eng.dispose()


def test_seeder_partial_failure_rolls_back_and_retries_cleanly(isolated_step210_db, monkeypatch) -> None:
    """Verify transactional rollback leaves DB spotless after mid-run failure and retry succeeds."""
    _insert_m1_demo_record(isolated_step210_db.pg_url)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated mid-transaction failure before commit")

    monkeypatch.setattr("db.seed.seed_known_fraud.verify_persisted_manifest", _boom)
    with pytest.raises(RuntimeError, match="simulated mid-transaction failure"):
        seed_manifest(isolated_step210_db.pg_url)

    # All manifest inserts must be completely rolled back while M1 demo record remains
    with psycopg.connect(isolated_step210_db.pg_url) as conn:
        assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM entity_edges").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM ground_truth_labels").fetchone()[0] == 0

    monkeypatch.undo()
    retry_summary = seed_manifest(isolated_step210_db.pg_url)
    assert retry_summary["reconciled"]["hash_verified"] is True
    assert retry_summary["reconciled"]["transactions"] == EXPECTED_COUNTS["transactions"]
    assert retry_summary["reconciled"]["labels"] == EXPECTED_COUNTS["labels"]
